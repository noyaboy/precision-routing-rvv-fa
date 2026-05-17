# GCC bug report — RVV vsetvli pass picks wrong vtype for widening-intrinsic chain at `-O2` (GCC 13.x only)

**Status**: Draft. Ready to file at https://gcc.gnu.org/bugzilla/ as a
**backport request** against the 13 branch — the bug was fixed in
GCC 14.2.0 (verified, see "Cross-version status" below). Submit under
your own name. Suggested component: `target` → `riscv`. Suggested
severity: `normal` (silent miscompile, not crash).

---

## Summary

GCC 13.2 (Buildroot riscv64) at `-O2 -march=rv64gcv` picks the wrong
unified vtype for a chain of RVV intrinsics that mix EEW-baked-in
loads with SEW-dependent widening ops. Specifically, for
`__riscv_vle16_v_u16m1` followed by `__riscv_vzext_vf2_u32m2`, the
optimizer selects vtype `e16,m1` (the SEW the `vle16` would set on its
own) and runs the entire chain at that vtype. `vzext.vf2` at SEW=16
silently reads source EEW=8 (= SEW/2) elements — half the data the
preceding `vle16` loaded. Downstream code that depends on the widened
result computes on garbage.

The bug does **not** reproduce at `-O0` (which emits the correct
four-vsetvli sequence: `e32,m2 → e16,m1 → e32,m2 → e32,m2`) and
**does not reproduce in GCC 14.2.0 or 15.1.0**, both of which pick
vtype `e32,m2` and correctly run `vle16.v` under it (the load's
baked-in EEW=16 is independent of vtype SEW; the EMUL derivation
`EEW/SEW · LMUL = 16/32 · 2 = m1` gives the correct destination
group). GCC 13 is the only branch that needs a fix or backport.

## Minimal reproducer

```c
/* 7 lines of body — full repro. */
#include <riscv_vector.h>
#include <stdint.h>

void widen(const uint16_t *src, uint32_t *dst, size_t n) {
    size_t vl = __riscv_vsetvl_e32m2(n);
    vuint16m1_t a = __riscv_vle16_v_u16m1(src, vl);
    vuint32m2_t b = __riscv_vzext_vf2_u32m2(a, vl);
    __riscv_vse32_v_u32m2(dst, b, vl);
}
```

## Build command

```
riscv64-linux-gcc -O2 -march=rv64gcv -mabi=lp64d -c minimal.c -o minimal.o
```

## Compiler / target

```
riscv64-linux-gcc.br_real (Buildroot 2021.11-11272-ge2962af) 13.2.0   ← BUGGY
Target: riscv64-buildroot-linux-gnu
--with-arch=rv64imafd_zicsr_zifencei --with-abi=lp64d
```

## Cross-version status

| GCC version              | Source                          | Status at `-O2` |
|--------------------------|---------------------------------|------------------|
| 13.2.0                   | Bootlin 2024.02-1 (Buildroot)   | **BUGGY** — emits 1 vsetvli at e16 m1; vzext.vf2 reads wrong-width source |
| 14.2.0                   | Bootlin 2024.05-1 (Buildroot)   | **FIXED** — emits 1 vsetvli at e32 m2; correct |
| 15.1.0                   | Bootlin 2025.08-1 (Buildroot)   | **FIXED** — same correct emission as 14.2 |

All three were tested with the identical 7-line reproducer below on
the same host (Ubuntu 20.04, x86_64). The fix landed somewhere in
the GCC 14 development cycle — the GCC 13 branch is the only one
still affected.

## Output: GCC 13.2 at `-O2` (BUG)

```asm
vsetvli zero, a2, e16, m1, ta, ma     # only one vsetvli emitted
vle16.v v26, (a0)
vzext.vf2 v24, v26                    # at SEW=16, source EEW=8 (BUG)
vse32.v v24, (a1)                     # vl still 16, but data wrong
```

The explicit `__riscv_vsetvl_e32m2(n)` call appears to have been
folded into the `vle16`'s `vsetvli`, and no transition is re-emitted
for the `vzext.vf2` that needs SEW=32 LMUL=m2.

## Output: GCC 14.2 / 15.1 at `-O2` (CORRECT — what GCC 13 should backport)

```asm
vsetvli zero, a2, e32, m2, ta, ma     # picks the WIDEST SEW in the chain
vle16.v v1, (a0)                      # EEW=16 baked in; EMUL=m1 at e32-m2
vzext.vf2 v2, v1                      # at SEW=32, source EEW=16: CORRECT
vse32.v v2, (a1)
```

Newer GCC picks `e32,m2` as the unified vtype (the widest SEW any op
in the chain demands — set by `vzext.vf2`'s destination need); the
`vle16.v` runs under that vtype with its own baked-in EEW=16, which
the RVV 1.0 spec defines as independent of vtype SEW (effective EMUL =
`EEW/SEW · LMUL` = `16/32 · 2` = m1). This is the natural correct
fix, and GCC 13's optimizer should be backported to do the same.

## Output: GCC 13.2 at `-O0` (CORRECT, no optimizer involved)

```asm
vsetvli zero, a2, e32, m2, ta, ma     # explicit __riscv_vsetvl_e32m2(n)
vsetvli zero, a5, e16, m1, ta, ma     # for vle16 (EEW=16, EMUL=m1)
vle16.v v26, (a0)
vsetvli zero, a5, e32, m2, ta, ma     # SWITCH BACK for vzext.vf2 dest
vzext.vf2 v24, v26
vsetvli zero, a5, e32, m2, ta, ma     # for vse32 (or kept from above)
vse32.v v24, (a1)
```

(Confirms the bug is in GCC 13's optimization pass, not in the
front-end intrinsic lowering.)

## Semantics at the wrong SEW

`vzext.vf2` is defined as: dest EEW = SEW, source EEW = SEW/2, source
EMUL = LMUL/2. At SEW=16 LMUL=m1, the instruction zero-extends *u8*
elements from the low bytes of `v26` into u16 elements in `v24`. The
preceding `vle16` loaded u16 elements, so the bytewise reinterpretation
produces wrong values.

## Impact / how we hit this

We hit this in a fused mixed-precision flash-attention kernel for the
Berkeley Saturn RVV vector unit (mixed-precision LLM inference,
MLArchSys 2027 submission). The kernel's QK^T dot product widens BF16 Q
to FP32 via `vle16 → vzext.vf2 → vsll.vi → vfmacc.vv`. The widening
chain ran at SEW=16 instead of SEW=32, producing garbage attention
scores throughout. The bug was masked initially by a checksum that fell
in plausible-looking range (−0.045 vs reference −20.27, ~0.2 % of
expected magnitude); diagnosis required disassembling the inner loop
and tracking the missing `vsetvli`.

Workaround (kernel side): pre-widen the BF16 input to FP32 at
initialization so the inner loop never crosses the SEW boundary. This
sidesteps the bug at minor scratch-buffer cost.

## Why this matters

The bug is subtle (silent miscompile, no warning), affects a common
RVV widening idiom (`vle16 → vzext.vf2` for BF16 widen-via-bit-shift,
needed because RVV 1.0 doesn't have a BF16-to-FP32 widening instruction
without Zvfbfwma), and is invisible to tests that don't compare against
a numeric reference.

## Additional context — workaround confirmed

Inserting an inline-asm opacity barrier on `vl` between the
`__riscv_vsetvl_e32m2` call and the `vle16` *does* fix the bug:

```c
size_t vl = __riscv_vsetvl_e32m2(n);
asm volatile ("" : "+r"(vl));   /* opacity barrier */
vuint16m1_t a = __riscv_vle16_v_u16m1(src, vl);
vuint32m2_t b = __riscv_vzext_vf2_u32m2(a, vl);
__riscv_vse32_v_u32m2(dst, b, vl);
```

With the barrier in place the emitted code is:

```asm
vsetvli a2, a2, e32, m2, ta, mu       # from __riscv_vsetvl_e32m2
vsetvli zero, a2, e32, m2, ta, ma     # second emission (mu→ma)
vle16.v v26, (a0)                     # EEW=16 baked in, runs correctly under SEW=32
vzext.vf2 v24, v26                    # at SEW=32 — correct
vse32.v v24, (a1)
```

(The `vle16.v` reads 16 × 16-bit elements regardless of vtype SEW per
the RVV 1.0 spec; effective EMUL = EEW/SEW · LMUL = 16/32 · 2 = 1
= m1 group, so `v26` correctly holds 16 u16 elements after the load.
`vzext.vf2` then reads those 16 u16 elements via its EEW=SEW/2=16
source. Correct.)

This strongly suggests the bug stems from the optimizer
constant-propagating `vl` and then concluding that the explicit
`__riscv_vsetvl_e32m2(n)` is dead (since the `vle16` will set its own
`vsetvli` with the same `vl` value at a different SEW). The optimizer
then fails to recognize that downstream intrinsics
(`vzext_vf2_u32m2`, etc.) demand the e32 m2 vtype it just eliminated.

## Files

`/tmp/gcc_repro/minimal.c` — the 7-line reproducer above.

`/tmp/gcc_repro/minimal_opaque.c` — same with the opacity-barrier
workaround.

Compiled outputs (objdump excerpts above):
- `minimal.o`        (GCC 13.2 -O2): only one `vsetvli` at e16 m1; `vzext.vf2` wrong-SEW.
- `minimal_O0.o`     (GCC 13.2 -O0): all four `vsetvli` correctly emitted.
- `minimal_opaque.o` (GCC 13.2 -O2 + asm barrier): `vsetvli` correctly emitted.
- `minimal_g14.o`    (GCC 14.2.0 -O2): correct e32-m2 emission, no bug.
- `minimal_new.o`    (GCC 15.1.0 -O2): correct e32-m2 emission, no bug.

## Suggested action for the GCC maintainers

The bug is silently miscompiled output on the 13 branch. Suggested
remediation: identify the GCC 14 commit that fixed the vsetvli
demand-set logic for this case and backport to the active 13 branch
(13.4 / 13.5 if applicable). Failing that, document as a known
limitation of the 13-branch vsetvli pass so users on GCC 13.x can
deploy the inline-asm-barrier workaround pre-emptively in
SEW-changing intrinsic chains.
