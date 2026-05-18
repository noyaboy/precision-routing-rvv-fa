# GCC bug report — RVV vsetvli pass picks wrong vtype for widening-intrinsic chain at `-O2`

**Status**: Draft. Ready to file at https://gcc.gnu.org/bugzilla/. Two
related issues now documented:

- **Part 1** (GCC 13.x): pure-intrinsic widening chain miscompiles —
  backport request against the 13 branch (the bug was fixed in
  14.2.0 for the no-asm-volatile case).
- **Part 2** (GCC 14.2.x, possibly 15.x): the 14.2 fix is **partial**
  — the same wrong-vtype emission still happens when an asm-volatile
  block precedes the widening chain. Discovered 2026-05-18 during the
  Mo 8 step 4d-1 intrinsic-rewrite work; section added at the end of
  this report.

Submit under your own name. Suggested component: `target` → `riscv`.
Suggested severity: `normal` (silent miscompile, not crash).

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
| 14.2.0 (pure intrinsic)  | Bootlin 2024.05-1 (Buildroot)   | **FIXED** — emits 1 vsetvli at e32 m2; correct |
| 14.2.0 (after asm-volatile) | Bootlin 2024.05-1 (Buildroot) | **STILL BUGGY** — see "Part 2" section below; the asm-volatile defeats GCC 14.2's vsetvli demand-set logic |
| 15.1.0 (pure intrinsic)  | Bootlin 2025.08-1 (Buildroot)   | **FIXED** — same correct emission as 14.2 (asm-volatile case not yet tested) |

All four were tested with the relevant reproducer below on
the same host (Ubuntu 20.04, x86_64). For the pure-intrinsic case,
the fix landed somewhere in the GCC 14 development cycle — the
GCC 13 branch is the only one still affected. For the
asm-volatile-preceded case (Part 2 below), GCC 14.2 still requires
a workaround.

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

---

# Part 2 — GCC 14.2 partial-fix: vsetvli pass still wrong after asm-volatile

**Discovered 2026-05-18** during Mo 8 step 4d-1 of the same project
(intrinsic-rewrite optimisation of a hand-coded Saturn-FU
asm-volatile macro to a standard-RVV intrinsic chain). The pure-
intrinsic widening pattern from Part 1 now compiles correctly at
GCC 14.2, but when a preceding `asm volatile (...)` block has
internally changed `vtype` (e.g., a custom-instruction macro that
issues its own `vsetvli`) and then exits, GCC 14.2's vsetvli pass
**fails to recognise that vtype has been clobbered** and emits no
vsetvli before the subsequent widening intrinsic. Result: the
identical miscompile as Part 1.

## Minimal reproducer (Part 2)

**Important**: this reproducer triggers the bug only when the
widening chain is inside a loop (the production case). The
straight-line, single-function-body form gets the correct vsetvli
emitted by GCC 14.2's local liveness analysis. The bug is
specifically that the *loop-hoisted* unified vtype gets clobbered
by the asm-volatile inside the loop body, but the demand-set pass
fails to re-emit a vsetvli per iteration. Verified 2026-05-18
with `riscv64-linux-gcc.br_real (Buildroot 2021.11-12449-g1bef613319) 14.2.0`.

```c
#include <riscv_vector.h>
#include <stdint.h>

/* Stand-in for a Saturn custom: opaque .4byte inside asm-volatile
 * that internally changes vtype to e16, m1. The .4byte 0x4E049057
 * is the real Saturn vfconv.nvfp4.bf16.v encoding; whether the
 * opcode is real is irrelevant — GCC sees only the bits. */
#define SATURN_VFCONV_M1(dst, src, vl) do {                        \
    asm volatile (                                                 \
      "vsetvli zero, %2, e16, m1, ta, ma\n\t"                      \
      "vle16.v v0, (%1)\n\t"                                       \
      ".4byte 0x4E049057\n\t"                                      \
      "vse16.v v0, (%0)\n\t"                                       \
      :: "r"(dst), "r"(src), "r"((size_t)(vl))                     \
      : "memory", "v0");                                           \
} while (0)

void widen_loop(const uint16_t *src_packed,
                uint16_t       *bf16_stash,
                float          *dst_f32,
                size_t          n_iters,
                size_t          vl) {
    for (size_t i = 0; i < n_iters; i++) {
        SATURN_VFCONV_M1(bf16_stash, src_packed + i*16, vl);
        /* widening chain that needs e32, m2: */
        vuint16m1_t a    = __riscv_vle16_v_u16m1(bf16_stash, vl);
        vuint32m2_t aw   = __riscv_vzext_vf2_u32m2(a, vl);
        vuint32m2_t shl  = __riscv_vsll_vx_u32m2(aw, 16, vl);
        vfloat32m2_t bf  = __riscv_vreinterpret_v_u32m2_f32m2(shl);
        __riscv_vse32_v_f32m2(dst_f32 + i*16, bf, vl);
    }
}
```

## Output: GCC 14.2 at `-O2` (BUG)

```asm
widen_loop:
    beqz    a3, .L8
    li      a5, 0
    vsetvli zero, a4, e32, m2, ta, ma       # hoisted OUTSIDE loop
.L3:
    vsetvli zero, a4, e16, m1, ta, ma       # asm-volatile body — clobbers vtype
    vle16.v v0, (a0)
    .4byte  0x4E049057
    vse16.v v0, (a1)
    vle16.v v1, (a1)                        # EEW=16 baked in (OK)
    addi    a5, a5, 1
    addi    a0, a0, 32
    vzext.vf2 v2, v1                        # runs at e16, m1 (BUG: needs e32, m2)
    vsll.vi   v2, v2, 16
    vse32.v   v2, (a2)
    addi    a2, a2, 64
    bne     a3, a5, .L3
.L8:
    ret
```

The vsetvli on entry sets `e32, m2` once outside the loop. Inside,
the asm-volatile's `vsetvli ... e16, m1` clobbers it every iteration,
and the demand-set pass never re-emits a fresh `vsetvli zero, ..., e32, m2`
before the `vzext.vf2`. Result: vzext.vf2 reads source EEW=8 instead
of EEW=16 — identical end-state miscompile as Part 1 of this report.

## Workarounds tested

1. **`(void)__riscv_vsetvl_e32m2(vl);`** — the explicit vsetvl
   intrinsic call returns a `size_t` we don't use, so GCC's DCE
   eliminates the call entirely. The vsetvli is **NOT emitted**.
   Verified by inspecting the emitted C and the disassembly.

2. **`asm volatile ("vsetvli zero, %0, e32, m2, ta, ma" ::
   "r"((size_t)(vl)));`** — single-instruction asm-volatile.
   GCC cannot elide this, so the vsetvli IS emitted. **This is
   the workaround we deploy in production**:

   ```c
   asm volatile ("vsetvli zero, %0, e32, m2, ta, ma"
                 :: "r"((size_t)(vl)));
   vuint16m1_t a = __riscv_vle16_v_u16m1(bf16_stash, vl);
   vuint32m2_t b = __riscv_vzext_vf2_u32m2(a, vl);
   __riscv_vse32_v_u32m2(dst, b, vl);
   ```

   Cost: 1 extra `vsetvli` per widening site. In the Mo 8 step
   4d-1 kernel this was ~250 K cycles of explicit-barrier overhead
   on top of an otherwise-clean intrinsic chain at L2K
   (cf. `paper/mo8_step4d1_results.md`).

3. **Opacity barrier on `vl`** (Part 1's workaround,
   `asm volatile ("" : "+r"(vl));`) — does **NOT** fix the Part 2
   case. The barrier is between the explicit vsetvl call and the
   vle16, but the issue is that the explicit vsetvl call itself
   gets DCE'd in the asm-volatile-preceded case.

## Where the bug bites — diagnosis

GCC 14.2's vsetvli demand-set pass needs to recognise that an
`asm volatile` block with no explicit declaration of the VTYPE
register as clobbered may *still* clobber VTYPE if the body
contains a `vsetvli` instruction. The pass currently appears to
treat such asm-volatile blocks as VTYPE-preserving.

Two possible remediation paths:

a. **Conservative**: treat every `asm volatile` block as a VTYPE
   barrier (forces a vsetvli before any subsequent vector op
   that doesn't have a baked-in EEW). Slight cost on legitimate
   non-vtype-modifying asm-volatile blocks, but correctness-safe.

b. **Precise**: scan the asm-volatile body for `vsetvli` (or
   `vsetivli`) and only treat as a VTYPE barrier if found. More
   complex but no false-positive vsetvli emissions.

Either path would close Part 2.

## Impact

Same impact class as Part 1: silent miscompile, no warning,
common RVV widening idiom. The Part 2 case is particularly
relevant for **co-design projects mixing custom-instruction
asm-volatile macros with standard-RVV intrinsics** — exactly
the Mo 8 step 4d-1 scenario where we hit it. Project-custom
RVV instructions (e.g., the Berkeley Saturn `.4byte` customs
in our case) cannot be expressed as GCC intrinsics, so they
must go through asm-volatile, putting them in direct conflict
with the surrounding intrinsic-based scheduling.

## Suggested action for the GCC maintainers (Part 2)

Treat asm-volatile blocks as VTYPE barriers in the vsetvli
demand-set pass (option (a) above is the safest, least-invasive
fix). Until then, document the workaround (option 2 above) as a
known interaction in the GCC RISC-V backend's vsetvli pass docs
so users mixing intrinsics and asm-volatile know to deploy it
defensively.
