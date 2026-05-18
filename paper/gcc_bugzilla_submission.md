# GCC Bugzilla submission — user-action checklist

Two bugs to file. Both are silent miscompiles in the RVV `vsetvli`
demand-set pass at `-O2 -march=rv64gcv`. Full technical detail and
reproducers in `paper/gcc_bug_report.md`.

## Prerequisites

1. Bugzilla account at https://gcc.gnu.org/bugzilla/. Sign up if you
   don't have one — takes a minute, no review.
2. Minimal reproducers are already prepared at
   `paper/gcc_repro/minimal.c` (Part 1) and
   `paper/gcc_repro/minimal_part2.c` (Part 2). If you don't see those
   files, the protocol section below has both inline.

## Bug #1 — GCC 13.2 vsetvli pass picks wrong vtype for widening chain (backport request)

**Click "New" at the top of the Bugzilla page.**

| Field                | Value                                                         |
|----------------------|---------------------------------------------------------------|
| Product              | `gcc`                                                         |
| Component            | `target`                                                      |
| Version              | `13.2.0`                                                      |
| Severity             | `normal` (silent miscompile, not crash)                       |
| Hardware             | `Other` (cross-target)                                        |
| OS                   | `Linux` (host)                                                |
| Summary (title)      | `[13 regression?] RVV vsetvli pass picks SEW=16 for vle16→vzext.vf2 widening chain at -O2 (silent miscompile, fixed in 14.2)` |

**Description** (paste verbatim):

```
GCC 13.2 at -O2 -march=rv64gcv silently miscompiles a common RVV
widening idiom: __riscv_vle16_v_u16m1 followed by
__riscv_vzext_vf2_u32m2. The optimizer collapses the explicit
__riscv_vsetvl_e32m2(n) into the vle16's implicit vsetvli, picks
SEW=16 LMUL=m1 as the unified vtype, and runs vzext.vf2 under
that vtype. At SEW=16 vzext.vf2 zero-extends u8 (SEW/2) elements
from the source, not the u16 elements the preceding vle16 just
loaded. Result: wrong-width source read, silent wrong values
downstream.

CONFIRMED FIXED in GCC 14.2.0 and 15.1.0, which both pick SEW=32
LMUL=m2 (the widest demanded vtype) and run vle16.v under it with
its baked-in EEW=16. The fix needs backporting to GCC 13.

Minimal reproducer (7 lines):

  #include <riscv_vector.h>
  #include <stdint.h>
  void widen(const uint16_t *src, uint32_t *dst, size_t n) {
      size_t vl = __riscv_vsetvl_e32m2(n);
      vuint16m1_t a = __riscv_vle16_v_u16m1(src, vl);
      vuint32m2_t b = __riscv_vzext_vf2_u32m2(a, vl);
      __riscv_vse32_v_u32m2(dst, b, vl);
  }

Build:
  riscv64-linux-gcc -O2 -march=rv64gcv -mabi=lp64d -c minimal.c

Buggy output (GCC 13.2 -O2):
  vsetvli zero, a2, e16, m1, ta, ma  ; only one vsetvli
  vle16.v v26, (a0)
  vzext.vf2 v24, v26                 ; source EEW=8 — BUG
  vse32.v v24, (a1)

Correct output (GCC 14.2 / 15.1 -O2):
  vsetvli zero, a2, e32, m2, ta, ma  ; picks widest SEW in chain
  vle16.v v1, (a0)                   ; EEW=16 baked in; EMUL=m1
  vzext.vf2 v2, v1                   ; source EEW=16 — correct
  vse32.v v2, (a1)

Correct output (GCC 13.2 -O0):
  ; all four vsetvli emitted — confirms bug is in optimization pass

Workaround on GCC 13:
  size_t vl = __riscv_vsetvl_e32m2(n);
  asm volatile ("" : "+r"(vl));   /* opacity barrier */
  /* rest as above */

Compiler:
  riscv64-linux-gcc.br_real (Buildroot 2021.11-11272-ge2962af) 13.2.0
  Target: riscv64-buildroot-linux-gnu
  --with-arch=rv64imafd_zicsr_zifencei --with-abi=lp64d

Impact: hit during fused-attention mixed-precision RVV development
for the Berkeley Saturn vector unit; silent wrong values throughout
the QK^T dot product. Diagnosis required disassembly + reference
comparison — the wrong values fell within plausible numeric range.

Suggested fix: backport the GCC-14-cycle commit that fixed the
vsetvli demand-set pass for this case. If backport is out of scope,
document as a known limitation of the 13-branch vsetvli pass.
```

Attach `paper/gcc_repro/minimal.c` if Bugzilla offers an upload
button (look for "Add an attachment" at the bottom of the form).

---

## Bug #2 — GCC 14.2 vsetvli pass: loop-hoisted vtype not re-emitted after asm-volatile vtype-clobber

**Click "New" again** (separate report; Bugzilla discourages
multi-issue tickets).

| Field                | Value                                                         |
|----------------------|---------------------------------------------------------------|
| Product              | `gcc`                                                         |
| Component            | `target`                                                      |
| Version              | `14.2.0`                                                      |
| Severity             | `normal` (silent miscompile, not crash)                       |
| Hardware             | `Other`                                                       |
| OS                   | `Linux`                                                       |
| Summary (title)      | `[14/15] RVV vsetvli demand-set: loop-hoisted vtype not re-emitted after asm-volatile clobber — wrong-SEW miscompile` |

**Description** (paste verbatim):

```
GCC 14.2 at -O2 -march=rv64gcv silently miscompiles a widening
intrinsic chain inside a loop when the same loop body contains an
asm-volatile block whose body internally executes a vsetvli that
changes vtype.

The 14.2 release fixed the *pure-intrinsic* widening miscompile from
GCC 13 (filed separately for backport). The fix is partial: the
demand-set pass hoists a single vsetvli to the loop preheader for the
loop's unified vtype, but does not recognize that an asm-volatile
block inside the loop body containing a `vsetvli` instruction
clobbers vtype. The subsequent widening intrinsic in the loop body
runs under the asm-volatile's residual vtype, producing the same
wrong-SEW miscompile as the GCC-13 widening-chain bug.

IMPORTANT: the bug reproduces ONLY in the loop case. A straight-line
function body with one asm-volatile vsetvli followed by one widening
intrinsic gets the correct vsetvli emitted by GCC 14.2's local
liveness analysis. The loop hoisting is what creates the gap.

Minimal reproducer (verified 2026-05-18 against
riscv64-linux-gcc.br_real (Buildroot 2021.11-12449-g1bef613319) 14.2.0
from Bootlin 2024.05-1 toolchain):

  #include <riscv_vector.h>
  #include <stdint.h>

  /* Stand-in for a Saturn vector custom: opaque .4byte inside
   * asm-volatile that internally changes vtype to e16, m1. The
   * .4byte opcode value is arbitrary — GCC sees only the bits. */
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

Build:
  riscv64-linux-gcc -O2 -march=rv64gcv -mabi=lp64d -c minimal_part2.c

Buggy output (GCC 14.2 -O2, observed disassembly):
  widen_loop:
      beqz    a3, .L8
      li      a5, 0
      vsetvli zero, a4, e32, m2, ta, ma   ; hoisted OUTSIDE loop
  .L3:
      vsetvli zero, a4, e16, m1, ta, ma   ; asm-volatile clobbers vtype
      vle16.v v0, (a0)
      .4byte  0x4E049057
      vse16.v v0, (a1)
      vle16.v v1, (a1)                    ; EEW=16 baked in (OK)
      addi    a5, a5, 1
      addi    a0, a0, 32
      vzext.vf2 v2, v1                    ; runs at e16, m1 — BUG
      vsll.vi   v2, v2, 16
      vse32.v   v2, (a2)
      addi    a2, a2, 64
      bne     a3, a5, .L3
  .L8:
      ret

The vsetvli on loop entry sets e32, m2 once. Inside, the asm-volatile's
inner vsetvli changes vtype to e16, m1 each iteration, and the
demand-set pass never re-emits a fresh vsetvli e32, m2 before the
vzext.vf2 — which then reads source EEW=8 instead of EEW=16.

Workaround (1 extra vsetvli per widening site, deployed in production):
  asm volatile ("vsetvli zero, %0, e32, m2, ta, ma"
                :: "r"((size_t)(vl)));
  vuint16m1_t a = __riscv_vle16_v_u16m1(bf16_stash, vl);
  vuint32m2_t b = __riscv_vzext_vf2_u32m2(a, vl);
  ...

Note: a (void)__riscv_vsetvl_e32m2(vl); statement does NOT work —
GCC's DCE eliminates the call because the size_t return is unused.
The asm-volatile form is required to force emission.

Compiler:
  riscv64-linux-gcc.br_real (Buildroot 2021.11-12449-g1bef613319) 14.2.0
  Target: riscv64-buildroot-linux-gnu
  Toolchain: Bootlin riscv64-lp64d--glibc--bleeding-edge-2024.05-1

Diagnosis: the vsetvli demand-set pass treats asm-volatile blocks as
VTYPE-preserving when computing the loop-hoisted unified vtype. A
vsetvli instruction inside the asm-volatile body genuinely changes
the architectural vtype register at run time, but the pass doesn't
track it.

Suggested remediation (least-invasive):
  Treat every asm-volatile block as a VTYPE barrier when hoisting
  the loop-unified vtype — emit a fresh vsetvli inside the loop
  before any subsequent vector op that doesn't have a baked-in EEW.
  Minor cost on legitimate vtype-preserving asm-volatile blocks;
  correctness-safe.

Suggested remediation (precise):
  Scan asm-volatile bodies for vsetvli / vsetivli and only treat as
  a barrier if found.

Impact: silent miscompile of a common RVV widening idiom in co-
design projects mixing custom-instruction asm-volatile (e.g.,
Berkeley Saturn .4byte customs) with standard-RVV intrinsics inside
tight loops. Such projects must use asm-volatile for the customs
(no GCC intrinsic surface), putting them in direct conflict with
the intrinsic-based vsetvli scheduling.

Same impact class as the GCC-13 widening-chain bug filed
separately — same end-state miscompile, different code path.
```

Attach `paper/gcc_repro/minimal_part2.c` if upload available.

---

## After submitting

1. Copy each Bugzilla URL (e.g., `https://gcc.gnu.org/bugzilla/show_bug.cgi?id=XXXXX`)
2. Add them at the top of `paper/gcc_bug_report.md` under a new
   "## Filed as" section (line ~5).
3. Add the URLs to the paper draft §7.7 (the GCC-bug paragraph) and
   to the README's Status section.
4. The Bugzilla notifications will tell you the next time anyone
   comments. Expect 1–4 weeks for first triage.

## Minimal `git add` after URLs land

```
git add paper/gcc_bug_report.md paper/paper_draft.md README.md
git commit -m "Paper: add GCC Bugzilla bug IDs"
git push origin main
```
