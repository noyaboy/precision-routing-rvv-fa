# Mo 8 step 2c — intra-tile online softmax chunk

## Headline

**Mo 8 step 2c PASS.** Single-tile online-softmax kernel
`online_softmax_chunk_naive` consumes 16 FP32 scores from a QK^T tile
(produced by step 2b's `qkt_chunk_naive`) and produces 16 FP32 exp
weights (for the downstream P·V pass), a running max scalar, and a
running sum scalar. Five new platform primitives land in the exo
fork: `vfsub.vf`, FP32→BF16 narrow (truncating, gem5-compatible),
vfredmax.vs to DRAM scalar, vfredusum.vs to DRAM scalar, plus the
companion narrow macro. The chunk uses **fmaxf** (math.h extern) as
the max-reduction body operator — confirmed Exo's unification matches
`dst[0] = fmaxf(dst[0], src[i])` against the new vfredmax @instr.

Lowered C body:

```c
void online_softmax_chunk_naive(void *ctxt,
                                const float* S_fp32, float* P_fp32,
                                float* m_out, float* l_out) {
  vfloat32m2_t S_reg;
  vfloat32m2_t S_shifted;
  vuint16m1_t  S_bf16;
  vfloat32m2_t P_reg;
  S_reg     = __riscv_vle32_v_f32m2(&S_fp32[0], 16);
  SATURN_VFREDMAX_F32M2(m_out[0], S_reg, 16);
  S_shifted = __riscv_vfsub_vf_f32m2(S_reg, m_out[0], 16);
  SATURN_F32_NARROW_BF16(&S_bf16, &S_shifted, 16);
  SATURN_VFEXP(&P_reg, &S_bf16, 16);
  __riscv_vse32_v_f32m2(&P_fp32[0], P_reg, 16);
  SATURN_VFREDUSUM_F32M2(l_out[0], P_reg, 16);
}
```

The chunk is the *intra-tile* piece of online softmax — it processes
one 16-lane score vector. Caller is responsible for initializing
`m_out[0] = -inf` and `l_out[0] = 0` before the first tile, then
chaining tiles by reusing the same scalars (each tile's max-reduce
takes the prior max as the running accumulator). The full online
softmax recurrence (cross-tile rescale of l, plus rescale of the O
accumulator) is the §6 outer loop's responsibility and composes
with step 3 (P·V).

## New @instrs and macros (exo fork saturn_rvv.py)

### `SATURN_F32_NARROW_BF16` macro + `saturn_f32_narrow_bf16_m2` @instr

Companion to step 2b's BF16→FP32 widen. Drops the lower 16 mantissa
bits of each FP32 lane:

```asm
vsetvli zero, %2, e32, m2, ta, ma
vle32.v v4, (src)
vsetvli zero, %2, e16, m1, ta, ma
vnsrl.wi v8, v4, 16    # narrow FP32→BF16 with right-shift 16
vse16.v v8, (dst)
```

Truncates (does not round). Matches the precision flow in
`bench_fa_mixed_rvv_native.c`'s softmax path: S is FP32 from QK^T,
narrowed to BF16 to feed `vfexp.v` (which consumes at SEW=16/m1).

### `saturn_vfsub_vf_f32m2` — vector minus DRAM-scalar broadcast

Standard RVV `vfsub.vf` intrinsic; scalar arg is a 1-element f32
DRAM buffer (matches the rest of the step 2c scalar surface). Body:

```python
for i in seq(0, vl):
    dst[i] = src[i] - scalar[0]
```

Used for the softmax max-subtract: `S' = S - m_tile`.

### `saturn_vfredmax_to_dram_f32m2` + `SATURN_VFREDMAX_F32M2` macro

Wraps RVV 1.0 `vfredmax.vs` with scalar-bouncing:

```c
vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1((dst_scalar), 1);
_acc = __riscv_vfredmax_vs_f32m2_f32m1((src), _acc, (vl));
(dst_scalar) = __riscv_vfmv_f_s_f32m1_f32(_acc);
```

The DRAM-scalar surface keeps the @instr signature uniform with the
rest of the platform. Body matches `dst[0] = fmaxf(dst[0], src[i])`
using the **`fmaxf` extern from `exo.libs.externs`** — Exo's
unification matches the high-level @proc's max-accumulator loop
against this @instr body shape.

### `saturn_vfredusum_to_dram_f32m2` + `SATURN_VFREDUSUM_F32M2` macro

Same pattern with `vfredusum.vs` (unordered sum-reduce — faster on
out-of-order Saturn since softmax sums already absorb FP rounding).
Body: `dst[0] += src[i]`.

### GCC 14.2 intrinsic-name gotcha (worth filing in memory)

The "extract scalar from vector lane 0" intrinsic is
`__riscv_vfmv_f_s_f32m1_f32` in GCC 14.2 (note the trailing `_f32`
return-type suffix), **not** `__riscv_vfmv_f_s_f32m1`. The latter
spelling errored as "implicit declaration"; GCC suggested
`__riscv_vfmv_s_f_f32m1` which is the opposite direction (scalar →
element 0). Confirmed correct spelling by grepping
`microbench-fa/bench_fa_mixed_rvv_stub.c`. Same `_f32` suffix
convention applies to other scalar-extract variants
(`vfmv_f_s_f64m1_f64`, etc.).

## Schedule (paper/exo_schedule_fa.py)

Same shape as step 2b — explicit (load, ..., compute, ..., store)
sequence in the @proc; schedule just sets memory + replaces 7 loops
in source order:

```python
def schedule_online_softmax_chunk(p=online_softmax_chunk_naive, ...):
    p = set_memory(p, "S_reg",     SaturnRVV_M2)
    p = set_memory(p, "S_shifted", SaturnRVV_M2)
    p = set_memory(p, "S_bf16",    SaturnRVV_M1)
    p = set_memory(p, "P_reg",     SaturnRVV_M2)
    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vfredmax_to_dram_f32m2)
    p = replace(p, "for i in _: _ #0", saturn_vfsub_vf_f32m2)
    p = replace(p, "for i in _: _ #0", saturn_f32_narrow_bf16_m2)
    p = replace(p, "for i in _: _ #0", vfexp_v)
    p = replace(p, "for i in _: _ #0", saturn_vse32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vfredusum_to_dram_f32m2)
    return p
```

Both reduction loops in the @proc use `fmaxf` / `+=` in the body
(no new Exo idioms beyond what's already supported); unification
matches against the new @instr bodies exactly.

## Cross-compile + disassembly verification

Compiled with `riscv64-linux-gcc 14.2 -O2 -march=rv64gcv -mabi=lp64d`.
Disassembly of `online_softmax_chunk_naive` (full softmax chain
excerpt):

```
000000000000021e <online_softmax_chunk_naive>:
  ...
  222: vsetivli zero,16,e32,m2,ta,ma
  226: vle32.v   v2,(a1)                    # S_reg load
  22a: flw       fa5,0(a3)                  # load m_out[0]
  22e: vsetivli  zero,1,e32,m1,ta,ma
  236: vfmv.v.f  v1,fa5                     # broadcast m_out into m1
  23a: vsetivli  zero,16,e32,m2,ta,ma
  242: vfredmax.vs v1,v2,v1                 # max-reduce ✓
  25e: vfmv.f.s  fa5,v1                     # extract scalar
  264: vfsub.vf  v2,v2,fa5                  # S - max ✓
  28c: fsw       fa5,0(a3)                  # store m_out[0]
  292: vsetvli   zero,a1,e32,m2,ta,ma
  296: vle32.v   v4,(a0)                    # reload S_shifted
  29a: vsetvli   zero,a1,e16,m1,ta,ma
  29e: vnsrl.wi  v8,v4,16                   # FP32→BF16 narrow ✓
  2a2: vse16.v   v8,(sp)
  2b8: vsetvli   zero,a1,e16,m1,ta,ma
  2bc: vle16.v   v8,(sp)
  2c0: 4e831457  .word 0x4e831457           # vfexp.v ✓
  2c8: vse32.v   v8,(a5)
  2d0: vse32.v   v2,(a2)                    # P_fp32 store
  2dc: vl2re32.v v2,(a5)
  2ec: vfmv.v.f  v1,fa5                     # init for vfredusum
  2f0: vsetivli  zero,16,e32,m2,ta,ma
  2f4: vfredusum.vs v2,v2,v1                # sum-reduce ✓
  2f8: vfmv.f.s  fa5,v2                     # extract scalar
  2fc: fsw       fa5,0(a4)                  # store l_out[0]
```

All 5 new primitives fire exactly once in `online_softmax_chunk_naive`:
- `vfredmax.vs v1,v2,v1` at 0x242
- `vfsub.vf v2,v2,fa5` at 0x264
- `vnsrl.wi v8,v4,16` at 0x29e
- `.word 0x4e831457` (vfexp) at 0x2c0
- `vfredusum.vs v2,v2,v1` at 0x2f4

The chain matches the @proc's loop order exactly, with the
scalar-bouncing through `flw`/`fsw` (DRAM ↔ FP register) and
`vfmv.v.f`/`vfmv.f.s` (FP register ↔ m1 vector register lane 0).

## Known-cost inefficiencies (deferred to step 4)

Same family as step 2b's known costs, plus one new one:

- **Scalar DRAM-bouncing**: every reduction does `flw` → `vfmv.v.f`
  → `vfredmax/vfredusum` → `vfmv.f.s` → `fsw`. The `fsw` writes the
  result back to the DRAM scalar — wasteful when the next `vfsub.vf`
  immediately reads it. A step-4 optimization is to add @instr
  variants that hold the scalar in a SaturnRVV_M1 register across
  multiple operations, then bulk-store at the end of the tile.
- **vsetvli churn** (carried over from step 2b): each macro flips
  SEW/LMUL. The softmax chain alone fires 6+ vsetvlis.
- **Stack-spill through asm clobber zones** (carried over from
  step 2b): the narrow and vfexp macros both spill through stack.

All three are addressable by either (a) intrinsic-based @instr
templates with shared clobber sets, or (b) a peephole pass that
merges adjacent macros sharing SEW/LMUL. Step 4 (cycle parity vs
`bench_fa_mixed_rvv_native`) will quantify the cost.

## Open Mo 8 substeps

- **3.** Wire remaining 2 vfconv lanes: `vfconv.bf16.fp8.v` for
  P-quant (BF16 attention weights → FP8 for downstream P·V matmul)
  and `vfconv.fp8.bf16.v` for P·V dequant (FP8 P → BF16 for the
  vfmacc accumulator). Both @instrs already declared in
  `exo/src/exo/platforms/saturn_rvv.py` (5.2, 5.3) — step 3 just
  wires them into a `pv_chunk_naive` @proc.
- **4.** Build the full Exo-generated kernel on gem5 and compare
  cycles to `bench_fa_mixed_rvv_native`. Target: within 10% (Mo 8
  PASS). Address the known-cost inefficiencies if the cycle delta
  is large.

## Reproducibility

```
cd /home/noah/project/riscv
pip install -e exo/                        # picks up new @instrs in fork
python3 paper/exo_schedule_fa.py           # 6/6 schedules; 14 markers
```

Disassembly probe:

```
cd /tmp/mo8s1
python3 -c "
import sys; sys.path.insert(0, '/home/noah/project/riscv')
sys.path.insert(0, '/home/noah/project/riscv/exo/src')
from exo.API import compile_procs_to_strings
from paper.exo_schedule_fa import (schedule_dequant_chunk,
    schedule_softmax_exp_chunk, schedule_dequant_64,
    schedule_fa_dequant_per_row, schedule_qkt_chunk,
    schedule_online_softmax_chunk)
c, h = compile_procs_to_strings(
    [schedule_dequant_chunk(), schedule_softmax_exp_chunk(),
     schedule_dequant_64(), schedule_fa_dequant_per_row(),
     schedule_qkt_chunk(), schedule_online_softmax_chunk()],
    'exo_schedule_fa.h')
open('exo_schedule_fa.c','w').write('#include \"exo_schedule_fa.h\"\n'+c)
open('exo_schedule_fa.h','w').write(h)
"
/tmp/bootlin-14/riscv64-lp64d--glibc--bleeding-edge-2024.05-1/bin/riscv64-linux-gcc \
    -O2 -march=rv64gcv -mabi=lp64d -c exo_schedule_fa.c -o exo_schedule_fa.o
/tmp/bootlin-14/riscv64-lp64d--glibc--bleeding-edge-2024.05-1/bin/riscv64-linux-objdump \
    -d exo_schedule_fa.o | sed -n '/<online_softmax_chunk_naive>:/,/<qkt_chunk_naive>:/p'
```
