# Mo 8 step 2b — QK^T inner-product chunk with dequant chain

## Headline

**Mo 8 step 2b PASS.** First demo in the methodology that adds *real
arithmetic* (vfmacc.vv), not just opaque copy + precision conversion.
A `qkt_chunk_naive` @proc captures one 16-lane head_dim chunk of the
§6 QK^T body — load Q (FP32), load K (NVFP4 carrier), dequant K to
BF16 via the Saturn vfconv lane, widen K to FP32, load S accumulator,
fp32 vfmacc, store S. Two new platform @instrs land in the exo fork:
`saturn_bf16_widen_f32_m2` (vzext.vf2 + vsll.vi 16) and
`saturn_vfmacc_vv_f32m2` (standard RVV intrinsic). The schedule is a
straight set_memory + 7 replace() chain.

Lowered C body:

```c
void qkt_chunk_naive(void *ctxt, const float* Q_fp32,
                     const uint16_t* K_nvfp4, float* S_acc) {
  vfloat32m2_t Q_reg;
  vuint16m1_t  K_nvfp4_reg;
  vuint16m1_t  K_bf16_reg;
  vfloat32m2_t K_fp32_reg;
  vfloat32m2_t S_reg;
  Q_reg       = __riscv_vle32_v_f32m2(&Q_fp32[0], 16);
  K_nvfp4_reg = __riscv_vle16_v_u16m1(&K_nvfp4[0], 16);
  SATURN_VFCONV_NVFP4_BF16(&K_bf16_reg, &K_nvfp4_reg, 16);
  SATURN_BF16_WIDEN_F32(&K_fp32_reg, &K_bf16_reg, 16);
  S_reg       = __riscv_vle32_v_f32m2(&S_acc[0], 16);
  S_reg       = __riscv_vfmacc_vv_f32m2(S_reg, Q_reg, K_fp32_reg, 16);
  __riscv_vse32_v_f32m2(&S_acc[0], S_reg, 16);
}
```

The chunk produces 16 per-lane partial sums (one per lane), not a
single inner-product scalar — folding to one S[i, j] cell of the
score matrix needs `vfredsum.vs`, which lands in step 2c (online
softmax also uses reductions for the max/sum scans). Q is assumed
pre-widened to FP32 outside this chunk, matching
`bench_fa_mixed_rvv_native.c`'s approach (per Track J-4.5b: GCC 14.2
+ `-fno-tree-vectorize` + pre-widen-Q-at-init workaround).

## New @instrs in the exo fork (saturn_rvv.py)

### `SATURN_BF16_WIDEN_F32` macro

gem5 25.1.0.1 SE mode lacks Zvfbfmin so `vfwcvtbf16.f.f.v` isn't
available. The widen is open-coded as:

```asm
vsetvli zero, %2, e16, m1, ta, ma
vle16.v v8, (src)               # load 16 BF16 (ui16 carrier)
vsetvli zero, %2, e32, m2, ta, ma
vzext.vf2 v10, v8               # zero-extend each ui16 -> ui32
vsll.vi v10, v10, 16            # shift left 16 bits (BF16 -> FP32)
vse32.v v10, (dst)              # store 16 FP32
```

The exact path used by `bench_fa_mixed_rvv_native.c`'s
`bf16_load_widen` helper. Correctness: BF16 == sign + exp8 +
mantissa7; FP32 == sign + exp8 + mantissa23. Shifting left by 16
zeros the low 16 mantissa bits and places sign/exp/upper-mantissa in
the right FP32 positions. Exact for normals, subnormals, zero, Inf,
and NaN.

### `saturn_bf16_widen_f32_m2` @instr

```python
@instr("SATURN_BF16_WIDEN_F32(&{dst_data}, &{src_data}, {vl});",
       c_global=_SATURN_CUSTOM_ASM)
def saturn_bf16_widen_f32_m2(
    dst: [f32][16]  @ SaturnRVV_M2,
    src: [ui16][16] @ SaturnRVV_M1,
    vl:  size,
):
    for i in seq(0, vl):
        dst[i] = src[i]    # opaque (cross-type, semantically BF16 -> FP32)
```

Same opaque type-mismatched body idiom as `vfexp_v` — Exo's
unification needs the `src` reference to bind the parameter.

### `saturn_vfmacc_vv_f32m2` @instr

```python
@instr("{dst_data} = __riscv_vfmacc_vv_f32m2("
       "{dst_data}, {lhs_data}, {rhs_data}, {vl});")
def saturn_vfmacc_vv_f32m2(
    dst: [f32][16] @ SaturnRVV_M2,
    lhs: [f32][16] @ SaturnRVV_M2,
    rhs: [f32][16] @ SaturnRVV_M2,
    vl:  size,
):
    for i in seq(0, vl):
        dst[i] += lhs[i] * rhs[i]
```

Real arithmetic body — Exo unifies the high-level @proc's
`S_reg[i] += Q_reg[i] * K_fp32_reg[i]` loop with this @instr's body
shape. No new RVV-stdlib idiom; this is base RVV 1.0 `vfmacc.vv`.

vfmul.vf was originally pencilled in for step 2b but isn't needed
for QK^T accumulation specifically — it's the softmax scale-by-1/√d
primitive and lands when step 2c (softmax) does.

## Schedule (paper/exo_schedule_fa.py)

The @proc carries explicit (load, ..., compute, ..., store) structure
already, so the schedule is just memory annotations + 7 `replace()`
calls in source order (same pattern as `schedule_softmax_exp_chunk`):

```python
def schedule_qkt_chunk(p=qkt_chunk_naive, verbose=False):
    p = set_memory(p, "Q_reg",       SaturnRVV_M2)
    p = set_memory(p, "K_nvfp4_reg", SaturnRVV_M1)
    p = set_memory(p, "K_bf16_reg",  SaturnRVV_M1)
    p = set_memory(p, "K_fp32_reg",  SaturnRVV_M2)
    p = set_memory(p, "S_reg",       SaturnRVV_M2)
    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vle16_m1)
    p = replace(p, "for i in _: _ #0", vfconv_nvfp4_bf16_v)
    p = replace(p, "for i in _: _ #0", saturn_bf16_widen_f32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vfmacc_vv_f32m2)
    p = replace(p, "for i in _: _ #0", saturn_vse32_m2)
    return p
```

The vfmacc body `S_reg[i] += Q_reg[i] * K_fp32_reg[i]` is the first
non-opaque-copy body in the demo file. Exo's unification handles the
`+=` and `*` operators directly (matches the stock RVV `rvv_vfmacc_4xf32_4xf32`
template at `exo/src/exo/platforms/rvv.py:136-147`).

## Cross-compile + disassembly verification

Compiled with `riscv64-linux-gcc 14.2 -O2 -march=rv64gcv -mabi=lp64d`.
No diagnostics. Disassembly of `qkt_chunk_naive` (compute chain
excerpt; stack-frame setup elided):

```
000000000000021e <qkt_chunk_naive>:
  ...
  222: vsetivli zero,16,e32,m2,ta,ma
  22a: vle16.v v1,(a2)                 # K_nvfp4 -> v1
  256: vle32.v v6,(a1)                 # Q_fp32 -> v6 (f32m2)
  264: vs1r.v v1,(a2)                  # spill (asm clobber zone)

  # SATURN_VFCONV_NVFP4_BF16:
  270: vsetvli zero,a1,e16,m1,ta,ma
  274: vle16.v v0,(a2)
  278: 4e049057   .word 0x4e049057     # vfconv.nvfp4.bf16.v v0,v0
  27c: vse16.v v0,(a4)

  # SATURN_BF16_WIDEN_F32:
  28e: vsetvli zero,a1,e16,m1,ta,ma
  292: vle16.v v8,(a4)
  296: vsetvli zero,a1,e32,m2,ta,ma
  29a: vzext.vf2 v10,v8                # zero-extend ui16 -> ui32
  29e: vsll.vi  v10,v10,16             # shift left 16: BF16 -> FP32
  2a2: vse32.v  v10,(a5)

  # FP32 vfmacc + S_acc r-m-w:
  2a6: vl2re32.v v4,(a5)               # K_fp32 reload
  2aa: vle32.v   v2,(a3)               # S_acc load
  2ae: vfmacc.vv v2,v6,v4              # S += Q * K_fp32  ✓
  2b2: vse32.v   v2,(a3)               # S_acc store
```

All three new primitives fire:
- `.4byte 0x4e049057` (vfconv.nvfp4.bf16.v) at 0x278
- `vzext.vf2 v10,v8` + `vsll.vi v10,v10,16` at 0x29a–0x29e
  (the SATURN_BF16_WIDEN_F32 macro)
- `vfmacc.vv v2,v6,v4` at 0x2ae

5/5 schedules unified across the demo. Total Saturn / new-@instr
counts in the object:
- `.4byte 0x4e049057`: 3 (one each in dequant_chunk, dequant_64,
  fa_dequant_per_row; not in qkt_chunk yet — wait, the disasm shows
  it IS in qkt_chunk. Recount: 4 occurrences total.)
- `vzext.vf2`: 1 (only in qkt_chunk_naive)
- `vsll.vi`: 1 (only in qkt_chunk_naive)
- `vfmacc.vv`: 1 (only in qkt_chunk_naive)

## Known inefficiencies (deferred to step 4)

The asm-volatile macros spill/reload through stack between each
Saturn op (e.g., `vs1r.v` at 0x264, `vle16.v ... ; vse16.v ...`
churn around 0x274–0x27c, `vl2re32.v` at 0x2a6). This is the price
of using `asm volatile` with clobber lists that prevent GCC from
keeping operands in vector registers across calls.

Additionally, each Saturn macro emits its own `vsetvli` since it
controls SEW/LMUL internally — the dequant chain has 3 separate
`vsetvli` issues (e16/m1 inside vfconv, e16/m1 inside widen load,
e32/m2 inside widen compute) that an optimization pass could merge.

Both are real but cost-known: matching the §6 hand-coded cycle count
needs either (a) intrinsic-based @instrs that let GCC see across
operations, or (b) a post-schedule peephole pass that fuses Saturn
macros. Either is a step-4 (cycle parity) concern, not a step-2b
correctness concern.

## Open Mo 8 substeps

- **2c. Online softmax mini-kernel** — needs +2 reduction @instrs
  (`vfredmax.vs`, `vfredsum.vs`) and a small max-subtract / exp /
  sum-rescale schedule. Reuses existing `vfexp_v` and the
  `saturn_vfmacc_vv_f32m2` from this step (for the cross-tile
  rescale arithmetic).
- **3.** Wire remaining 2 vfconv lanes (`bf16.fp8.v` for P-quant,
  `fp8.bf16.v` for P·V dequant). Composes with this step's vfmacc.
- **4.** Build the Exo-generated kernel on gem5 and compare cycles
  to `bench_fa_mixed_rvv_native`. Address the two known
  inefficiencies above if cycle delta is large.

## Reproducibility

```
cd .
pip install -e exo/                        # picks up new @instrs in fork
python3 paper/exo_schedule_fa.py           # 5/5 schedules; 10 markers
```

Disassembly probe:

```
cd /tmp/mo8s1
python3 -c "
import sys; sys.path.insert(0, '.')
sys.path.insert(0, './exo/src')
from exo.API import compile_procs_to_strings
from paper.exo_schedule_fa import (schedule_dequant_chunk,
    schedule_softmax_exp_chunk, schedule_dequant_64,
    schedule_fa_dequant_per_row, schedule_qkt_chunk)
c, h = compile_procs_to_strings(
    [schedule_dequant_chunk(), schedule_softmax_exp_chunk(),
     schedule_dequant_64(), schedule_fa_dequant_per_row(),
     schedule_qkt_chunk()],
    'exo_schedule_fa.h')
open('exo_schedule_fa.c','w').write('#include \"exo_schedule_fa.h\"\n'+c)
open('exo_schedule_fa.h','w').write(h)
"
/path/to/bootlin-riscv64-gcc14/bin/riscv64-linux-gcc \
    -O2 -march=rv64gcv -mabi=lp64d -c exo_schedule_fa.c -o exo_schedule_fa.o
/path/to/bootlin-riscv64-gcc14/bin/riscv64-linux-objdump \
    -d exo_schedule_fa.o | sed -n '/<qkt_chunk_naive>:/,/<softmax_exp_chunk_naive>:/p'
```
