# Mo 8 step 4b-2 — softmax_full + dequant_row (final FA building blocks)

## Headline

**Mo 8 step 4b-2 PASS.** Two new @procs and one new @instr land the
remaining FA building blocks needed for the step 4b-3 fused-kernel
composition:

- `softmax_full_naive`: SEQ_LEN-parameterized two-pass softmax
  (vfredmax pass + vfsub.vf+narrow+vfexp+vfredusum pass), matching
  `bench_fa_mixed_rvv_native.c` lines 207-249.
- `dequant_row_naive`: head_dim=64 NVFP4 → FP32 dequant with E4M3
  per-block scale, matching the bench's `dequant_64elt_chunk` at
  one-block granularity (lines 58-108).
- `saturn_vfmul_vf_f32m2` @instr: standard RVV 1.0 vfmul.vf, scalar
  arg as f32[1] @ DRAM (uniform with the platform's scalar surface).

Together with step 4a's `pv_macc_chunk` and step 4b-1's `qkt_dot` +
`pv_macc_row`, all 5 FA-shape building blocks now exist as Exo-
scheduled @procs. Step 4b-3 composes them into `fa_kernel_decode_naive`.

## What was added

### `saturn_vfmul_vf_f32m2` @instr (exo fork saturn_rvv.py)

```python
@instr("{dst_data} = __riscv_vfmul_vf_f32m2("
       "{src_data}, {scalar_data}, {vl});")
def saturn_vfmul_vf_f32m2(
    dst:    [f32][16] @ SaturnRVV_M2,
    src:    [f32][16] @ SaturnRVV_M2,
    scalar: [f32][1]  @ DRAM,
    vl:     size,
):
    for i in seq(0, vl):
        dst[i] = src[i] * scalar[0]
```

Used for two purposes in FA:
- E4M3 per-block scale apply inside `dequant_row` (matches
  `bench_fa_mixed_rvv_native.c:83,88,93,98` — `vfmul.vf v8, v8, %[sk]`).
- Output row-scale at the end of FA (`O *= row_dequant_scale` per
  the bench's row normalization, line 281-282 — to be wired in
  step 4b-3).

### `softmax_full_naive` @proc (main repo)

SEQ_LEN-parameterized two-pass softmax over the full S vector:

```python
@proc
def softmax_full_naive(
    seq_len: size,
    S_fp32:  f32[seq_len] @ DRAM,
    P_fp32:  f32[seq_len] @ DRAM,
    m_out:   f32[1]       @ DRAM,
    l_out:   f32[1]       @ DRAM,
):
    assert seq_len > 0
    assert seq_len % 16 == 0

    # Pass 1: max-reduce
    for so in seq(0, seq_len / 16):
        S_reg1: f32[16]
        for i in seq(0, 16):
            S_reg1[i] = S_fp32[16 * so + i]
        for i in seq(0, 16):
            m_out[0] = fmaxf(m_out[0], S_reg1[i])

    # Pass 2: sub + narrow + exp + store + sum
    for so in seq(0, seq_len / 16):
        S_reg2:    f32[16]
        S_shifted: f32[16]
        S_bf16:    ui16[16]
        P_reg:     f32[16]
        for i in seq(0, 16):                         # vle32 S
            S_reg2[i] = S_fp32[16 * so + i]
        for i in seq(0, 16):                         # vfsub.vf
            S_shifted[i] = S_reg2[i] - m_out[0]
        for i in seq(0, 16):                         # FP32→BF16 narrow
            S_bf16[i] = S_shifted[i]
        for i in seq(0, 16):                         # vfexp
            P_reg[i] = S_bf16[i]
        for i in seq(0, 16):                         # vse32 P
            P_fp32[16 * so + i] = P_reg[i]
        for i in seq(0, 16):                         # vfredusum
            l_out[0] += P_reg[i]
```

Caller initializes m_out[0]=-inf and l_out[0]=0 before calling.

**Loop bound `seq_len / 16` is a valid Exo expression** — confirmed
this session. The frontend accepts integer-division of a `size`
parameter by a constant directly in the loop bound. Eliminates the
need for divide_loop + simplify in the schedule.

Lowered C body (Pass 1 + Pass 2 sections):

```c
void softmax_full_naive(void *ctxt, int_fast32_t seq_len,
                        const float* S_fp32, float* P_fp32,
                        float* m_out, float* l_out) {
  EXO_ASSUME(seq_len > 0);
  EXO_ASSUME(seq_len % 16 == 0);
  for (int_fast32_t so = 0; so < ((seq_len) / (16)); so++) {
    vfloat32m2_t S_reg1;
    S_reg1 = __riscv_vle32_v_f32m2(&S_fp32[16 * so], 16);
    SATURN_VFREDMAX_F32M2(m_out[0], S_reg1, 16);
  }
  for (int_fast32_t so = 0; so < ((seq_len) / (16)); so++) {
    vfloat32m2_t S_reg2;
    vfloat32m2_t S_shifted;
    vuint16m1_t  S_bf16;
    vfloat32m2_t P_reg;
    S_reg2 = __riscv_vle32_v_f32m2(&S_fp32[16 * so], 16);
    S_shifted = __riscv_vfsub_vf_f32m2(S_reg2, m_out[0], 16);
    SATURN_F32_NARROW_BF16(&S_bf16, &S_shifted, 16);
    SATURN_VFEXP(&P_reg, &S_bf16, 16);
    __riscv_vse32_v_f32m2(&P_fp32[16 * so], P_reg, 16);
    SATURN_VFREDUSUM_F32M2(l_out[0], P_reg, 16);
  }
}
```

Compare to bench lines 207-249: structurally identical except the
bench inlines vfexp inside an asm-volatile block (and runs it at
SEW=32/m2 with FP32 input); ours uses the existing vfexp_v @instr
which consumes BF16 (so we explicitly insert SATURN_F32_NARROW_BF16
between vfsub.vf and SATURN_VFEXP). **This is a known precision-
chain divergence flagged for step 4d** — reconciling vfexp_v's @instr
surface to FP32-input would eliminate the narrow step and align
with the bench's hot loop exactly.

### `dequant_row_naive` @proc (main repo)

Head_dim=64 NVFP4 → FP32 dequant with E4M3 per-block scale. 4 NVFP4
blocks of 16 elements each, scaled separately:

```python
@proc
def dequant_row_naive(
    K_nvfp4: ui16[4, 16] @ DRAM,    # 4 blocks × 16 ui16 carriers
    K_scale: f32[4]      @ DRAM,    # E4M3 scales decoded to FP32
    K_fp32:  f32[64]     @ DRAM,    # output FP32 head_dim row
):
    for blk in seq(0, 4):
        K_nvfp4_reg: ui16[16]
        K_bf16_reg:  ui16[16]
        K_fp32_reg:  f32[16]
        K_scaled:    f32[16]
        for i in seq(0, 16):                          # vle16 nvfp4 block
            K_nvfp4_reg[i] = K_nvfp4[blk, i]
        for i in seq(0, 16):                          # vfconv.nvfp4.bf16.v
            K_bf16_reg[i] = K_nvfp4_reg[i]
        for i in seq(0, 16):                          # bf16_widen_f32
            K_fp32_reg[i] = K_bf16_reg[i]
        for i in seq(0, 16):                          # vfmul.vf (scale)
            K_scaled[i] = K_fp32_reg[i] * K_scale[blk]
        for i in seq(0, 16):                          # vse32 fp32 block
            K_fp32[16 * blk + i] = K_scaled[i]
```

**Exo's replace() unifies `K_scale[blk]` against the vfmul.vf
@instr's `scalar[0]` automatically.** No stage_mem or window
manipulation needed — Exo's unifier binds the indexed scalar
access to the 1-element windowed view at the @instr surface. This
is the same pattern used by stock RVV's `rvv_vfmacc_4xf32_1xf32`
in `exo/src/exo/platforms/rvv.py`.

Lowered C body:

```c
void dequant_row_naive(void *ctxt, const uint16_t* K_nvfp4,
                       const float* K_scale, float* K_fp32) {
  for (int_fast32_t blk = 0; blk < 4; blk++) {
    vuint16m1_t  K_nvfp4_reg;
    vuint16m1_t  K_bf16_reg;
    vfloat32m2_t K_fp32_reg;
    vfloat32m2_t K_scaled;
    K_nvfp4_reg = __riscv_vle16_v_u16m1(&K_nvfp4[blk * 16], 16);
    SATURN_VFCONV_NVFP4_BF16(&K_bf16_reg, &K_nvfp4_reg, 16);
    SATURN_BF16_WIDEN_F32(&K_fp32_reg, &K_bf16_reg, 16);
    K_scaled = __riscv_vfmul_vf_f32m2(K_fp32_reg, K_scale[blk], 16);
    __riscv_vse32_v_f32m2(&K_fp32[16 * blk], K_scaled, 16);
  }
}
```

The bench's `dequant_64elt_chunk` issues vfconv at LMUL=4 (64 lanes
in one shot), then loops the widen+scale 4 times. Our @proc instead
loops everything 4 times at LMUL=1/m2 — denser-but-tighter inner
work vs. one big vfconv + four widen+scale chunks. Cycle delta
between these layouts is a step-4c measurement question; the @instr
surface choice (no LMUL=4 vfconv variant yet) reflects the
foundation work's m1-only NVFP4 convention.

## Cross-compile + disassembly verification

Compiled with `riscv64-linux-gcc 14.2 -O2 -march=rv64gcv -mabi=lp64d`.
13/13 schedules unify in the demo file; 23 markers fire.

### softmax_full_naive (Pass 1 + Pass 2 excerpts)

Pass 1 (`.L70` — vfredmax loop):

```
<.L70>:
  7f0: vle32.v     v2,(a0)             # S chunk load
  7f4: vfmv.s.f    v1,fa5              # init reduce acc from m_out
  7f8: addi        a0,a0,64
  7fc: vfredmax.vs v2,v2,v1            # ✓ max-reduce
  800: vfmv.f.s    fa5,v2              # extract
  804: fsw         fa5,0(a4)           # m_out store
  808: bne         a0,a6,7f0 <.L70>
```

Pass 2 (`.L72` — sub + narrow + vfexp + sum):

```
<.L72>:
  840: flw         fa5,0(a4)           # load m_out for sub
  844: vle32.v     v2,(a2)             # S chunk load
  848: vfsub.vf    v2,v2,fa5           # ✓ S - max
  84c: vs2r.v      v2,(t1)             # spill (asm clobber)
  850: vsetvli     ...,e32,m2,...
  854: vle32.v     v4,(t1)
  858: vsetvli     ...,e16,m1,...
  85c: vnsrl.wi    v8,v4,16            # ✓ FP32→BF16 narrow
  860: vse16.v     v8,(a7)
  864: vsetvli     ...,e16,m1,...
  868: vle16.v     v8,(a7)
  86c: 4e831457    .word 0x4e831457    # ✓ vfexp.v
  870: vsetvli     ...,e32,m2,...
  ...
```

### dequant_row_naive (blk loop body `.L16`)

```
<.L16>:
  1a8: vle16.v   v1,(a1)               # K_nvfp4 load
  ...
  1b8: 4e049057  .word 0x4e049057      # ✓ vfconv.nvfp4.bf16.v
  ...
  1cc: vzext.vf2 v10,v8                # widen step 1
  1d0: vsll.vi   v10,v10,16            # widen step 2
  ...
  1dc: flw       fa5,0(a2)             # load K_scale[blk]
  1e0: addi      a1,a1,32              # K_nvfp4 + 32 bytes (16 ui16)
  1e4: addi      a2,a2,4               # K_scale + 4 bytes (1 f32)
  1e6: vfmul.vf  v2,v2,fa5             # ✓ scale apply
  1ea: vse32.v   v2,(a3)               # K_fp32 store
  1ee: addi      a3,a3,64              # K_fp32 + 64 bytes (16 f32)
  1f2: bne       a1,a7,1a8 <.L16>
```

All primitives fire in the expected sequence. The blk loop advances
3 pointers per iteration (K_nvfp4 += 32, K_scale += 4, K_fp32 += 64)
exactly matching the @proc's `K_nvfp4[blk, i] / K_scale[blk] /
K_fp32[16 * blk + i]` addressing.

## Mo 8 step 4 substep status

| substep | scope                                         | status |
|---------|-----------------------------------------------|--------|
| 4a      | vfmacc.vf @instr + pv_macc_chunk              | done   |
| 4b-1    | vfmv_zero @instr + qkt_dot + pv_macc_row      | done   |
| 4b-2    | vfmul.vf @instr + softmax_full + dequant_row  | **this** |
| 4b-3    | compose fa_kernel_decode_naive                | next   |
| 4c      | C harness + gem5 first cycle measurement      | TBD    |
| 4d      | optimize known-cost macros if delta > 10%     | TBD    |

After 4b-3, every piece of `bench_fa_mixed_rvv_native.c`'s decode
kernel has an Exo-scheduled counterpart. Step 4c builds the C
harness and runs first measurement.

## Known divergence flagged for step 4d

The vfexp_v @instr's macro currently sets SEW=16/m1 (consumes BF16
input) but the bench's inline-asm runs vfexp.v at SEW=32/m2 (FP32
input). Both presumably work on Saturn's RTL (which internally
operates on FP32 per Track E's design), but the Exo-side surface
adds a forced FP32→BF16 narrow step before vfexp that the bench
skips. Reconciling the surface (either a vfexp_f32_v variant or
fixing the existing macro to consume FP32 at SEW=32/m2) is on the
step 4d optimization list. Cycle cost of this divergence: 1
SATURN_F32_NARROW_BF16 macro per chunk × `seq_len/16` chunks in
softmax — measurable, addressable.

## Reproducibility

```
cd /home/noah/project/riscv
pip install -e exo/
python3 paper/exo_schedule_fa.py     # 13/13 schedules; 23 markers
```

Disassembly probe: extend prior recipe with `schedule_softmax_full()`
and `schedule_dequant_row()` in the procs list.
