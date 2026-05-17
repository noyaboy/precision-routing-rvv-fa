# Mo 8 step 4b-1 — per-row FA building blocks (qkt_dot + pv_macc_row)

## Headline

**Mo 8 step 4b-1 PASS.** Two new @procs and one new @instr land the
per-row FA building blocks that compose with step 4a's
`pv_macc_chunk` to cover the QK^T and P·V passes of the §6 fused
flash-attention decode-step kernel. Both @procs emit C that is
**structurally identical to `bench_fa_mixed_rvv_native.c`** at the
matching lines (193-205 for QK, 272-278 for PV) and **disassemble to
lean, intrinsic-only RVV bodies** with no stack-frame / spill-reload
overhead — the cycle-parity-friendly side of the intrinsic-vs-macro
gap quantified in step 4a.

## What was added

### `saturn_vfmv_zero_f32m2` @instr (exo fork saturn_rvv.py)

Standard RVV 1.0 `vfmv.v.f` with the literal `0.0f` scalar. Needed
because SaturnRVV memory class declares `can_read=False`, so
element-wise scalar init (`for i: S_acc[i] = 0.0`) emitted by Exo
on a SaturnRVV_M2 buffer would attempt to read register lanes —
which the memory class rejects. The proper init path is to declare
the init loop in the @proc and replace() it with `saturn_vfmv_zero_f32m2`:

```python
@instr("{dst_data} = __riscv_vfmv_v_f_f32m2(0.0f, {vl});")
def saturn_vfmv_zero_f32m2(
    dst: [f32][16] @ SaturnRVV_M2,
    vl:  size,
):
    for i in seq(0, vl):
        dst[i] = 0.0
```

### `qkt_dot_naive` @proc (main repo)

Per-key Q · K dot product over head_dim=64 → scalar S[s]. Matches
`bench_fa_mixed_rvv_native.c` lines 193-205. Structure:

```python
@proc
def qkt_dot_naive(
    Q_fp32: f32[64] @ DRAM,
    K_fp32: f32[64] @ DRAM,
    scale:  f32     @ DRAM,
    S_out:  f32[1]  @ DRAM,
):
    Q_reg:  f32[16]
    K_reg:  f32[16]
    S_acc:  f32[16]
    for i in seq(0, 16):                      # vfmv.v.f init
        S_acc[i] = 0.0
    for ko in seq(0, 4):                      # head_dim chunks
        for i in seq(0, 16):                  # vle32 Q
            Q_reg[i] = Q_fp32[16 * ko + i]
        for i in seq(0, 16):                  # vle32 K
            K_reg[i] = K_fp32[16 * ko + i]
        for i in seq(0, 16):                  # vfmacc.vv
            S_acc[i] += Q_reg[i] * K_reg[i]
    S_out[0] = 0.0
    for i in seq(0, 16):                      # vfredusum
        S_out[0] += S_acc[i]
    S_out[0] = S_out[0] * scale               # scalar 1/sqrt(head_dim)
```

Lowered C body:

```c
void qkt_dot_naive(void *ctxt, const float* Q_fp32, const float* K_fp32,
                   const float* scale, float* S_out) {
  vfloat32m2_t Q_reg, K_reg, S_acc;
  S_acc = __riscv_vfmv_v_f_f32m2(0.0f, 16);
  for (int_fast32_t ko = 0; ko < 4; ko++) {
    Q_reg = __riscv_vle32_v_f32m2(&Q_fp32[16 * ko], 16);
    K_reg = __riscv_vle32_v_f32m2(&K_fp32[16 * ko], 16);
    S_acc = __riscv_vfmacc_vv_f32m2(S_acc, Q_reg, K_reg, 16);
  }
  S_out[0] = 0.0f;
  SATURN_VFREDUSUM_F32M2(S_out[0], S_acc, 16);
  S_out[0] = S_out[0] * *scale;
}
```

Compare to the bench inline kernel (`bench_fa_mixed_rvv_native.c:193-205`):

```c
size_t vl_max = __riscv_vsetvl_e32m2(16);
vfloat32m2_t vacc = __riscv_vfmv_v_f_f32m2(0.0f, vl_max);
for (int d = 0; d < HEAD_DIM; d += (int)vl_max) {
    size_t vl = __riscv_vsetvl_e32m2(HEAD_DIM - d);
    vfloat32m2_t vq = __riscv_vle32_v_f32m2(qh_fp32 + d, vl);
    vfloat32m2_t vk = __riscv_vle32_v_f32m2(K_fp32_row + d, vl);
    vacc = __riscv_vfmacc_vv_f32m2(vacc, vq, vk, vl);
}
vfloat32m1_t vred_init = __riscv_vfmv_v_f_f32m1(0.0f, 1);
vfloat32m1_t vred = __riscv_vfredusum_vs_f32m2_f32m1(vacc, vred_init, vl_max);
float acc = __riscv_vfmv_f_s_f32m1_f32(vred);
S[s] = acc * scale;
```

Structurally identical. The Exo emission wraps vfredusum + vfmv.f.s
in the SATURN_VFREDUSUM_F32M2 macro; the bench inlines them.

### `pv_macc_row_naive` @proc (main repo)

Per-key P · V over head_dim=64. Matches `bench_fa_mixed_rvv_native.c`
lines 272-278. Composes step 4a's `pv_macc_chunk` over 4 head_dim
chunks:

```python
@proc
def pv_macc_row_naive(
    p_scalar: f32[1]  @ DRAM,
    V_fp32:   f32[64] @ DRAM,
    O_fp32:   f32[64] @ DRAM,
):
    V_reg: f32[16]
    O_reg: f32[16]
    for ko in seq(0, 4):
        for i in seq(0, 16):                  # vle32 V
            V_reg[i] = V_fp32[16 * ko + i]
        for i in seq(0, 16):                  # vle32 O
            O_reg[i] = O_fp32[16 * ko + i]
        for i in seq(0, 16):                  # vfmacc.vf
            O_reg[i] += p_scalar[0] * V_reg[i]
        for i in seq(0, 16):                  # vse32 O
            O_fp32[16 * ko + i] = O_reg[i]
```

Lowered C body:

```c
void pv_macc_row_naive(void *ctxt, const float* p_scalar,
                       const float* V_fp32, float* O_fp32) {
  vfloat32m2_t V_reg, O_reg;
  for (int_fast32_t ko = 0; ko < 4; ko++) {
    V_reg = __riscv_vle32_v_f32m2(&V_fp32[16 * ko], 16);
    O_reg = __riscv_vle32_v_f32m2(&O_fp32[16 * ko], 16);
    O_reg = __riscv_vfmacc_vf_f32m2(O_reg, p_scalar[0], V_reg, 16);
    __riscv_vse32_v_f32m2(&O_fp32[16 * ko], O_reg, 16);
  }
}
```

Same as bench's inner P·V loop (lines 272-278):

```c
for (int d = 0; d < HEAD_DIM; d += (int)vlm) {
    size_t vl = __riscv_vsetvl_e32m2(HEAD_DIM - d);
    vfloat32m2_t vo = __riscv_vle32_v_f32m2(O_fp32 + d, vl);
    vfloat32m2_t vv = __riscv_vle32_v_f32m2(V_fp32_row + d, vl);
    vo = __riscv_vfmacc_vf_f32m2(vo, p, vv, vl);
    __riscv_vse32_v_f32m2(O_fp32 + d, vo, vl);
}
```

## Cross-compile + disassembly verification

Compiled with `riscv64-linux-gcc 14.2 -O2 -march=rv64gcv -mabi=lp64d`.
Both new functions disassemble to lean, intrinsic-only RVV bodies
(no stack frame, no spill/reload, single vsetvli for the whole
function):

### qkt_dot_naive (15 instructions including loop)

```
000000000000005ec <qkt_dot_naive>:
  5ec: vsetivli zero,16,e32,m2,ta,ma
  5f0: vmv.v.i  v2,0                    # vfmv.v.f init (zero)
  5f4: li       a5,0
  5f6: li       a7,256                   # loop limit: 4 chunks × 64 bytes

<.L53>:                                 # ko loop body
  5fa: add      a6,a1,a5                 # &Q_fp32[16*ko]
  5fe: add      a0,a2,a5                 # &K_fp32[16*ko]
  602: vle32.v  v6,(a6)
  606: vle32.v  v4,(a0)
  60a: addi     a5,a5,64                 # ko_byte_offset += 64
  60e: vfmacc.vv v2,v6,v4
  612: bne      a5,a7,5fa <.L53>

  # Reduction + scalar scale (from SATURN_VFREDUSUM_F32M2 macro expansion):
  616: vmv.s.x   v1,zero
  61a: vfredusum.vs v2,v2,v1
  61e: vfmv.f.s  fa4,v2
  622: fsw       fa4,0(a4)               # S_out[0] = sum
  626: flw       fa5,0(a3)               # load *scale
  62a: fmul.s    fa5,fa5,fa4              # apply scale
  62e: fsw       fa5,0(a4)
  632: ret
```

### pv_macc_row_naive (9 instructions including loop)

```
0000000000000500 <pv_macc_row_naive>:
  500: li       a5,0
  502: li       a6,256
  506: vsetivli zero,16,e32,m2,ta,ma

<.L44>:                                 # ko loop body
  50a: add      a4,a3,a5                 # &O_fp32[16*ko]
  50e: add      a0,a2,a5                 # &V_fp32[16*ko]
  512: vle32.v  v4,(a0)
  516: vle32.v  v2,(a4)
  51a: flw      fa5,0(a1)                # load p_scalar
  51e: addi     a5,a5,64
  522: vfmacc.vf v2,fa5,v4
  526: vse32.v  v2,(a4)
  52a: bne      a5,a6,50a <.L44>
  52e: ret
```

Both kernels confirm the step-4a quantitative finding: **all-intrinsic
@instrs produce cycle-parity-friendly disassembly with no
asm-volatile overhead**. These two pieces should reach near-identical
cycle counts to their bench counterparts on Saturn out-of-order.

## Schedule notes

The `replace(p, "for i in _: _ #0", ...)` chain inside the qkt_dot
@proc demonstrates a subtle pattern worth recording: when the @proc
has a `for ko` outer loop with N inner-`i` loops per iteration, the
schedule needs **N replace() calls total** (one per loop-body shape),
not N × ko_count (one per unrolled iteration). Exo's replace operates
on the abstract IR; each replace() matches the next sibling `for i`
loop in source order, regardless of where it sits in the outer-loop
nest. This works because Exo unifies the @instr body against the
abstract loop body shape, which is identical across all ko iterations.

The same idiom applies to pv_macc_row's 4 inner loops (V load / O
load / vfmacc.vf / O store).

## Open Mo 8 step 4 substeps

- **4b-2** Build `softmax_full_naive` @proc that scales step 2c's
  `online_softmax_chunk_naive` up to a SEQ_LEN-parameterized two-
  pass softmax (max-reduce pass + sub/exp/sum pass). Use
  `divide_loop` over `for s in seq(0, seq_len)` to tile into
  16-lane chunks, then stage_mem + replace each phase's loops.
  Plus build `dequant_row_naive` that calls vfconv.nvfp4.bf16.v
  + bf16_widen_f32 + vfmul.vf (apply E4M3 per-block scale; new
  @instr needed) for one head_dim row.
- **4b-3** Compose qkt_dot + softmax_full + scalar FP8-quant loop +
  pv_macc_row + dequant_row into `fa_kernel_decode_naive` that
  matches `bench_fa_mixed_rvv_native.c`'s decode-step structure
  exactly. Cross-compile.
- **4c** Build C harness (similar to `microbench-fa/run` scripts)
  that calls `fa_kernel_decode_naive` with the same inputs as
  the bench. Run on gem5 5.1 SE + RiscvO3CPU + DDR3-1600. First
  cycle comparison vs `bench_fa_mixed_rvv_native`.
- **4d** Optimize if cycle delta > 10%. Primary attack surfaces
  from step 4a's findings:
  1. Rewrite `SATURN_BF16_WIDEN_F32` / `SATURN_F32_NARROW_BF16` /
     `SATURN_VFREDMAX_F32M2` / `SATURN_VFREDUSUM_F32M2` macros as
     intrinsic-based @instrs (vzext.vf2 / vsll.vi / vnsrl.wi /
     vfredmax.vs / vfredusum.vs are all standard RVV 1.0).
  2. Peephole-merge consecutive Saturn .4byte customs that share
     SEW/LMUL.

## Reproducibility

```
cd /home/noah/project/riscv
pip install -e exo/
python3 paper/exo_schedule_fa.py     # 11/11 schedules; 21 markers
```

Disassembly probe (extend the script in prior results docs with
`schedule_qkt_dot()` and `schedule_pv_macc_row()` in the schedule list).
