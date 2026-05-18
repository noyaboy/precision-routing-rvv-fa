# Mo 8 step 3 — P-quant + P·V dequant lanes wired

## Headline

**Mo 8 step 3 PASS.** Two new @procs land in
`paper/exo_schedule_fa.py` that wire the remaining 2 Saturn vfconv
lanes (`vfconv.bf16.fp8.v` and `vfconv.fp8.bf16.v`) into the
scheduling methodology. After this step **all 4 Saturn customs are
reachable from the schedule's @instr surface** end-to-end. The exo
fork required **no new @instrs** — both lanes were already declared
in `saturn_rvv.py` (5.2, 5.3) since the foundation; step 3 just
composes them with the step 2b/2c primitives into @procs that
demonstrate the §6 P-quant and P·V passes.

## What was added

### `pq_chunk_naive`: post-softmax P-quant

After softmax produces 16 FP32 attention weights, this chunk
quantizes them down to FP8 for storage / downstream P·V matmul:

```
P_fp32 (FP32)
  -> SATURN_F32_NARROW_BF16     (FP32 -> BF16, step 2c primitive)
  -> SATURN_VFCONV_BF16_FP8     (BF16 -> FP8, vfconv.bf16.fp8.v)
  -> vse8 to DRAM (16 valid bytes, mf2 LMUL)
```

Output DRAM buffer is `ui8[32]` because `vfconv.bf16.fp8.v` writes
16 valid FP8 bytes into the lower mf2 half of a SaturnRVV_M1
`ui8[32]` register slot — the upper half is unmeaningful but the
slot is allocated at LMUL=1 width (32 lanes at SEW=8). The
`vse8_m1` store writes 16 bytes since vl=16.

Lowered C body:

```c
void pq_chunk_naive(void *ctxt, const float* P_fp32, uint8_t* P_fp8) {
  vfloat32m2_t P_reg;
  vuint16m1_t  P_bf16_reg;
  vuint8m1_t   P_fp8_reg;
  P_reg = __riscv_vle32_v_f32m2(&P_fp32[0], 16);
  SATURN_F32_NARROW_BF16(&P_bf16_reg, &P_reg, 16);
  SATURN_VFCONV_BF16_FP8(&P_fp8_reg, &P_bf16_reg, 16);
  __riscv_vse8_v_u8m1(&P_fp8[0], P_fp8_reg, 16);
}
```

### `pv_chunk_naive`: P·V inner-product accumulator

The "inverse partner" of step 2b's `qkt_chunk_naive` — composes
both dequant chains (FP8→BF16→FP32 for P, NVFP4→BF16→FP32 for V)
and uses the step 2b vfmacc.vv to accumulate into the FP32 O
output buffer:

```
P_fp8   -> SATURN_VFCONV_FP8_BF16   -> SATURN_BF16_WIDEN_F32 -> P_fp32
V_nvfp4 -> SATURN_VFCONV_NVFP4_BF16 -> SATURN_BF16_WIDEN_F32 -> V_fp32
O_reg   += P_fp32 * V_fp32          (vfmacc.vv f32m2)
```

Lowered C body:

```c
void pv_chunk_naive(void *ctxt, const uint8_t* P_fp8,
                    const uint16_t* V_nvfp4, float* O_acc) {
  vuint8m1_t   P_fp8_reg;
  vuint16m1_t  P_bf16_reg;
  vfloat32m2_t P_fp32_reg;
  vuint16m1_t  V_nvfp4_reg;
  vuint16m1_t  V_bf16_reg;
  vfloat32m2_t V_fp32_reg;
  vfloat32m2_t O_reg;
  P_fp8_reg   = __riscv_vle8_v_u8m1(&P_fp8[0], 16);
  SATURN_VFCONV_FP8_BF16(&P_bf16_reg, &P_fp8_reg, 16);
  SATURN_BF16_WIDEN_F32(&P_fp32_reg, &P_bf16_reg, 16);
  V_nvfp4_reg = __riscv_vle16_v_u16m1(&V_nvfp4[0], 16);
  SATURN_VFCONV_NVFP4_BF16(&V_bf16_reg, &V_nvfp4_reg, 16);
  SATURN_BF16_WIDEN_F32(&V_fp32_reg, &V_bf16_reg, 16);
  O_reg = __riscv_vle32_v_f32m2(&O_acc[0], 16);
  O_reg = __riscv_vfmacc_vv_f32m2(O_reg, P_fp32_reg, V_fp32_reg, 16);
  __riscv_vse32_v_f32m2(&O_acc[0], O_reg, 16);
}
```

Same per-lane-partial-sum semantics as step 2b — one tile produces
16 partial-sum updates to O. Cross-tile composition (resetting O
between Q rows, multiplying by softmax cross-tile rescale α) is the
§6 outer loop's responsibility.

## Schedule

Both schedules are pure set_memory + replace() chains. No stage_mem,
no divide_loop. The @procs carry explicit (load, ..., compute, ...,
store) structure already.

```python
def schedule_pq_chunk(p=pq_chunk_naive, ...):
    p = set_memory(p, "P_reg",      SaturnRVV_M2)
    p = set_memory(p, "P_bf16_reg", SaturnRVV_M1)
    p = set_memory(p, "P_fp8_reg",  SaturnRVV_M1)
    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)
    p = replace(p, "for i in _: _ #0", saturn_f32_narrow_bf16_m2)
    p = replace(p, "for i in _: _ #0", vfconv_bf16_fp8_v)
    p = replace(p, "for i in _: _ #0", saturn_vse8_m1)
    return p

def schedule_pv_chunk(p=pv_chunk_naive, ...):
    p = set_memory(p, "P_fp8_reg",   SaturnRVV_M1)
    p = set_memory(p, "P_bf16_reg",  SaturnRVV_M1)
    p = set_memory(p, "P_fp32_reg",  SaturnRVV_M2)
    p = set_memory(p, "V_nvfp4_reg", SaturnRVV_M1)
    p = set_memory(p, "V_bf16_reg",  SaturnRVV_M1)
    p = set_memory(p, "V_fp32_reg",  SaturnRVV_M2)
    p = set_memory(p, "O_reg",       SaturnRVV_M2)
    p = replace(p, "for i in _: _ #0", saturn_vle8_m1)
    p = replace(p, "for i in _: _ #0", vfconv_fp8_bf16_v)
    p = replace(p, "for i in _: _ #0", saturn_bf16_widen_f32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vle16_m1)
    p = replace(p, "for i in _: _ #0", vfconv_nvfp4_bf16_v)
    p = replace(p, "for i in _: _ #0", saturn_bf16_widen_f32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vfmacc_vv_f32m2)
    p = replace(p, "for i in _: _ #0", saturn_vse32_m2)
    return p
```

Both unify cleanly (8/8 schedules in the demo). 14 → 18 expected
markers; 4 new ones (`SATURN_VFCONV_BF16_FP8`, `__riscv_vse8_v_u8m1`,
`SATURN_VFCONV_FP8_BF16`, `__riscv_vle8_v_u8m1`) all fire.

## Cross-compile + disassembly verification

Compiled with `riscv64-linux-gcc 14.2 -O2 -march=rv64gcv -mabi=lp64d`.
Both new Saturn encodings fire in the expected functions:

### pq_chunk_naive (excerpt)

```
000000000000031c <pq_chunk_naive>:
  ...
  328: vsetivli zero,16,e32,m2,ta,ma
  32c: vle32.v  v2,(a1)                   # P_fp32 load
  ...
  376: vsetvli  zero,a0,e16,m1,ta,ma
  37a: vnsrl.wi v8,v4,16                  # FP32 -> BF16 narrow (step 2c primitive)
  ...
  388: vle16.v  v8,(a3)
  38c: 4e841457  .word 0x4e841457         # vfconv.bf16.fp8.v ✓
  390: vsetvli  zero,a0,e8,mf2,ta,ma
  394: vnsrl.wi v0,v8,0                   # macro-internal narrow ui16->ui8
  398: vse8.v   v0,(a5)                   # macro-internal store
  ...
  3a0: vsetivli zero,16,e8,m1,ta,ma
  3a4: vse8.v   v1,(a2)                   # saturn_vse8_m1 outer store (16 bytes)
```

### pv_chunk_naive (excerpt)

```
00000000000003cc <pv_chunk_naive>:
  ...
  3d0: vsetivli zero,16,e8,m1,ta,ma
  3d4: vle8.v   v1,(a1)                   # P_fp8 load
  ...
  41e: vle8.v   v8,(a1)                   # (macro reload at mf2)
  422: vsetvli  zero,a6,e16,m1,ta,ma
  426: 4e839057  .word 0x4e839057         # vfconv.fp8.bf16.v ✓
  ...
  442: vsetvli  zero,a6,e16,m1,ta,ma
  446: vle16.v  v8,(a4)
  44a: vsetvli  zero,a6,e32,m2,ta,ma
  44e: vzext.vf2 v10,v8                   # P widen step 1 (step 2b primitive)
  452: vsll.vi  v10,v10,16                # P widen step 2
  ...
  45e: vle16.v  v1,(a2)                   # V_nvfp4 load
  ...
  47a: vle16.v  v0,(a2)
  47e: 4e049057  .word 0x4e049057         # vfconv.nvfp4.bf16.v ✓ (V dequant)
  ...
  49a: vzext.vf2 v10,v8                   # V widen step 1
  49e: vsll.vi  v10,v10,16                # V widen step 2
  ...
  4aa: vle32.v  v2,(a3)                   # O_acc load
  4ae: vl2re32.v v6,(a0)                  # P_fp32 reload
  4ba: vfmacc.vv v2,v6,v4                 # O += P * V (step 2b primitive) ✓
  4be: vse32.v  v2,(a3)                   # O_acc store
```

The full §6 P·V chain renders correctly: load P (FP8) → dequant →
widen, load V (NVFP4) → dequant → widen, vfmacc into O FP32
accumulator, store. Two distinct Saturn `.4byte` encodings
(`0x4e839057` for fp8→bf16, `0x4e049057` for nvfp4→bf16) plus the
shared widen path firing twice for both operands.

## All 4 Saturn customs now reachable

After step 3, every Saturn custom instruction declared in the
paper's §4-5 design has a corresponding demo @proc + schedule:

| Saturn custom              | Encoding     | Demo @proc                     | Step  |
|----------------------------|--------------|--------------------------------|-------|
| vfconv.nvfp4.bf16.v        | 0x4e049057   | dequant_chunk + dequant_64 +   | found |
|                            |              | fa_dequant_per_row + qkt_chunk |       |
|                            |              | + pv_chunk                     |       |
| vfconv.bf16.fp8.v          | 0x4e841457   | pq_chunk                       | **3** |
| vfconv.fp8.bf16.v          | 0x4e839057   | pv_chunk                       | **3** |
| vfexp.v                    | 0x4e831457   | softmax_exp_chunk +            | found |
|                            |              | online_softmax_chunk           |       |

Combined with the BF16<->FP32 widen/narrow bridge (added in steps
2b + 2c) and the FP32 vfmacc.vv + vfsub.vf + vfredmax/vfredusum
primitives, the full precision-routing flow described in
paper §4-5 is reachable through Exo's scheduling API.

## Open Mo 8 substeps

Only step 4 remains: build the Exo-generated kernel on gem5 and
compare cycles to `bench_fa_mixed_rvv_native`. The shape of step 4:

1. Pick a representative tile (e.g., one head × one Q row × one
   K/V tile = 16 keys × head_dim=64).
2. Compose all 5 step-2/3 @procs into a single fused kernel that
   tiles over the §6 loop structure (step 2a's outer-loop shape).
3. Wrap in a benchmark harness (similar to microbench-fa/) that
   feeds it real Q/K/V/O buffers and times under gem5 5.1 +
   RiscvO3CPU + DDR3-1600.
4. Compare cycles to `bench_fa_mixed_rvv_native` at matching
   seq_len. Target: within 10% (Mo 8 PASS).
5. If delta is large, attack the known-cost inefficiencies:
   - Scalar DRAM-bouncing (replace with register-resident scalar
     @instr variants).
   - vsetvli churn (peephole merge across macros with shared
     SEW/LMUL).
   - asm-volatile stack spills (intrinsic-based @instr variants
     where the Saturn customs allow).

## Reproducibility

```
cd .
python3 paper/exo_schedule_fa.py     # 8/8 schedules; 18 markers
```

Disassembly probe (the script section in `mo8_step2c_results.md`'s
Reproducibility block; extend the `schedule_*` list to include
`schedule_pq_chunk(), schedule_pv_chunk()`).
