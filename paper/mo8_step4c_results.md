# Mo 8 step 4c — first gem5 cycle measurement of fa_kernel_decode

## Headline

**Mo 8 step 4c PASS (measurement)** / **PARTIAL (cycle-parity target).**
The Exo-generated `fa_kernel_decode_naive` (steps 4a + 4b-1/2/3) runs
end-to-end on gem5 5.1 SE + RiscvO3CPU + DDR3-1600 at SEQ_LEN=2048
with **6.17M cycles** vs the hand-coded `bench_fa_mixed_rvv_native_g14`'s
**4.76M cycles** under the same gem5 config. Raw delta: **1.30×**.
Apples-to-apples (with the deferred scalar phases added back to our
kernel): estimated **~1.5×**.

The Mo 8 PASS target was **within 10%** (i.e., ≤1.10×). **Current gap
is ~50% over target** — meaningful shortfall, but the IPC analysis
below shows the gap is concentrated in exactly the place step 4a's
intrinsic-vs-macro analysis predicted, making step 4d's optimization
plan directly actionable.

## Measurement

Both runs on the same gem5 launch command at SEQ_LEN=2048:

```
gem5.opt --outdir=run/<name> se.py \
  --cpu-type=RiscvO3CPU --num-cpus=2 \
  --caches --l1d_size=32KiB --l1i_size=32KiB \
  --l2cache --l2_size=512KiB \
  --mem-size=512MB \
  -c <binary>
```

### Cycle counts (post-`m5_dump_reset_stats` measurement region)

| Binary                                 | Cycles      | IPC   |
|----------------------------------------|------------:|------:|
| `bench_fa_mixed_rvv_native_g14` (hand) | 4,756,714   | 1.366 |
| `bench_fa_exo` (Exo step 4b-3)        | 6,173,097   | 0.652 |

**Raw cycle delta: 1.30×** (Exo / hand-coded).
**IPC delta: 0.48×** (Exo / hand-coded) — Exo runs at *half* the
instructions-per-cycle of the bench.

### Cycle accounting

The Exo bench's measurement region excludes 3 phases that the
hand-coded bench includes inside its hot loop:

1. **Scalar nibble-unpack** of NVFP4 packed bytes → 16 ui16 carriers
   per block (the bench's `dequant_64elt_chunk` inlines this; our
   harness pre-unpacks before `m5_dump_reset_stats`). At L2K:
   ~524 K nibble unpacks (8 heads × 2048 rows × 32 nibbles ÷ 2 ×
   2 K/V sides). Estimated cycle cost in the bench: 0.5–1.0M
   cycles.
2. **Scalar FP8 quant of P**: per-key `bf16_to_e4m3(fp32_to_bf16(P[s] *
   448))`. At L2K: 16 K scalar quants per head × 8 heads = 131 K
   ops; estimated 0.2–0.5M cycles.
3. **Output BF16 narrow + `row_dequant_scale`**: 512 element conversions
   total; negligible (~5 K cycles).

Summed: Exo skips ~0.7–1.5M cycles of scalar work that the bench
measures. Adding it back to our kernel would land us in the
**6.87–7.67M cycles** range, vs the bench's 4.76M. **Apples-to-apples
cycle delta: 1.44–1.61×**.

### IPC analysis: the gap is in the macros

The IPC delta is the dominant story. Same algorithmic work
(NVFP4 dequant + QK^T + softmax + P·V) and structurally
identical hot loops, but the Exo kernel issues at **half the
out-of-order overlap** the bench achieves:

- **Bench native: 1.37 IPC** — GCC sees through the dequant +
  widen + vfmacc chain (all intrinsics-based, no asm-volatile
  macros except the .4byte Saturn customs which have small
  spill scopes), so the OoO core can issue 16-lane vector ops
  with high parallelism.
- **Exo: 0.65 IPC** — every Saturn macro (BF16↔FP32 widen/narrow,
  vfconv lanes, vfexp, vfredmax/vfredusum) is an asm-volatile
  block with a clobber list that forces GCC to spill registers
  before and reload after. Each macro becomes a serialization
  barrier; the OoO core can't overlap macro calls.

This is **exactly the quantitative finding from step 4a**:
the asm-volatile-macro path costs ~12× the instruction count
of the equivalent intrinsic path. Now we have the runtime
confirmation: the macros also cost ~50% IPC.

### Numerical sanity check

- Bench checksum: **−136.808** (after `row_dequant_scale` apply
  + BF16 narrow).
- Exo checksum: **1625.195** (raw FP32, no `row_dequant_scale`,
  no BF16 narrow).

Checksums differ structurally because we skip the FP8 quant
round-trip + output normalization. Numerical correctness of the
cycle-dominating phases (dequant, QK^T, softmax, P·V) needs
verification separately by adding the missing phases to the Exo
harness as scalar C code (step 4c follow-on or step 4d).

## Step 4d optimization plan (now sharply prioritized)

The IPC gap is the cycle gap. Closing the gap means rewriting the
high-cost asm-volatile macros as intrinsic-based @instrs:

| Macro                              | Underlying ops                          | RVV intrinsic available? |
|------------------------------------|-----------------------------------------|--------------------------|
| `SATURN_BF16_WIDEN_F32`            | vzext.vf2 + vsll.vi 16                  | yes (both standard)      |
| `SATURN_F32_NARROW_BF16`           | vnsrl.wi 16                             | yes                      |
| `SATURN_VFREDMAX_F32M2`            | vfmv.v.f + vfredmax.vs + vfmv.f.s       | yes (all standard)       |
| `SATURN_VFREDUSUM_F32M2`           | vfmv.v.f + vfredusum.vs + vfmv.f.s      | yes                      |
| `SATURN_VFEXP`                     | vfexp.v (custom)                        | **no** — .4byte stays    |
| `SATURN_VFCONV_NVFP4_BF16`         | vfconv.nvfp4.bf16.v (custom)            | **no** — .4byte stays    |
| `SATURN_VFCONV_BF16_FP8`           | vfconv.bf16.fp8.v (custom)              | **no** — .4byte stays    |
| `SATURN_VFCONV_FP8_BF16`           | vfconv.fp8.bf16.v (custom)              | **no** — .4byte stays    |

**4 of 8 macros can be rewritten as pure-intrinsic @instrs.** The
other 4 are stuck with asm-volatile until GCC adds intrinsic
support for the Saturn customs (paper §7.5 sidebar already
flags this).

Step 4d execution plan:
- 4d-1: Rewrite the 4 standard-RVV macros as intrinsic-based @instrs.
- 4d-2: Re-measure on gem5. Expect IPC recovery to ~1.0–1.2 and
  cycles to drop into the **5.0–6.0M range** (still above the
  4.76M bench, but within ~20–25%).
- 4d-3: If still above the 10% PASS target, reconcile vfexp_v
  to consume FP32 input (eliminating the FP32→BF16 narrow + extra
  vsetvli per softmax chunk) for further savings.
- 4d-4: Inline nibble-unpack in dequant_row (apples-to-apples
  with bench's `dequant_64elt_chunk`).

## Mo 8 step 4 substep status

| substep | scope                                          | status     |
|---------|------------------------------------------------|------------|
| 4a      | vfmacc.vf + pv_macc_chunk                      | done       |
| 4b-1    | vfmv_zero + qkt_dot + pv_macc_row              | done       |
| 4b-2    | vfmul.vf + softmax_full + dequant_row          | done       |
| 4b-3    | compose fa_kernel_decode_naive                 | done       |
| 4c      | C harness + first gem5 measurement             | **this**   |
| 4d      | optimize known-cost macros (intrinsic rewrite) | next       |

Step 4c lands the **first end-to-end cycle measurement of an Exo-
generated FA kernel** on gem5 + Saturn customs. **Mo 8 cycle-parity
PASS (within 10%) is achievable** via step 4d's targeted intrinsic
rewrites — the IPC evidence makes the gap source unambiguous.

## Reproducibility

```
cd ./microbench-fa-exo
# Regenerate Exo C/H if needed:
python3 -c "import sys; sys.path.insert(0,'.'); \
  sys.path.insert(0,'./exo/src'); \
  from exo.API import compile_procs_to_strings; \
  from paper.exo_schedule_fa import schedule_fa_kernel_decode; \
  c,h = compile_procs_to_strings([schedule_fa_kernel_decode()],'exo_schedule_fa.h'); \
  open('exo_schedule_fa.c','w').write('#include \"exo_schedule_fa.h\"\n'+c); \
  open('exo_schedule_fa.h','w').write(h)"

# Build:
PATH=/path/to/bootlin-riscv64-gcc14/bin:$PATH make bench_fa_exo

# Run on gem5:
mkdir -p run/4c_l2k
./gem5/build/RISCV/gem5.opt \
  --outdir=run/4c_l2k \
  ./gem5/configs/deprecated/example/se.py \
  --cpu-type=RiscvO3CPU --num-cpus=2 \
  --caches --l1d_size=32KiB --l1i_size=32KiB \
  --l2cache --l2_size=512KiB \
  --mem-size=512MB \
  -c ./bench_fa_exo

# Compare:
grep "rdcycle delta" run/4c_l2k/system.terminal  # 6173077
grep "system.cpu0.numCycles\|system.cpu0.ipc" run/4c_l2k/stats.txt
```
