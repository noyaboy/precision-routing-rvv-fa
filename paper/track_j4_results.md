# Track J-4 — gem5 custom-encoding patch for Saturn FUs (POC, 2026-05-17)

## Goal

Patch gem5's RISC-V decoder to issue Saturn's custom encodings
(`vfconv.{nvfp4,bf16,fp8}.*` + `vfexp.v`) as native vector instructions.
This unblocks the verified Mo 6 cycle measurement that Track J-2-O3-long
ruled out via long-context alone: with the FU instructions in-decoder, the
mixed-prec FA kernel can model Saturn's 16-lane parallel dequant directly
instead of paying the scalar dequant cost on simulated hardware.

## Scope of this POC session

All **four** Saturn custom instructions land in gem5's decoder + execute
bit-exactly on test vectors.  Full path: encoding pick → decoder patch →
gem5 rebuild → inline asm wrapper → functional validation.

## Encoding scheme

All custom Saturn ops sit in the OPFVV space (RVV major opcode 0x57,
funct3=0x1) under the single-source FP family VFUNCT6=0x13.  This puts
them next to the existing vfsqrt/vfrsqrt7/vfrec7/vfclass single-source
ops and inherits the same VS1-sub-decode + register-file plumbing.

| Instruction              | VFUNCT6 | VS1   | Encoding (vd=v0, vs2=v0) | Validation     |
|--------------------------|--------:|------:|-------------------------:|----------------|
| vfexp.v                  |    0x13 | 0x06  | 0x4E031057               | 8/8 bit-exact  |
| vfconv.fp8.bf16.v        |    0x13 | 0x07  | 0x4E039057               | 8/8 bit-exact  |
| vfconv.bf16.fp8.v        |    0x13 | 0x08  | 0x4E041057               | 8/8 bit-exact  |
| vfconv.nvfp4.bf16.v      |    0x13 | 0x09  | 0x4E049057               | 16/16 bit-exact|

Pre-existing VS1 slots in VFUNCT6=0x13 are 0x00 (vfsqrt_v), 0x04
(vfrsqrt7_v), 0x05 (vfrec7_v), 0x10 (vfclass_v).  Our 0x06-0x09 don't
collide.

To assemble in inline C: `asm volatile (".4byte 0x4E031057");` for vfexp.v
v0,v0.  General formula:
```
encoding = 0x4E001057 | (vfunct6 == 0x13 ? 0 : ...)
         | ((vs1 & 0x1f) << 15)
         | ((vd  & 0x1f) << 7)
         | ((vs2 & 0x1f) << 20)
```

## Decoder patch

File: `src/arch/riscv/isa/decoder.isa`, in the `0x13: decode VS1 { format
VectorFloatCvtFormat { ... } }` block.  Each entry follows the existing
vfsqrt template — single-source vector op, per-element body in `{{ ... }}`,
op class for FU latency.  Excerpt:

```
0x06: vfexp_v({{
    if (sizeof(et) == 4) {
        uint32_t in_u = (uint32_t)Vs2_vu[i];
        float in_f;
        std::memcpy(&in_f, &in_u, sizeof(in_f));
        float out_f = expf(in_f);
        uint32_t out_u;
        std::memcpy(&out_u, &out_f, sizeof(out_u));
        Vd_vu[i] = out_u;
    } else {
        Vd_vu[i] = 0;
    }
}}, OPFVV, SimdFloatSqrtOp);
```

The per-element semantics use libm `expf`; the gem5 op class
(`SimdFloatSqrtOp` placeholder, target `SimdFloatExpOp` with 10-cycle
latency once we add a dedicated class) governs the cycle behavior on O3.

`vfconv.fp8.bf16.v` follows the same shape: SEW=16 (low byte is the FP8
input, output is BF16), per-element body decodes E4M3-FN to FP32 then
truncates to BF16.  Bit-accurate against Track F3's
`VFConvFp8Bf16Lane.scala` for the standard mapping; sub-normal handling
matches the Chisel module.

## POC validation

Two test microbenches, both run on gem5 25.1.0.1 / RISC-V / TimingSimpleCPU:

### `test_vfexp.c`
- Input: 8 FP32 values [0, 1, 2, ln(2), -1, 0.5, 3, -2.5]
- Issues vfexp.v on the vector
- Compares to scalar libm expf

Result: **8/8 elements match expf bit-exactly** (rel_err = 0).

### `test_vfconv_fp8_bf16.c`
- Input: 8 FP8-E4M3 bytes [0x38, 0x40, 0x48, 0x30, 0xB0, 0xB8, 0x00, 0x7E]
  (= FP8 values 1.0, 2.0, 4.0, 0.5, -0.5, -1.0, 0.0, 448.0)
- Issues vfconv.fp8.bf16.v on the vector
- Compares to expected BF16 representation

Result: **8/8 elements match bit-exactly**.  Sample mappings:
  - 0x38 → 0x3F80 (1.0)
  - 0x40 → 0x4000 (2.0)
  - 0x48 → 0x4080 (4.0)
  - 0xB8 → 0xBF80 (-1.0)
  - 0x7E → 0x43E0 (448.0, FP8-E4M3 max)
  - 0x00 → 0x0000 (+0)

## Build cost

| Build step                                | Wall time |
|-------------------------------------------|----------:|
| No-change incremental rebuild (32 cores)  | ~5 min    |
| One-instruction .isa change rebuild       | ~3-5 min  |

The .isa parser regeneration is the slow step; subsequent .cc compilation
parallelizes well.

## Remaining work (next sessions)

J-4.4 done + J-4.5a (op-class latency wiring) done.

### J-4.5a — FU op-class latency wiring (DONE 2026-05-17 late PM)

Added 4 new op classes to gem5 and wired them through:

- `src/cpu/FuncUnit.py` — added `SimdFloatExp`, `SimdNvfp4Cvt`,
  `SimdBf16Fp8Cvt`, `SimdFp8Bf16Cvt` to the `OpClass` enum.
- `src/cpu/op_class.hh` — added matching C++ aliases (`SimdFloatExpOp`, etc.).
- `src/cpu/o3/FuncUnitConfig.py` — added 4 `OpDesc` entries in `SIMD_Unit`
  with the Saturn latencies (10c / 3c / 2c / 1c).
- `src/arch/riscv/isa/decoder.isa` — re-tagged the 4 customs from
  placeholder `SimdFloatSqrtOp`/`SimdFloatCvtOp` to the new op classes.

Verification: `test_vfexp_latency.c` runs a tight load-vfexp-store loop
1000 times.  Result on RiscvO3CPU: **13.05 cyc/iter**, consistent with
a 10-cycle vfexp.v latency plus load/store overlap.  All 4 functional
tests continue to pass after the wiring change.

### J-4.5b — FA-kernel integration (DONE, 2026-05-17 EVENING)

`bench_fa_mixed_rvv_native.c` replaces the K/V dequant + softmax exp with
native Saturn customs:
- `dequant_row_native()`: scalar 8-byte → 16-nibble unpack × 4 blocks,
  then **one** asm block doing `vle16 v0 (m4, vl=64)` → `vfconv.nvfp4.bf16.v
  v0,v0` → `vsetivli e32 m2 vl=16` → 4× `(vzext.vf2 / vsll.vi / vfmul.vf
  scale_blk / vse32)`.  Two vsetvli transitions per row (down from eight
  in the prior cut).
- Softmax exp loop: `vfexp.v v8,v8` in an asm loop, LMUL=2, reducing
  into a scalar `vfredusum.vs` accumulator.
- FP8 quant follows J3 mixed proper (`bf16_to_e4m3(fp32_to_bf16(P*448))`)
  with per-row `inv_sum/448` fold-in.  PV dequant uses `e4m3_decode[]`.

#### Two bugs found + fixed (vs prior session's hypothesis)

**Bug A (numerical correctness)**: Real root cause was *not* the
`"f"`-constraint scale-width hypothesis from the kickoff doc — `objdump`
shows `flw fa5, ...` + `vfmul.vf v8, v8, fa5` (correct FP32 binding).
Actual cause: **GCC 13.2's RVV vsetvli optimization pass mis-handles
the dequant-asm → `bf16_load_widen` intrinsic interaction in the QK
inner loop**.  The asm leaves vtype = e32 m2 vl=16; GCC's tracker
matches; the per-iter intrinsic chain emits a single `vsetvli zero, t3,
e16, m1` (for the `vle16` of Q), but then *fails to emit the e32 m2
transition* before the subsequent `vzext.vf2 / vsll.vi / vfmacc.vv`.
Those run at SEW=16: `vzext.vf2` reads u8 (source EEW = SEW/2) instead
of u16, `vsll.vi 16` is shift-by-(16 mod 16)=0, and `vfmacc.vv` lands
on half-precision FMA without Zvfh.  Result: garbage `S[s]`, all of
softmax + PV downstream is junk.  **Fix**: pre-widen Q to FP32 at init
(2 KiB scratch), use `__riscv_vle32_v_f32m2` for Q in the QK loop —
removes the e16↔e32 switch from the inner loop entirely, sidesteps the
GCC pass bug.  Q-load count drops by ~2 ops/iter (vle16+vzext+vsll → vle32).

**Bug B (cycle regression)**: 8 vsetvli per row × 16K rows = ~130K cyc
overhead in the prior single-block design.  Restructured to one e16-m4
vfconv over all 64 nibbles + one e32-m2 widen/scale/store pass with the
4 block scales pre-loaded.  2 vsetvli per row.

**Bug C (FP8 quant stub, inherited from J3+stub)**: The prior kernel
carried `P_fp8[s] = (uint8_t)((int)pf & 0xff)` from `bench_fa_mixed_rvv_stub.c`.
With `pf ∈ (0,1]` from softmax, `(int)pf` is 0 for almost all s →
P_fp8 ≈ 0 → O ≈ 0 → checksum near zero (this is why J3+stub's checksum
9.94 is labelled "NOT a correctness check" in its own file).  Replaced
with the J3-mixed-RVV proper FP8 quant.

#### Results on RiscvO3CPU (seq_len=2048 head_dim=64)

| Variant                            | Cycles | IPC  | Insts   | Checksum     | Status |
|------------------------------------|-------:|-----:|--------:|-------------:|--------|
| BF16 RVV (J3, Mo 6 baseline)       | 3.39 M | 0.82 |  2.78 M | −19.6924     | ref |
| J3 mixed RVV **proper** (SW dequant)| 8.89 M | 2.46 | 21.86 M | −20.2981     | apples-to-apples baseline |
| J3+stub mixed (cycle-only, broken FP8)| 7.65 M | 1.87 | 14.29 M |  9.9375 ✗   | cycle-only |
| **J4 native (this session)**       | **7.00 M** | **2.19** | **15.36 M** | **−20.3000** ✓ | DONE |

Correctness: native checksum −20.3000 vs J3-mixed-proper −20.2981 →
**0.009 % relative diff**, well inside the 5 % NVFP4 quantization-noise
window from the kickoff success criterion.

Cycles: native is **1.27× faster than J3 mixed RVV proper** (8.89 → 7.00),
**1.09× faster than J3+stub** (7.65 → 7.00).  Instruction count drops
30 % vs J3 proper (15.36 M vs 21.86 M) — the FU collapses ~6 M scalar
NVFP4-dequant ops into ~32 K `vfconv` issues + ~1 K `vfexp` issues.

**Mo 6 verdict at seq_len=2048 head_dim=64: MISSES.**  Native / BF16-RVV-J3
= 7.00 / 3.39 = **2.07× slower** (target ≥1.5× faster, i.e. ≤2.26 M).
This confirms the J-2-O3-long finding: at this kernel scale the BF16 RVV
path is compute-bound at 0.22 B/cyc DRAM, well within DDR3-1600's 6.4 B/cyc
cap; NVFP4's BW advantage (1.18 MiB read vs 4.20 MiB) does not translate
to cycles.  FU integration buys 21 % cycle savings vs the all-SW mixed
path, but doesn't close the 2× gap to BF16 RVV at this scale.

#### Bandwidth claim (verified, unchanged)

NVFP4 K/V reads 1.18 MiB; BF16 K/V reads 4.20 MiB.  **3.56× DRAM
reduction**, identical to Tracks J / J-2 / Track H findings.

#### Paper-framing implication (unchanged from J-2-O3-long)

Mo 6 misses the literal ≥1.5×-over-BF16-RVV bar on every gem5 config
tested.  Two paths forward, neither blocking the MLArchSys 2027 submission:
1. **Re-frame Mo 6** (drafted in `track_j2_results.md` § "Best Mo 6
   framing"): (a) verified BW claim (3.5–14× DRAM reduction across
   simulators + seq_lens); (b) projected cycle claim with FU
   integration.  J-4.5b now *verifies* the FU-integration cycle delta —
   J4 native at 7.00 M is the projection's experimental ground truth on
   this gem5 + DDR3 config.  Long-context regime or larger head_dim
   could still tip the balance (Track J-2-O3-long ran out at L16K /
   hd=64 with the 2.22-2.32× gap flat).
2. **FireSim FPGA path**: real Saturn RTL with proper FU pipeline +
   memory subsystem.  Out of scope for this paper revision.

### J-4.5c — Long-context O3 sweep with native FU integration (DONE 2026-05-17 EVENING)

Six O3 sims (3 native_lN + 3 mixed_rvv_lN proper) at seq_len ∈ {4 K,
8 K, 16 K}, hd=64, on the same gem5 config as J-4.5b.  Existing
J-2-O3-long runs cover BF16 RVV (Mo 6 ref) and mixed-stub (cycle-only
baseline).

| seq_len | BF16 RVV (ref) | Mixed RVV proper | Mixed RVV stub | **Native (FU)** | Native/BF16 | Native/Proper |
|--------:|---------------:|-----------------:|---------------:|---------------:|-----------:|--------------:|
|    2 K  |  3.39 M (0.82) |   8.89 M (2.46)  |  7.65 M (1.87) | **7.00 M (2.19)** | **2.07×** | 0.79× (21 % faster) |
|    4 K  |  6.74 M (0.82) |  17.79 M (2.46)  | 14.96 M (1.91) | **15.80 M (1.94)** | **2.34×** | 0.89× (11 % faster) |
|    8 K  | 13.43 M (0.83) |  35.95 M (2.43)  | 30.68 M (1.86) | **31.73 M (1.94)** | **2.36×** | 0.88× (12 % faster) |
|   16 K  | 26.88 M (0.83) |  71.49 M (2.44)  | 62.23 M (1.84) | **62.33 M (1.97)** | **2.32×** | 0.87× (13 % faster) |

IPC in parens.  Native checksums all in [−20.45, −20.26], matching J3
proper to <1 % across the sweep.  Native instruction count grows
strictly linearly (30.7 M → 61.5 M → 122.8 M, exactly 2× per doubling),
matching the expected per-row work.

#### Mo 6 verdict (locked)

The native/BF16 ratio is **flat at 2.07–2.36× across the entire
L2K–L16K sweep**.  Long-context does *not* bridge the gap — confirms
the J-2-O3-long projection that BF16 RVV stays compute-bound at this
gem5 config (DRAM use ~1.3-1.6 B/cyc, well within DDR3-1600's
6.4 B/cyc cap; IPC pinned at 0.82-0.83 across the sweep).  The
NVFP4 BW advantage is real (3.5× DRAM reduction, verified) but cannot
translate to cycles when the compute side isn't bottlenecked.

The best case for native is the smallest seq_len (L2K → 2.07×); ratio
grows slightly with seq_len, plateaus at L8K-L16K.  No tested seq_len
in [2 K, 16 K] gets within 38 % of the ≥1.5× target.

#### FU integration cycle benefit (verified)

Native beats mixed-RVV proper (iso-FP8-quant, SW dequant) by 11–21 %
cycles at every seq_len, with FU savings scaling roughly linearly:
1.89 M (L2K) → 1.99 M (L4K) → 4.22 M (L8K) → 9.17 M (L16K).  Native
instruction count drops 30 % vs proper at L2K (15.4 M vs 21.9 M), 30 %
at L16K too (122.8 M vs 174.8 M) — the FU collapses ~50 M scalar
dequant ops at L16K into ~256 K vfconv issues.

Native IPC (1.94-2.19) is consistently lower than proper (2.43-2.46)
because FU ops queue against fewer dependencies per op.  But native
*cycle* count wins because it has 30 % fewer instructions total.

#### Paper framing implication (cement)

The L-sweep gives 4 data points for the Mo 6 framing:

- **Verified BW claim**: 3.56× DRAM reduction across L2K-L16K (identical
  to J / J-2 / Track H findings).
- **Projected cycle claim, now experimentally bounded**: with FU
  integration, mixed-prec is 11-21 % faster than SW-dequant mixed-prec
  on iso-FP8-quant comparison.  Full ≥1.5× over BF16 RVV requires
  either (a) a memory subsystem where BF16 RVV is BW-bound (HBM-class
  bandwidth ceiling much lower than gem5+DDR3's), (b) larger head_dim
  shifting the compute/BW balance, or (c) real Saturn µarch with
  16-lane FU pipeline that the gem5 model approximates with
  3c/10c-latency single-instruction stubs.

#### Remaining for Track J-4

1. **head_dim=128 sweep** (~3 h): rewrite `dequant_row_native` to loop
   over `blocks_per_row = HEAD_DIM / NVFP4_BLOCK` (currently hardcoded
   to 4 blocks) and `bench_fa_common.h` to make `HEAD_DIM` configurable
   via `-DHEAD_DIM`.  Re-run native + BF16 RVV at hd128 across L2K-L16K.
   Hypothesis: per-row compute doubles, per-row dequant doubles —
   probably ratio stays flat (BF16 stays compute-bound).  Tests the
   last open dimension before committing fully to the projected-cycle
   framing.
2. **Native FP8 quant** (~2 h): replace SW `bf16_to_e4m3` loop with
   `vfconv.bf16.fp8.v` (VS1=0x08, already decoded).  Expected savings:
   a few × 100 K cyc per kernel run.  Tightens the "all Saturn customs
   integrated" story but doesn't change Mo 6 verdict.
3. **Exhaustive bit-exact verification** (~1 day): port Tracks E/F/F2/F3
   goldens (321 / 4096 / 65536 / 256 + 256 cases) to gem5 microbenches.
   The 16/16-sample test_vfconv_nvfp4_bf16 and 8/8 test_vfexp cover the
   integration path; the full sweep would close the loop on decoder
   semantics matching Chisel module bit patterns.

## Note on the two-source vfconv.nvfp4.bf16.v

Saturn's real FU takes both an NVFP4 packed nibble vector AND an E4M3
scale vector, producing the scaled BF16 internally.  The single-source
decoder entry here returns the **unscaled** BF16 value; the application
multiplies by the E4M3-decoded scale separately (one vfmul per block).
This matches the Track F2 stub variant pattern used in `bench_fa_mixed_stub.c`
and is a compose-rather-than-fuse choice that keeps the decoder simple.
A two-source variant could later be added at VS1=0x0a using a different
format machinery (read both vs1 and vs2 as vector registers, with the
sub-opcode moving to a higher bit field).

## Files

- `gem5/src/arch/riscv/isa/decoder.isa` — patched with vfexp_v at
  VFUNCT6=0x13/VS1=0x06 and vfconv_fp8_bf16_v at VS1=0x07.
- `microbench-j4/test_vfexp.c`, `test_vfconv_fp8_bf16.c` — POC microbenches.
- `microbench-j4/run_test_vfexp/` — gem5 output from the first POC run
  (PASS, 8/8 bit-exact against libm expf).

## Reproducibility

```bash
cd ./gem5
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gem5-build
export CC=gcc CXX=g++ CPATH=$CONDA_PREFIX/include \
       LIBRARY_PATH=$CONDA_PREFIX/lib \
       LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
scons build/RISCV/gem5.opt -j32 --quiet

cd ./microbench-j4
PATH=/path/to/bootlin-riscv64/bin:$PATH \
  riscv64-linux-gcc -O2 -march=rv64gcv -mabi=lp64d -static \
  test_vfexp.c -o test_vfexp -lm

mkdir -p run_test_vfexp && cd run_test_vfexp
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  $GEM5_DIR/build/RISCV/gem5.opt --outdir=. \
  $GEM5_DIR/configs/deprecated/example/se.py \
  --cpu-type=TimingSimpleCPU --num-cpus=4 --mem-size=512MB \
  -c ../test_vfexp
```
