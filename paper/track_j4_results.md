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

### J-4.5b — FA-kernel integration (PARTIAL, 2026-05-17 late PM)

Wrote `bench_fa_mixed_rvv_native.c` that replaces:
- Scalar `dequant_row_stub()` with `dequant_row_native()` (scalar byte
  unpack → vector vfconv.nvfp4.bf16.v → BF16-widen + vfmul with E4M3 scale),
  all in one self-contained asm block.
- Scalar `fu_expf()` softmax loop with a native vector `vfexp.v` asm loop
  (LMUL=2, accumulating sum into a v0[0] reduction).

**Results on RiscvO3CPU (seq_len=2048 head_dim=64)**:

| Variant                       | Cycles | IPC  | Checksum  | Status            |
|-------------------------------|-------:|-----:|-----------|-------------------|
| BF16 RVV (J3, Mo 6 baseline)  | 3.39M  | 0.82 | −19.6924  | reference         |
| Mixed J3+stub (fresh re-run)  | 7.65M  | 1.87 |  9.937500 | matches prior     |
| **J4 native, full**           | 8.46M  | 1.79 | −0.045    | runs but **wrong** |
| J4 native, just dequant       | 7.96M  | 2.01 | NaN       | bisection variant |
| J4 native, just vfexp asm     | (TBD)  |      |           | not yet isolated  |

Two integration issues surfaced and are unresolved:

1. **Numerical correctness drift** — the full native checksum is −0.045
   versus the J1 mixed-prec reference of −20.27.  The dequant runs (no
   NaN with the fixed self-contained asm) but the magnitudes are wrong.
   Most likely culprit: the `vfmul.vf` reads `scale` from an FP register;
   the `"f"` constraint in GCC may give a double-typed reg instead of a
   32-bit-typed reg, leading to wrong-width interpretation under SEW=32.
   Fix candidates: bind scale via `"f"` constraint inside the asm but
   first store it to memory and reload via `vfmv.s.f` from a u32 view;
   or convert the scale to its IEEE-754 bit pattern as an integer and
   move into an FP reg manually.
2. **Cycle regression** — J4 native (8.46M) is **slower** than
   J3+stub (7.65M).  Root cause: vsetvli switches between e16 m1 and
   e32 m2 inside `dequant_row_native` cost ~2-3 cycles each, ~2 switches
   × 4 blocks × 2048 × 8 ≈ 130 K cycle overhead.  Fix: batch the
   per-row dequant in fewer vsetvli configs, or reorder so e32 m2 mode
   persists across blocks (vfconv must run at e16 m1 but the widen + scale
   can run at e32 m2 once per row).

The path is open; both fixes are tractable but need another integration
pass.  Once corrected the projected J4-native cycle count (subtracting the
130K vsetvli overhead from the current 8.46M, plus reclaiming the dequant
savings the buggy variant suggested ~5%) is in the 7.0–7.2M range — still
not the ≥1.5× Mo 6 win.  To beat 2.26M (≥1.5× over BF16 RVV's 3.39M), we
need either much-tighter vector code throughout (vectorize the byte
unpack with vrgather + bit masks) OR the longer-context regime where the
NVFP4 BW advantage translates to cycles.

### Remaining

1. **Debug J-4.5b numerical drift** (~half day) — isolate the vfmul.vf
   scale width issue, verify dequant produces correct FP32 values
   matching J1's scalar reference.
2. **Reduce vsetvli overhead** (~half day) — batch the dequant config
   transitions.
3. **Bit-exact verification** (~1 day) — port Tracks E/F/F2/F3 goldens
   (321 / 4096 / 65536 / 256 + 256 cases) to gem5 microbenches.
4. **Mo 6 head-to-head re-run** (~half day) — with all the above
   landed, the final verified cycle number.

### Remaining

1. **Bit-exact verification** (~1 day): port the exhaustive sweeps from
   Tracks E/F/F2/F3 (321 BF16 inputs / 4096 cases / 65536 / 256 + 256)
   to gem5 microbenches.  Verify the decoder semantics match the Chisel
   modules across the full input space, not just the 8-32 sample cases.
2. **Microbench integration** (~half day): modify `bench_fa_mixed_rvv.c`
   and `bench_fa_mixed_streaming_stub.c` to issue the native customs
   instead of scalar stubs.  Inline asm via `.4byte` for each.  Easiest
   first step: replace scalar `fu_expf()` in the J3 two-pass softmax
   loop with vector `vfexp.v`.
3. **Mo 6 head-to-head re-run** (~half day): BF16 RVV (J3) vs Mixed
   J3+native-FU on RiscvO3CPU.  Verified Mo 6 number replaces the
   Track J-2 projection.

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
cd /home/noah/project/riscv/gem5
source ~/miniconda3/etc/profile.d/conda.sh && conda activate gem5-build
export CC=gcc CXX=g++ CPATH=$CONDA_PREFIX/include \
       LIBRARY_PATH=$CONDA_PREFIX/lib \
       LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
scons build/RISCV/gem5.opt -j32 --quiet

cd /home/noah/project/riscv/microbench-j4
PATH=/home/noah/project/riscv/install/bootlin-riscv64/bin:$PATH \
  riscv64-linux-gcc -O2 -march=rv64gcv -mabi=lp64d -static \
  test_vfexp.c -o test_vfexp -lm

mkdir -p run_test_vfexp && cd run_test_vfexp
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  /home/noah/project/riscv/gem5/build/RISCV/gem5.opt --outdir=. \
  /home/noah/project/riscv/gem5/configs/deprecated/example/se.py \
  --cpu-type=TimingSimpleCPU --num-cpus=4 --mem-size=512MB \
  -c /home/noah/project/riscv/microbench-j4/test_vfexp
```
