# Track F3 — VFConvFp8Bf16Lane results (2026-05-17)

## Headline

**Track F3 PASS.** 110 LOC Chisel, 1-stage pipeline, 14 ChiselTests pass
including **0/256 mismatches on the exhaustive FP8-input sweep** and
**0/256 non-identity on the F2/F3 round-trip** test. This closes the
conversion-lane trio (Tracks F + F2 + F3) and means **all 4 Saturn precision-
routing custom instructions** (`vfconv.{nvfp4.bf16, bf16.fp8, fp8.bf16}.v` +
`vfexp.v`) now have RTL with bit-exact validation.

| Metric | Value |
|---|---|
| LOC (Chisel) | 110 |
| LOC (test) | 331 |
| Pipeline depth | 1 stage |
| ChiselTests | 14/14 |
| Exhaustive sweep | 256 FP8 inputs, **0 mismatches** vs e4m3-via-Double-to-BF16 golden |
| F2/F3 round-trip | 256 bytes, **0 non-identity** |
| Cumulative project | **57/57** tests across 5 modules |
| Latency vs `fu_sketch.md` spec | matches (1 cycle) |
| Throughput | 1 elt/cycle sustained |
| `@instr` decl | `exo_instr_decls.md §5.3` |

## What this is

Per-lane primitive for `vfconv.fp8.bf16.v`: FP8-E4M3 (OCP FN) → BF16. Used by
the FU at the start of the P · V matmul to dequantise the FP8-stored attention
weights back into BF16 for the multiply-accumulate. Symmetric inverse of
`vfconv.bf16.fp8.v` (Track F2).

The conversion is **mathematically exact** — BF16 has strictly more mantissa
precision (7 bits vs E4M3-FN's 3 bits) and strictly wider exponent range
(8-bit biased vs 4-bit biased), so every finite E4M3 value maps to a unique
BF16. Hence no guard/round/sticky logic; no cascade handling; one pipeline
stage suffices.

## Pipeline

```
s1 (combinational):
  decode FP8 -> (sign, exp_field, mant_field)
  classify: is_nan | is_zero | is_subn | is_normal
  normal:    BF16 = (sign, exp_field + 120, mant_field << 4)
  subnormal: priority-encode MSB of 3-bit mant; shift to fill BF16 mantissa
             {m=1 -> exp 118, mant=0;
              m=2 -> exp 119, mant=0;
              m=3 -> exp 119, mant = m[0] << 6;
              m=4 -> exp 120, mant=0;
              m=5 -> exp 120, mant = m[0] << 5;
              m=6 -> exp 120, mant = m[1] << 6;
              m=7 -> exp 120, mant = m[1:0] << 5}
  zero:      Cat(sign, 0)
  nan:       Cat(sign, 0xFF, 0x40)  -- sign-preserving BF16 QNaN
  mux on (is_nan, is_zero, is_subn, else=is_normal) -> output register

s1 -> output register
```

Latency = 1 cycle fill, 1 elt/cycle sustained. Same skeleton pass-through
backpressure as the other three lanes; skid buffer remains Mo 4 work.

## Numerical validation

Two complementary checks:

1. **Exhaustive vs FP-decoder golden (256 cases, 0 mismatches).** Golden
   uses `e4m3ToDouble` (same decoder as VFConvNvfp4Bf16Lane's E4M3 path),
   special-cases NaN (sign-preserving 0x7FC0 / 0xFFC0) and zero (signed),
   else round-trips via `Float.toFloat` and `floatToBf16Bits`. Since every
   non-NaN E4M3 value fits exactly in BF16, the round-trip is bit-exact.

2. **F2/F3 round-trip (256 cases, 0 non-identity).** For every FP8 byte,
   compose `fp8E4M3FnToBf16` (this lane's semantics) with `bf16ToFp8E4M3Fn`
   (F2's distance-based RNE golden); the result equals the original byte.
   This is the strongest correctness statement available without an
   independent reference — it shows the two software goldens agree, **and**
   that the two DUTs are exact inverses on the entire 256-byte FP8 space.

Coverage includes: all 14 E4M3 normal-exp values for both signs, all 7 E4M3
subnormal-mantissa values for both signs, ±0, ±max (448), ±NaN.

## Bug-of-record during bring-up

**None.** All 14 tests passed on first run. The Chisel decode logic is a
near-copy of the E4M3 scale decoder already validated under Track F (`sc_*`
subset of `VFConvNvfp4Bf16Lane.scala`), so the structural correctness was
inherited; F3 contributes the explicit-lane wrapping plus the round-trip
validation.

## Area / LOC estimate

110 LOC vs the original 250-LOC budget for `VFConvLane.scala` in
`fu_sketch.md`. This is the simplest of the three conversion lanes.
Synthesis target (Mo 4): ~6K NAND2 equivalent at 16nm per `fu_sketch.md`,
≈0.015 mm².

## Cumulative state — all 4 custom instructions DONE

| Instruction | Track | LOC | Pipeline | Validation |
|---|---|---|---|---|
| `vfexp.v` | E | 300 | 10 stages | 3.20e-6 max rel err vs scalar expf over 321 BF16 pts |
| `vfconv.nvfp4.bf16.v` | F | 197 | 3 stages | 0/4096 vs FP64-and-round-to-BF16 golden |
| `vfconv.bf16.fp8.v` | F2 | 192 | 2 stages | 0/65536 vs distance-based RNE golden |
| `vfconv.fp8.bf16.v` | F3 | 110 | 1 stage | 0/256 + 0/256 F2-round-trip |

Plus `PolyExpQ2_30.scala` (95 LOC) shared by vfexp. Project total **904 LOC
Chisel + tests**. Cumulative ChiselTests **57/57**.

Mo 4 area-estimate prerequisites are met: all 4 lanes have realistic LOC and
pipeline depths matching the `fu_sketch.md` projections (which estimated
~54K NAND2 ≈ 0.14 mm² ≈ 3% of Saturn baseline at 16nm).

## Next track recommendation

Conversion-lane trio is closed. Highest-leverage unblocked tracks (per the
handoff list):

- **Track H** (~1 day): Re-run Mo 2 microbench with an FU-latency-stubbed
  dequant path. Bridges the "73.5% BW reduction" headline to a cycle-speedup
  number, which is what Mo 6 / Mo 10 need. F3 gives us the exact latency to
  stub (1 cycle for `vfconv.fp8.bf16.v`; 3 cycles for `vfconv.nvfp4.bf16.v`;
  2 cycles for `vfconv.bf16.fp8.v`).
- **Track D-follow** (~1-2 d): fork `exo-lang/exo`, land the ~30-LOC BF16
  patch + the `SaturnRVV_M{1,2}` memory class, write a 10-line `@proc` that
  type-checks and emits the §5 `@instr`-using C. Brings `exo_instr_decls.md`
  from artifact to compiling-smoke-test; minimum-viable upstream-PR candidate
  is the BF16 type patch alone.
- **Track G** (~2 h): Mo 2 V-cache extension + long-seq sweep.
- **Mo 4 prep** (~2-3 d): Yosys/OpenROAD synthesis at 16nm on all 4 lanes for
  the area-estimate checkpoint. The validated RTL is now ready for this.

## File pointers

- DUT: `saturn-fu/src/main/scala/saturnfu/VFConvFp8Bf16Lane.scala` (110 LOC)
- Tests: `saturn-fu/src/test/scala/saturnfu/VFConvFp8Bf16LaneTest.scala` (331 LOC)
- Run: `cd saturn-fu && PATH=$HOME/.local/share/sbt/bin:$PATH sbt test`
- Spec: `paper/fu_sketch.md §"FP8-E4M3→BF16 dequant lane"` (1-cycle target)
- Compiler-side: `paper/exo_instr_decls.md §5.3` (`@instr` decl)
- Forward direction: `paper/track_f2_results.md` (the F2 quant lane this inverts)
