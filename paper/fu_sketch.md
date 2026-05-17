# Saturn precision-routing FU — block sketch (2026-05-16 PM)

Anchors the RTL work for merged X+Y Y1. Locked precision config in `/home/noah/project/riscv/paper/precision_config.md`.

## High-level block diagram

```
                       ┌─────────────────────────────────────────────────────┐
                       │              Precision-Routing FU                    │
                       │                                                      │
   vs2 (vec data) ──┬──┼──▶  ┌──────────────┐                                │
                    │  │     │   NVFP4→BF16  │ ───▶ vd (BF16 vec)           │
                    │  │     │  dequant lane │       (decode-time K/V load)  │
   vs1 (scales) ──┐ │  │     │  16 lanes × 3c│                                │
                  │ │  │     └──────────────┘                                │
                  └─┼──┼──▶                                                  │
                    │  │                                                      │
                    ├──┼──▶  ┌──────────────┐                                │
                    │  │     │   BF16→FP8    │ ───▶ vd (FP8 vec)            │
                    │  │     │  quant lane   │     (after softmax normalize) │
                    │  │     │  16 lanes × 2c│                                │
                    │  │     └──────────────┘                                │
                    │  │                                                      │
                    ├──┼──▶  ┌──────────────┐                                │
                    │  │     │   FP8→BF16    │ ───▶ vd (BF16 vec)           │
                    │  │     │  dequant lane │     (attn weights → matmul)   │
                    │  │     │  16 lanes × 1c│                                │
                    │  │     └──────────────┘                                │
                    │  │                                                      │
                    └──┼──▶  ┌──────────────┐                                │
                       │     │     vfexp     │ ───▶ vd (FP32 vec)           │
                       │     │  BF16→FP32    │     (inside softmax kernel)   │
                       │     │  16 lanes × 8c│                                │
                       │     └──────────────┘                                │
                       │            │                                         │
                       │            ▼                                         │
                       │    ┌──────────────┐                                  │
                       │    │ FP32 softmax │                                  │
                       │    │ sum-accum reg│ ◀── (scalar fwd to vfredusum)   │
                       │    └──────────────┘                                  │
                       └─────────────────────────────────────────────────────┘
```

## Custom instruction set (4 new RVV instructions)

| Mnemonic | funct6 | inputs | outputs | latency | LMUL behavior |
|---|---|---|---|---|---|
| `vfconv.nvfp4.bf16.v vd, vs2, vs1` | 6'b111000 | vs2 = NVFP4 vec, vs1 = E4M3 scales (per block-16) | vd = BF16 vec | 3 cyc | vd is 2× wider than vs2 (LMUL doubles) |
| `vfconv.bf16.fp8.v   vd, vs2` | 6'b111001 | vs2 = BF16 vec | vd = FP8-E4M3 vec | 2 cyc | vd is 0.5× width (LMUL halves) |
| `vfconv.fp8.bf16.v   vd, vs2` | 6'b111010 | vs2 = FP8-E4M3 vec | vd = BF16 vec | 1 cyc | vd is 2× wider (LMUL doubles) |
| `vfexp.v             vd, vs2` | 6'b111011 | vs2 = BF16 vec | vd = FP32 vec | 8 cyc | vd is 2× wider (LMUL doubles) |

funct6 codes 111000–111011 are in the vendor-extension area; Saturn has no existing claims at this range (verified per memory's `FunctionalUnitFactory` pattern).

## Per-lane microarchitecture

### NVFP4→BF16 dequant lane (3 cycles, 16 lanes wide)

```
   Cycle 1: parse 4-bit NVFP4 mantissa M (sign | 2-bit exp | 1-bit mantissa)
            parse 8-bit E4M3 scale S (per block-16)
   Cycle 2: expand M to BF16 mantissa via 16-entry LUT (one LUT per lane)
            convert S (E4M3 scale) to BF16
   Cycle 3: vfmul to apply scale: output = expand(M) × convert(S)
   → output BF16 vec
```
LUT contents: 16 BF16 values, one per NVFP4 codepoint (signed 4-bit). Constant ROM, ~16 × 16 b = 256 bits per lane × 16 lanes = 4 Kb. Negligible area.

### BF16→FP8-E4M3 quant lane (2 cycles, 16 lanes wide)

```
   Cycle 1: extract BF16 sign + exp + mantissa
            check exponent in E4M3 range [-9, 8]; saturate if out
   Cycle 2: round-to-nearest-even mantissa (3 bits); pack into E4M3
   → output FP8 vec (8-bit per element)
```
Per-element logic: exp range check + 3-bit rounding. Pure combinational, just registered.

### FP8-E4M3→BF16 dequant lane (1 cycle, 16 lanes wide)

```
   Cycle 1: extract E4M3 sign + exp + mantissa
            expand mantissa from 3 to 7 bits (zero-extend)
            re-bias exp from E4M3 (bias 7) to BF16 (bias 127)
            pack BF16
   → output BF16 vec
```
Simplest of the conversion lanes.

### vfexp lane (8 cycles pipelined, 16 lanes wide, BF16→FP32)

```
   Cycle 1: load BF16 x; widen to FP32 → x_f32
   Cycle 2: range reduce: n = round(x_f32 × log2(e)); r = x_f32 - n × ln(2)
            (clamp n to [-126, 127] for FP32 exponent range)
   Cycle 3: polynomial degree-5 in r (reusing FMA hardware)
            P(r) = 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
            implemented as Horner: ((((1/120 * r + 1/24) * r + 1/6) * r + 1/2) * r + 1) * r + 1
   Cycles 4-6: 5 sequential FMAs (one per Horner step) — pipelined per-lane, parallel across lanes
   Cycle 7: reconstruct: result = P(r) × 2^n via bias-set on FP32 exp field
   Cycle 8: output FP32
```

Compared to `ggml_v_expf_m2`'s ~12 RVV ops (estimated ~20 cycles total on Saturn): **2.5× per-row speedup** at hardware level.

VEXP precedent (Snitch, BF16-only) gives ~6 cycles for similar polynomial. Our 8 is conservative; with aggressive scheduling could hit 6.

### FP32 softmax sum-accumulator

Reuse Saturn's existing scalar register file with a reserved FP32 "softmax sum" register written by `vfredusum` over the FP32 vfexp outputs. No new HW; just a software convention enforced by the Exo-generated code. Total cost: 0 area.

## Area estimate (16nm, rough Synopsys norm)

| Component | NAND2 equiv | mm² @ 16nm |
|---|---|---|
| NVFP4→BF16 dequant (16 lanes) | 8 K | 0.020 |
| BF16→FP8 quant (16 lanes) | 4 K | 0.010 |
| FP8→BF16 dequant (16 lanes) | 6 K | 0.015 |
| vfexp (16 lanes, reuses FMA) | 30 K | 0.080 |
| Pipeline regs + ctrl | 5 K | 0.012 |
| Decoder additions | 1 K | 0.003 |
| **Total FU** | **54 K** | **~0.14 mm²** |

Saturn baseline area: ~5 mm² @ 16nm (estimate from Zhao et al. arXiv:2412.00997). Our FU = **~3% of Saturn baseline**. **Mo 4 threshold ≤10% comfortably met.**

## LOC budget (Chisel)

| File | LOC | Purpose |
|---|---|---|
| `VFExpLane.scala` | 300 | Per-lane vfexp pipeline (range red + poly + recon) |
| `VFConvLane.scala` | 250 | Per-lane conversion variants (3 conversion types) |
| `PrecisionRoutingFU.scala` | 200 | Top FU wrapper, lane parallelism, dispatch |
| `decoder additions in `Saturn/exu/common/Decoder.scala` | 150 | Decode 4 new funct6 → FU port |
| Testbench (`PrecisionRoutingFUTest.scala`) | 500 | Per-instruction unit tests + random vectors |
| **Total** | **~1400** | Within original 1500-3000 estimate. |

## Decoder integration (Saturn pattern from memory)

Per memory's verified Saturn finding: `FunctionalUnitFactory` adds new FUs without dispatch surgery. Pattern:

```scala
// In Saturn/exu/common/Parameters.scala or equivalent
val precisionRoutingFU = FunctionalUnitFactory("PrecisionRoutingFU")
  .withCustomOpcodes(Seq(
    CustomOp("vfconv.nvfp4.bf16.v", funct6 = "111000"),
    CustomOp("vfconv.bf16.fp8.v",   funct6 = "111001"),
    CustomOp("vfconv.fp8.bf16.v",   funct6 = "111010"),
    CustomOp("vfexp.v",             funct6 = "111011"),
  ))
  .withLatency(...)
```

No surgery on issue queue / scoreboard / chaining logic.

## Exo `@instr` declarations (compiler-side)

Grounded declarations are in `paper/exo_instr_decls.md` (Track D, 2026-05-17). That artifact:
- audits the upstream `exo-lang/exo` repo's `platforms/rvv.py` (minimal VLEN=128 f32 stub, 9 intrinsics)
- pins the **type-extension surface** (~30 LOC patch to add BF16 first-class to Exo) and FP8 / NVFP4 carrier convention (opaque `uint8`)
- declares the parametric `SaturnRVV_M{1,2}` memory class
- gives real `@instr` decls for each of the 4 customs that compile against Exo conventions verified against `src/exo/API.py:52`
- spells out the FA tile schedule (`divide_loop` / `replace`) that Mo 5–8 has to land

The original "syntax illustrative" placeholder here is superseded by that file. Funct6 + latency
are not first-class fields on Exo's `@instr` — they live in the Chisel decoder + the inline-asm
macros in `saturn_custom_asm.h`, separate from the compiler-facing semantic body.

## Open design questions (deferred to actual Chisel implementation)

1. **vfexp polynomial degree**: 5 (proposed) vs 4 (faster, less accurate) vs 6 (slower, more accurate). Decide via accuracy sweep on Llama-3.2-1B perplexity.
2. **NVFP4 unpack on load**: explicit `vfconv.nvfp4.bf16` instruction (current proposal) vs streaming dequant during `vle` (would need vle extension). Explicit is simpler; streaming saves a register.
3. **Scale broadcast pattern**: per-block-16 scale carried in `vs1` (current proposal, ~1 byte per 16 NVFP4 values) vs separate stream. Per-block in `vs1` is RVV-idiomatic and keeps cache layout simple.
4. **vfexp output: FP32 vs BF16**: FP32 (current proposal) is safer for softmax sum accumulation. BF16 would halve the output bandwidth but risk dynamic-range issues in sum.
5. **Saturn chaining hookup**: vfexp output feeds `vfredusum` for softmax sum. Need to verify Saturn's chaining can carry FP32 vfexp output → BF16 vfredusum input (need an intermediate widening register).

## Next RTL milestone (Mo 4)

Standalone `VFExpLane.scala` + testbench, verified against scalar `expf()` reference on 10⁶ random BF16 inputs. Target: max relative error <2⁻¹⁶ (1 ULP for FP32 output). Area + cycle estimate confirmed via Yosys synthesis on a 7nm PDK.

If the Mo 4 vfexp lane lands at <10% Saturn area and the relative error fits, the full FU is well within budget.
