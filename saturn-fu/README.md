# saturn-fu

Standalone Chisel project for the precision-routing FU that extends the
Berkeley Saturn vector unit with four new RVV custom instructions for
mixed-precision flash attention. Companion to the paper draft in
`../paper/paper_draft.md` (MLArchSys 2027 submission).

The four instructions and their RTL state:

| Instruction              | Module                          | Stages | Tests | Validation                                                    |
|--------------------------|---------------------------------|-------:|------:|---------------------------------------------------------------|
| `vfexp.v`                | `VFExpLane.scala`               |     10 |     6 | 321-pt BF16 sweep, max rel err 3.20e-6 vs scalar `expf`       |
| `vfconv.nvfp4.bf16.v`    | `VFConvNvfp4Bf16Lane.scala`     |      3 |    12 | 0 / 4 096 mismatches vs FP64-multiply-and-BF16-round golden   |
| `vfconv.bf16.fp8.v`      | `VFConvBf16Fp8Lane.scala`       |      2 |    21 | 0 / 65 536 mismatches vs distance-based RNE golden            |
| `vfconv.fp8.bf16.v`      | `VFConvFp8Bf16Lane.scala`       |      1 |    14 | 0 / 256 + 0 / 256 F2/F3 round-trip non-identity               |

Plus `PolyExpQ2_30.scala` (5-stage Horner FMA polynomial in Q2.30
signed fixed-point) shared by `VFExpLane`. Cumulative **57/57
ChiselTests pass**. Versions deliberately match the Chipyard pin
(Chisel 6.7.0, ChiselTest 6.0.0, Scala 2.13.16) so the modules drop
into Saturn's `FunctionalUnitFactory` without churn.

## Build + test

```bash
# sbt 1.10.6, installed no-sudo by extracting the tarball under ~/.local/share/sbt/
PATH=$HOME/.local/share/sbt/bin:$PATH sbt test       # ~30s after first run; 57/57 pass
```

First run downloads Chisel + ChiselTest + Scala from Maven (~150 MB).

## Yosys synthesis + 16 nm area estimate

`Emit.scala` emits firtool-compatible SystemVerilog for each module
into `build/<ModuleName>/<ModuleName>.sv` (with
`disallowLocalVariables,disallowPackedArrays` so Yosys 0.9's frontend
parses cleanly). `build/synth_all.sh` then runs Yosys on each module
and dumps NAND2-equivalent stats; `build/nand2_count.py` projects to
16 nm area at 0.26 µm² / NAND2 (TSMC-16FF+-class).

```bash
PATH=$HOME/.local/share/sbt/bin:$PATH sbt "runMain saturnfu.Emit"
cd build && ./synth_all.sh && python3 nand2_count.py
```

Result (verified): **0.317 mm² standalone (6.34 % of Saturn 5 mm²)**
or **0.055 mm² FMA-shared (1.10 %)** at 16 SIMD lanes — both
comfortably under the ≤ 10 % threshold typical for vector-pipeline
extensions. Caveats: Yosys 0.9 is ~2× conservative vs commercial DC
on multiplier-heavy designs; the generic library is not a real 16 nm
PDK; no timing closure performed. Full writeup: `../paper/track_i_results.md`.

## Integration with Saturn

This project intentionally does not depend on Chipyard / Saturn — it
isolates the four lanes for unit-level bring-up and bit-exact
validation. To integrate, copy the five Scala source files into
`chipyard/generators/saturn/src/main/scala/exu/fp/` and wire via
`FunctionalUnitFactory.withCustomOpcodes(Seq(...))`. The op-class
tags + latencies expected by Saturn are in
`../paper/paper_draft.md` § 4.1, Table 1; the encoding scheme
(OPFVV.VFUNCT6=0x13.VS1 ∈ {0x06, 0x07, 0x08, 0x09}) is in the same
section. The gem5 5.1 decoder + FU-latency patch that mirrors this
integration lives in `../gem5/` on the `stable` branch (decoder file:
`src/arch/riscv/isa/decoder.isa`).

## Layout

```
saturn-fu/
├── build.sbt                                  Chisel 6.7.0 + ChiselTest 6.0.0
├── build/
│   ├── nand2_count.py                         NAND2-eq → 16 nm area projection
│   └── synth_all.sh                           Yosys per-module synth driver
├── project/build.properties                   sbt 1.10.6
└── src/
    ├── main/scala/saturnfu/
    │   ├── Emit.scala                         firtool / SV emit driver
    │   ├── PolyExpQ2_30.scala                 5-stage Horner FMA polynomial
    │   ├── VFExpLane.scala                    vfexp.v lane
    │   ├── VFConvNvfp4Bf16Lane.scala          NVFP4 K/V dequant lane
    │   ├── VFConvBf16Fp8Lane.scala            FP8 attention-weights quant
    │   └── VFConvFp8Bf16Lane.scala            FP8 attention-weights dequant
    └── test/scala/saturnfu/                   57 ChiselTests, exhaustive sweeps
```
