# microbench-j4 — Per-instruction functional tests for the 4 Saturn customs

Single-instruction smoke + latency tests for the four custom RVV
instructions added to gem5's RISC-V decoder under
[`../gem5/`](../gem5/) `stable` branch (commit `841d376`). Companion to
[`../paper/track_j4_results.md`](../paper/track_j4_results.md), the
gem5 decoder-patch track.

Each test issues the custom encoding via inline `asm volatile (".4byte
...");` on a small vector input and validates output bit-exactly
against a scalar reference (matching the Chisel `*Lane.scala` modules
under [`../saturn-fu/`](../saturn-fu/)).

## Tests

| Source                          | Custom instruction        | Validation                              |
|---------------------------------|---------------------------|-----------------------------------------|
| `test_vfexp.c`                  | `vfexp.v`                 | 8 / 8 bit-exact vs libm `expf`          |
| `test_vfexp_m2.c`               | `vfexp.v`                 | same, at LMUL=2                         |
| `test_vfexp_latency.c`          | `vfexp.v`                 | tight-loop probe — 13.05 cyc/iter on O3 |
| `test_vfconv_fp8_bf16.c`        | `vfconv.fp8.bf16.v`       | 8 / 8 bit-exact (sample mapping)        |
| `test_vfconv_bf16_fp8.c`        | `vfconv.bf16.fp8.v`       | 8 / 8 bit-exact (50+ line RNE)          |
| `test_vfconv_nvfp4_bf16.c`      | `vfconv.nvfp4.bf16.v`     | 16 / 16 bit-exact (full nibble space)   |

## Build

```bash
export PATH=/path/to/bootlin-riscv64/bin:$PATH
make
```

## Run on gem5

```bash
export GEM5_DIR=$PWD/../gem5
./run_remaining_tests.sh                       # vfconv lanes only
# Or per-test:
$GEM5_DIR/build/RISCV/gem5.opt \
  $GEM5_DIR/configs/deprecated/example/se.py \
  --cpu-type=TimingSimpleCPU --num-cpus=4 \
  -c ./test_vfexp
```

`run_remaining_tests.sh` activates the `gem5-build` conda env and
expects `GEM5_DIR` set externally; see [`../README.md`](../README.md).
