# microbench-mo2 — NVFP4 vs BF16 K-cache DRAM-bandwidth microbench

Companion to the **Mo 2** validation checkpoint in
[`../paper/paper_draft.md`](../paper/paper_draft.md) § 7.4: *does NVFP4
K/V storage reduce decode DRAM bandwidth by ≥ 30 % vs BF16?* Verdict:
**PASS at 73.5 % reduction** (3.77× compression). Full writeup with
methodology, raw numbers, and caveats: [`../paper/mo2_results.md`](../paper/mo2_results.md).

Track H follow-up adds FU-latency-stubbed dequant variants
(`bench_nvfp4_stub.c`, `bench_nvfp4_stub2.c`) that resolve the cycle
caveat — full discussion in [`../paper/track_h_results.md`](../paper/track_h_results.md).

## Benches

| Source                  | Variant                                                            |
|-------------------------|--------------------------------------------------------------------|
| `bench_fp16.c`          | BF16 K-cache baseline                                              |
| `bench_nvfp4.c`         | NVFP4 K-cache with SW LUT dequant                                  |
| `bench_nvfp4_stub.c`    | Track H conservative FU-latency stub                               |
| `bench_nvfp4_stub2.c`   | Track H aggressive FU-latency stub (matches Track F semantics)     |

## Build

```bash
# Bootlin GCC 13.2 or 14.2+ riscv64-linux-gnu cross-toolchain on PATH
export PATH=/path/to/bootlin-riscv64/bin:$PATH
make
```

Outputs four static `rv64gcv` ELFs.

## Run on gem5 SE

```bash
# Activate gem5-build conda env first; see ../README.md
GEM5_DIR=$PWD/../gem5
$GEM5_DIR/build/RISCV/gem5.opt \
  $GEM5_DIR/configs/deprecated/example/se.py \
  --cpu-type=TimingSimpleCPU --num-cpus=4 \
  -c ./bench_nvfp4
```

Full reproduce recipe (DDR3-1600 + L1D/L2 sizes + run-dir conventions):
[`../paper/mo2_results.md`](../paper/mo2_results.md).
