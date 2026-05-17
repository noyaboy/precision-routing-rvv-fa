# microbench-fa — Mixed-precision flash-attention decode-step microbench

The headline measurement of the paper: a single fused FA decode kernel
across BF16-baseline / mixed-software / mixed-RVV / FU-integrated
variants, swept over `seq_len ∈ {2 K, 4 K, 8 K, 16 K}` and
`head_dim ∈ {64, 128}`. Companion to
[`../paper/paper_draft.md`](../paper/paper_draft.md) § 6 (kernel) + § 7
(evaluation). Per-track measurement writeups:

| Writeup                                                    | Coverage                                                      |
|------------------------------------------------------------|---------------------------------------------------------------|
| [`../paper/track_j_results.md`](../paper/track_j_results.md)   | J1 / J2 / J3 iteration — kernel bring-up + FU stub bracket   |
| [`../paper/track_j2_results.md`](../paper/track_j2_results.md) | SpacemiT BF16 port + long-context O3 sweep                    |
| [`../paper/track_j4_results.md`](../paper/track_j4_results.md) | Native FU integration via the gem5 decoder patch              |

## Bench variants

`bench_fa_common.h` carries the shared kernel shape (8 KV-heads, `head_dim = 64`,
NVFP4 block-16 with E4M3 scales). Each `bench_fa_*.c` is one variant:

| Source                                | Variant                                                                 |
|---------------------------------------|-------------------------------------------------------------------------|
| `bench_fa_bf16.c`                     | Scalar BF16 baseline                                                    |
| `bench_fa_bf16_rvv.c`                 | BF16 RVV row-major (J3 baseline — the cycle reference for Mo 6)         |
| `bench_fa_mixed.c`                    | Scalar mixed-prec (NVFP4 K/V + BF16 logits + FP32 softmax + FP8 weights) |
| `bench_fa_mixed_stub.c` / `_stub2.c`  | J2 conservative + aggressive FU-latency stubs                            |
| `bench_fa_mixed_rvv.c`                | RVV mixed-prec, software dequant (J3 mixed-proper)                       |
| `bench_fa_mixed_rvv_stub.c`           | RVV mixed-prec + FP8-quant cycle stub                                    |
| `bench_fa_mixed_rvv_native.c`         | J-4.5b native FU integration (vfconv.nvfp4.bf16.v + vfexp.v)             |
| `bench_fa_mixed_rvv_native_allfu.c`   | J-4.5d sidebar — all four FU lanes wired (incl. vfconv.bf16.fp8.v)       |
| `bench_fa_mixed_rvv_native_dq.c`      | Variant with dedicated dequant pre-pass (sanity-check)                   |
| `bench_fa_spacemit_bf16.c`            | SpacemiT K1 FP16 m1 path ported to VLEN=256 (Mo 6 SpacemiT reference)    |
| `bench_fa_mixed_streaming_stub.c`     | SpacemiT algorithm + NVFP4 K/V + aggressive FU stub                      |

## Build

```bash
# Default toolchain: Bootlin GCC 14.2 (riscv64-linux-gnu). The
# `-fno-tree-vectorize` flag is REQUIRED on GCC 14 -O2 for the native
# bench (see ../paper/gcc_bug_report.md) and a < 2 % no-op elsewhere.
export PATH=/path/to/bootlin-riscv64/bin:$PATH       # GCC 14.2+
make                                                  # builds 11 benches
```

Fallback to GCC 13.2 (older measurements) with the pre-widen-Q workaround
already in the source:

```bash
export PATH=/path/to/older-bootlin-riscv64/bin:$PATH  # GCC 13.2
make CFLAGS="-O2 -march=rv64gcv -mabi=lp64d -Wall -static -Wno-unused-result -I../microbench-mo2"
```

## Run on gem5

```bash
export GEM5_DIR=$PWD/../gem5
$GEM5_DIR/build/RISCV/gem5.opt \
  $GEM5_DIR/configs/deprecated/example/se.py \
  --cpu-type=RiscvO3CPU --num-cpus=4 \
  -c ./bench_fa_mixed_rvv_native
```

For the full L-sweep + hd-sweep run matrix, see § "Recipes" in
[`../paper/track_j4_results.md`](../paper/track_j4_results.md).
