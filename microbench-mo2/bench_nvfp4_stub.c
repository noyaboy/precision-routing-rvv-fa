/*
 * Mo 2 microbench — NVFP4 K-cache dequant + GEMV with FU-latency-stubbed
 * dequant (Track H, 2026-05-17).
 *
 * Identical to bench_nvfp4.c except the SW LUT path is replaced with an
 * arithmetic stub that models the future `vfconv.nvfp4.bf16.v` FU output:
 *   - The byte load (NVFP4 packed K) still happens — essential for BW.
 *   - The scale byte load (E4M3 per-block-16) still happens — essential for BW.
 *   - The LUT lookups (nvfp4_decode[], e4m3_decode[]) are GONE: in the
 *     Saturn-FU integration, the FU computes the dequant directly from the
 *     byte streams; the scalar pipeline never sees the LUT traffic.
 *   - The dequant math is replaced with a uint8->float cast + multiply,
 *     modeling the "1 FU instruction = 1 cycle per element" sustained
 *     throughput of vfconv.nvfp4.bf16.v (Track F: 3-cycle fill, 1-elt/cycle
 *     after fill; on 64-element rows the fill is <5% overhead).
 *
 * The resulting cycle count is the projected Mo 2 outcome on a Saturn FU
 * path WITHOUT modeling the FU's parallel 16-lane execution (which would be
 * additional speedup). The 3-cycle FU latency on Saturn would correspond to
 * 3 sequential scalar instructions on TimingSimpleCPU; on 64-element rows
 * that fill cost is amortized to <5% of the row cycle count and is omitted.
 *
 * Numerical output is intentionally different from bench_nvfp4.c: the BW
 * pattern (loads of K_packed[] and scales[]) is preserved, but the dequant
 * math is a stub.  The checksum is not a correctness check — it's only
 * present to prevent dead-code elimination of the FMA chain.
 *
 * Run config identical to bench_nvfp4.c: TimingSimpleCPU + 32 KiB L1D +
 * 512 KiB L2 + DDR3-1600.  Same kernel shape (n_kv_heads=8, head_dim=64,
 * seq_len=2048; GQA decode for Llama-3.2-1B).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "bench_common.h"
#include "m5ops.h"

int main(void) {
    /* Packed K: 2 elements per byte → K_NUM_ELTS/2 bytes. */
    uint8_t *K_packed = (uint8_t *)malloc(K_NUM_ELTS / 2);
    uint8_t *scales   = (uint8_t *)malloc(K_NUM_BLOCKS);
    float   *Q   = (float *)malloc(N_HEADS * HEAD_DIM * sizeof(float));
    float   *out = (float *)malloc(N_HEADS * SEQ_LEN  * sizeof(float));
    if (!K_packed || !scales || !Q || !out) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    /* Same deterministic init as bench_nvfp4.c so BW patterns match. */
    for (int i = 0; i < (int)(K_NUM_ELTS / 2); i++)
        K_packed[i] = (uint8_t)((i * 13 + 7) & 0xff);
    for (int i = 0; i < (int)K_NUM_BLOCKS; i++) {
        scales[i] = (uint8_t)(0x20 + ((i * 31 + 11) & 0x3f));
    }
    for (int h = 0; h < N_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++)
            Q[h * HEAD_DIM + d] =
                ((float)((h * 31 + d * 17) % 7) / 7.0f) - 0.5f;

    memset(out, 0, N_HEADS * SEQ_LEN * sizeof(float));

    /* ===================== MEASUREMENT REGION ===================== */
    m5_dump_reset_stats(0, 0);
    uint64_t cyc_start = rdcycle();

    for (int h = 0; h < N_HEADS; h++) {
        const float *qh = &Q[h * HEAD_DIM];
        float *outh = &out[h * SEQ_LEN];
        const uint8_t *kh_packed =
            &K_packed[(size_t)h * SEQ_LEN * (HEAD_DIM / 2)];
        const uint8_t *sh =
            &scales[(size_t)h * SEQ_LEN * (HEAD_DIM / NVFP4_BLOCK)];

        for (int s = 0; s < SEQ_LEN; s++) {
            const uint8_t *ks_packed = &kh_packed[s * (HEAD_DIM / 2)];
            const uint8_t *ss        = &sh[s * (HEAD_DIM / NVFP4_BLOCK)];
            float acc = 0.0f;

            for (int blk = 0; blk < HEAD_DIM / NVFP4_BLOCK; blk++) {
                /* FU stub: scale load + 1-cycle "FU produces FP32 scale".
                 * The 0.015625f = 1/64 keeps values O(1) without affecting
                 * the load pattern. */
                uint8_t sc_byte = ss[blk];
                float scale_stub = (float)sc_byte * 0.015625f;

                /* 16 elements per block = 8 packed bytes. */
                for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
                    int d = blk * NVFP4_BLOCK + 2 * b;
                    uint8_t byte = ks_packed[blk * (NVFP4_BLOCK / 2) + b];

                    /* FU stub per nibble: 1 cycle modeling vfconv.nvfp4.bf16.v
                     * output (no LUT load, no E4M3 LUT load — both happen
                     * "inside the FU" on Saturn). */
                    float k0 = (float)(byte & 0x0f)       * scale_stub;
                    float k1 = (float)((byte >> 4) & 0x0f) * scale_stub;

                    acc += qh[d]     * k0;
                    acc += qh[d + 1] * k1;
                }
            }
            outh[s] = acc;
        }
    }

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * SEQ_LEN; i++) checksum += out[i];

    size_t k_bytes  = K_NUM_ELTS / 2;
    size_t sc_bytes = K_NUM_BLOCKS;

    printf("nvfp4_stub bench done (FU-latency-stubbed dequant)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f  (not a correctness check)\n", checksum);
    printf("  K bytes       = %lu  (packed)\n", (unsigned long)k_bytes);
    printf("  scale bytes   = %lu  (E4M3 per block-%d)\n",
           (unsigned long)sc_bytes, NVFP4_BLOCK);
    printf("  K + scale     = %lu\n", (unsigned long)(k_bytes + sc_bytes));

    free(K_packed); free(scales); free(Q); free(out);
    return 0;
}
