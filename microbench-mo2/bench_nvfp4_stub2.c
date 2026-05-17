/*
 * Mo 2 microbench — NVFP4 K-cache, "aggressive" FU stub variant (Track H).
 *
 * Same as bench_nvfp4_stub.c except the scalar FMUL with the scale is removed
 * entirely.  The real Saturn `vfconv.nvfp4.bf16.v` FU emits BF16 values that
 * are ALREADY scaled by the E4M3 scale (the FU does both the dequant and the
 * scale-multiply internally, 16 lanes parallel).  On scalar TimingSimpleCPU,
 * an honest model of that is: byte load (essential for BW) + 1 instruction
 * per element representing the FU output (no SW multiply) + FMA.
 *
 * The scale byte load still happens (it must — Saturn fetches it for the FU),
 * but we use an inline-asm clobber to prevent the compiler from
 * dead-eliminating the load now that no SW computation depends on it.
 *
 * Compared to stub-v1 (bench_nvfp4_stub.c), this is one fewer FMUL per
 * element.  The truth lies between the two: stub-v1 over-counts (scalar
 * FMUL not parallelized) and stub-v2 under-counts (an FMUL has to land
 * somewhere in the scalar pipeline).  The pair brackets the cycle answer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "bench_common.h"
#include "m5ops.h"

int main(void) {
    uint8_t *K_packed = (uint8_t *)malloc(K_NUM_ELTS / 2);
    uint8_t *scales   = (uint8_t *)malloc(K_NUM_BLOCKS);
    float   *Q   = (float *)malloc(N_HEADS * HEAD_DIM * sizeof(float));
    float   *out = (float *)malloc(N_HEADS * SEQ_LEN  * sizeof(float));
    if (!K_packed || !scales || !Q || !out) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

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
                /* Scale byte load happens; prevent dead-elim via asm clobber. */
                uint8_t sc_byte = ss[blk];
                asm volatile ("" :: "r"((uint32_t)sc_byte) : );

                for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
                    int d = blk * NVFP4_BLOCK + 2 * b;
                    uint8_t byte = ks_packed[blk * (NVFP4_BLOCK / 2) + b];

                    /* "FU output" stub: 1 op per element (the FU does the
                     * dequant + scale-multiply internally; we just see a
                     * BF16-equivalent FP32 here). */
                    float k0 = (float)(byte & 0x0f);
                    float k1 = (float)((byte >> 4) & 0x0f);

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

    printf("nvfp4_stub2 bench done (aggressive FU stub)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f  (not a correctness check)\n", checksum);
    printf("  K + scale     = %lu\n", (unsigned long)(k_bytes + sc_bytes));

    free(K_packed); free(scales); free(Q); free(out);
    return 0;
}
