/*
 * Mo 2 microbench — FP16/BF16 K-cache GEMV baseline.
 *
 * One decode step: out[h][s] = sum_d Q[h][d] * K[h][s][d].
 * K is BF16 (effectively same memory footprint as FP16 — 16 bits/elt).
 *
 * We use BF16 rather than IEEE FP16 because gem5 25.1 SE does NOT model
 * Zvfh/Zfh and BF16 → FP32 conversion is a trivial uint32 shift (no FP
 * extension needed). Memory footprint is identical (2 bytes/elt), so the
 * Mo 2 bandwidth question is preserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "bench_common.h"
#include "m5ops.h"

typedef uint16_t bf16_t;

static inline float bf16_to_fp32(bf16_t b) {
    union { uint32_t u; float f; } u;
    u.u = ((uint32_t)b) << 16;
    return u.f;
}

static inline bf16_t fp32_to_bf16(float f) {
    union { uint32_t u; float f; } u;
    u.f = f;
    return (bf16_t)(u.u >> 16);
}

int main(void) {
    bf16_t *K = (bf16_t *)malloc(K_NUM_ELTS * sizeof(bf16_t));
    float *Q  = (float  *)malloc(N_HEADS * HEAD_DIM * sizeof(float));
    float *out = (float *)malloc(N_HEADS * SEQ_LEN * sizeof(float));
    if (!K || !Q || !out) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* Deterministic init — not counted in the measurement window. */
    for (int h = 0; h < N_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++)
            Q[h * HEAD_DIM + d] =
                ((float)((h * 31 + d * 17) % 7) / 7.0f) - 0.5f;

    for (int h = 0; h < N_HEADS; h++)
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < HEAD_DIM; d++) {
                int idx = (h * SEQ_LEN + s) * HEAD_DIM + d;
                float v = ((float)((idx * 13 + 7) % 11) / 11.0f) - 0.5f;
                K[idx] = fp32_to_bf16(v);
            }

    /* Touch out buffer so its pages are mapped before measurement. */
    memset(out, 0, N_HEADS * SEQ_LEN * sizeof(float));

    /* ===================== MEASUREMENT REGION ===================== */
    m5_dump_reset_stats(0, 0);
    uint64_t cyc_start = rdcycle();

    for (int h = 0; h < N_HEADS; h++) {
        const float *qh = &Q[h * HEAD_DIM];
        const bf16_t *kh = &K[h * SEQ_LEN * HEAD_DIM];
        float *outh = &out[h * SEQ_LEN];
        for (int s = 0; s < SEQ_LEN; s++) {
            float acc = 0.0f;
            const bf16_t *ks = &kh[s * HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; d++) {
                acc += qh[d] * bf16_to_fp32(ks[d]);
            }
            outh[s] = acc;
        }
    }

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    /* Side-effect: prevent the compiler from optimizing away the loop. */
    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * SEQ_LEN; i++) checksum += out[i];

    printf("fp16 bench done\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f\n", checksum);
    printf("  K bytes       = %lu  (BF16)\n",
           (unsigned long)(K_NUM_ELTS * sizeof(bf16_t)));

    free(K); free(Q); free(out);
    return 0;
}
