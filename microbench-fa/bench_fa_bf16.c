/*
 * Track J — Flash-attention decode-step baseline (BF16 throughout).
 *
 * Single decode step: each of N_HEADS heads runs an independent
 *   QK^T -> softmax -> P*V
 * over (Q[1, head_dim], K[seq_len, head_dim], V[seq_len, head_dim]) and writes
 * O[head_dim].  All tensors in BF16; intermediate accumulators in FP32.
 *
 * This is the Mo 5/6 baseline against which the mixed-precision Saturn-FU
 * kernel (bench_fa_mixed.c) is compared.  Both kernels use the same shape and
 * the same scalar pseudo-deterministic data so the cycle / BW comparison is
 * apples-to-apples on the gem5 TimingSimpleCPU we used for Mo 2 + Track H.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bench_fa_common.h"
#include "m5ops.h"

int main(void) {
    bf16_t *K = (bf16_t *)malloc(K_NUM_ELTS * sizeof(bf16_t));
    bf16_t *V = (bf16_t *)malloc(K_NUM_ELTS * sizeof(bf16_t));
    bf16_t *Q = (bf16_t *)malloc(N_HEADS * HEAD_DIM * sizeof(bf16_t));
    bf16_t *O = (bf16_t *)malloc(N_HEADS * HEAD_DIM * sizeof(bf16_t));
    if (!K || !V || !Q || !O) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* Init from shared FP32 ground truth (bench_fa_common.h). */
    for (int h = 0; h < N_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++)
            Q[h * HEAD_DIM + d] = fp32_to_bf16(init_q_fp32(h, d));

    for (int h = 0; h < N_HEADS; h++)
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < HEAD_DIM; d++) {
                int idx = (h * SEQ_LEN + s) * HEAD_DIM + d;
                K[idx] = fp32_to_bf16(init_k_fp32(h, s, d));
                V[idx] = fp32_to_bf16(init_v_fp32(h, s, d));
            }

    memset(O, 0, N_HEADS * HEAD_DIM * sizeof(bf16_t));

    /* Scratch logits + attn-weight buffer (small, reused per head). */
    float *S = (float *)malloc(SEQ_LEN * sizeof(float));     /* per-head logits */
    float *P = (float *)malloc(SEQ_LEN * sizeof(float));     /* per-head softmax */
    if (!S || !P) { fprintf(stderr, "alloc failed\n"); return 1; }
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    /* ===================== MEASUREMENT REGION ===================== */
    m5_dump_reset_stats(0, 0);
    uint64_t cyc_start = rdcycle();

    for (int h = 0; h < N_HEADS; h++) {
        const bf16_t *qh = &Q[h * HEAD_DIM];
        const bf16_t *kh = &K[h * SEQ_LEN * HEAD_DIM];
        const bf16_t *vh = &V[h * SEQ_LEN * HEAD_DIM];
        bf16_t *oh = &O[h * HEAD_DIM];

        /* QK^T -> S, with FP32 accumulation per dot product. */
        for (int s = 0; s < SEQ_LEN; s++) {
            const bf16_t *ks = &kh[s * HEAD_DIM];
            float acc = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++)
                acc += bf16_to_fp32(qh[d]) * bf16_to_fp32(ks[d]);
            S[s] = acc * scale;
        }

        /* Online softmax: max, exp, sum. */
        float max_s = S[0];
        for (int s = 1; s < SEQ_LEN; s++) if (S[s] > max_s) max_s = S[s];

        float sum_p = 0.0f;
        for (int s = 0; s < SEQ_LEN; s++) {
            float e = expf(S[s] - max_s);
            P[s] = e;
            sum_p += e;
        }
        float inv_sum = 1.0f / sum_p;

        /* P * V -> O, with FP32 accumulation per output channel. */
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int s = 0; s < SEQ_LEN; s++) {
                float p = P[s] * inv_sum;
                acc += p * bf16_to_fp32(vh[s * HEAD_DIM + d]);
            }
            oh[d] = fp32_to_bf16(acc);
        }
    }

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    /* Side-effect: prevent dead-code elimination + sanity-check value range. */
    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * HEAD_DIM; i++)
        checksum += bf16_to_fp32(O[i]);

    printf("fa_bf16 bench done\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f\n", checksum);
    printf("  K bytes       = %lu\n",
           (unsigned long)(K_NUM_ELTS * sizeof(bf16_t)));
    printf("  V bytes       = %lu\n",
           (unsigned long)(K_NUM_ELTS * sizeof(bf16_t)));

    free(K); free(V); free(Q); free(O); free(S); free(P);
    return 0;
}
