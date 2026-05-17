/*
 * Track J3 — Flash-attention decode-step BF16 baseline, RVV-vectorized
 * row-major P*V.
 *
 * Apples-to-apples comparator for bench_fa_mixed_rvv.c (same access pattern,
 * same vectorization strategy, BF16 K/V instead of NVFP4).  Identical
 * unified-FP32 init -> fp32_to_bf16(...) for K/V/Q, RVV-vectorized
 * Q*K dot product / softmax max/sum / P*V FMA chain.
 *
 * This lets the Mo 5/6 cycle delta isolate the precision-routing benefit
 * from the row-major + vectorization benefit (which both kernels get).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>
#include "bench_fa_common.h"
#include "m5ops.h"

static inline vfloat32m2_t bf16_load_widen(const bf16_t *p, size_t vl) {
    vuint16m1_t bf  = __riscv_vle16_v_u16m1(p, vl);
    vuint32m2_t z   = __riscv_vzext_vf2_u32m2(bf, vl);
    vuint32m2_t shf = __riscv_vsll_vx_u32m2(z, 16, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(shf);
}

int main(void) {
    bf16_t *K = (bf16_t *)malloc(K_NUM_ELTS * sizeof(bf16_t));
    bf16_t *V = (bf16_t *)malloc(K_NUM_ELTS * sizeof(bf16_t));
    bf16_t *Q = (bf16_t *)malloc(N_HEADS * HEAD_DIM * sizeof(bf16_t));
    bf16_t *O = (bf16_t *)malloc(N_HEADS * HEAD_DIM * sizeof(bf16_t));
    if (!K || !V || !Q || !O) { fprintf(stderr, "alloc failed\n"); return 1; }

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

    float *S = (float *)malloc(SEQ_LEN * sizeof(float));
    float *P = (float *)malloc(SEQ_LEN * sizeof(float));
    float O_fp32[HEAD_DIM] __attribute__((aligned(64)));
    float V_fp32_row[HEAD_DIM] __attribute__((aligned(64)));
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

        size_t vl_max = __riscv_vsetvl_e32m2(16);

        /* QK^T with RVV widen-and-FMA. */
        for (int s = 0; s < SEQ_LEN; s++) {
            const bf16_t *ks = &kh[s * HEAD_DIM];
            vfloat32m2_t vacc = __riscv_vfmv_v_f_f32m2(0.0f, vl_max);
            for (int d = 0; d < HEAD_DIM; d += (int)vl_max) {
                size_t vl = __riscv_vsetvl_e32m2(HEAD_DIM - d);
                vfloat32m2_t vq = bf16_load_widen(qh + d, vl);
                vfloat32m2_t vk = bf16_load_widen(ks + d, vl);
                vacc = __riscv_vfmacc_vv_f32m2(vacc, vq, vk, vl);
            }
            vfloat32m1_t vred_init = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t vred = __riscv_vfredusum_vs_f32m2_f32m1(vacc, vred_init, vl_max);
            S[s] = __riscv_vfmv_f_s_f32m1_f32(vred) * scale;
        }

        /* Softmax max. */
        vfloat32m1_t vmax_init = __riscv_vfmv_v_f_f32m1(-1e30f, 1);
        vfloat32m1_t vmax_red = vmax_init;
        for (int s = 0; s < SEQ_LEN; s += (int)vl_max) {
            size_t vl = __riscv_vsetvl_e32m2(SEQ_LEN - s);
            vfloat32m2_t vs = __riscv_vle32_v_f32m2(S + s, vl);
            vmax_red = __riscv_vfredmax_vs_f32m2_f32m1(vs, vmax_red, vl);
        }
        float max_s = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

        /* Softmax exp + sum (scalar libm expf — same as J1 baseline). */
        float sum_p = 0.0f;
        for (int s = 0; s < SEQ_LEN; s++) {
            float e = expf(S[s] - max_s);
            P[s] = e;
            sum_p += e;
        }
        float inv_sum = 1.0f / sum_p;

        /* P*V row-major s-outer/d-inner with vector FMA. */
        for (int d = 0; d < HEAD_DIM; d++) O_fp32[d] = 0.0f;

        for (int s = 0; s < SEQ_LEN; s++) {
            float p = P[s] * inv_sum;
            const bf16_t *vs = &vh[s * HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; d += (int)vl_max) {
                size_t vl = __riscv_vsetvl_e32m2(HEAD_DIM - d);
                vfloat32m2_t vo = __riscv_vle32_v_f32m2(O_fp32 + d, vl);
                vfloat32m2_t vv = bf16_load_widen(vs + d, vl);
                vo = __riscv_vfmacc_vf_f32m2(vo, p, vv, vl);
                __riscv_vse32_v_f32m2(O_fp32 + d, vo, vl);
            }
            (void)V_fp32_row;
        }

        for (int d = 0; d < HEAD_DIM; d++)
            oh[d] = fp32_to_bf16(O_fp32[d]);
    }

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * HEAD_DIM; i++)
        checksum += bf16_to_fp32(O[i]);

    printf("fa_bf16_rvv bench done (RVV BF16 baseline, row-major P*V)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f\n", checksum);
    printf("  K bytes       = %lu\n",
           (unsigned long)(K_NUM_ELTS * sizeof(bf16_t)));
    printf("  V bytes       = %lu\n",
           (unsigned long)(K_NUM_ELTS * sizeof(bf16_t)));

    free(K); free(V); free(Q); free(O); free(S); free(P);
    return 0;
}
