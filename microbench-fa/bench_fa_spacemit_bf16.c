/*
 * Track J-2 — SpacemiT K1 FA kernel ported to our gem5 microbench
 * (VLEN=256, BF16 instead of FP16 since gem5 lacks Zvfh/Zvfbfwma).
 *
 * Direct port of `flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1`
 * from upstream llama.cpp `ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp:913`
 * with two adaptations:
 *
 *   1. FP16 -> BF16 + FP32 widen.  The upstream kernel keeps VKQ in FP16
 *      vector reg + does FP16 × FP16 widening multiplies (`vfwmul.vv f32m4`).
 *      gem5 25.1.0.1 has no Zvfh / Zvfbfwma support, so we store K/V/Q
 *      as BF16 (matches our J3 baseline), widen to FP32 via the vle16 ->
 *      vzext.vf2 -> vsll vx 16 -> reinterpret trick, and keep VKQ in FP32
 *      vector regs throughout (no final widen cast at the end).
 *
 *   2. VLEN=1024 -> VLEN=256.  Upstream uses LMUL=2 with FP16 at VLEN=1024
 *      = 128 elts per vector op (enough for the worst-case HEAD_DIM=128
 *      in one op).  Our VLEN=256 + FP32 = 16 elts per m2 op, so
 *      HEAD_DIM=64 = 4 vector ops per row.  We use LMUL=4 (32 elts at
 *      f32m4) to keep this to 2 ops per row, matching SpacemiT's
 *      register-pressure profile (m4 uses 4 vregs of 32 total).
 *
 * The algorithm structure — the actual SpacemiT contribution we're
 * benchmarking — is preserved:
 *   - Online softmax with running max M + running sum S, rescaling VKQ by
 *     `ms = expf(Mold - M)` whenever a new max is seen.
 *   - Single pass over the seq_len dimension, K and V streamed together
 *     one row at a time per `ic` iteration.
 *   - Q vector cached once before the loop, reused across all ic.
 *
 * Unified FP32 init via bench_fa_common.h, same shape as J3
 * (n_heads=8, head_dim=64, seq_len=2048) so cycles + DRAM are directly
 * comparable against bench_fa_bf16_rvv.c (J3 two-pass BF16 RVV) and
 * bench_fa_mixed_rvv_stub.c (J3+stub mixed-prec).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>
#include "bench_fa_common.h"
#include "m5ops.h"

static inline vfloat32m2_t bf16_load_widen_m2(const bf16_t *p, size_t vl) {
    vuint16m1_t bf  = __riscv_vle16_v_u16m1(p, vl);
    vuint32m2_t z   = __riscv_vzext_vf2_u32m2(bf, vl);
    vuint32m2_t shf = __riscv_vsll_vx_u32m2(z, 16, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(shf);
}

static inline vfloat32m4_t bf16_load_widen_m4(const bf16_t *p, size_t vl) {
    vuint16m2_t bf  = __riscv_vle16_v_u16m2(p, vl);
    vuint32m4_t z   = __riscv_vzext_vf2_u32m4(bf, vl);
    vuint32m4_t shf = __riscv_vsll_vx_u32m4(z, 16, vl);
    return __riscv_vreinterpret_v_u32m4_f32m4(shf);
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

    /* VKQ FP32 accumulator: HEAD_DIM=64 = 2 chunks of f32m4 (each m4 = 32 elts). */
    float VKQ_scratch[HEAD_DIM] __attribute__((aligned(64)));
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    /* ===================== MEASUREMENT REGION ===================== */
    m5_dump_reset_stats(0, 0);
    uint64_t cyc_start = rdcycle();

    size_t vl_m4 = __riscv_vsetvl_e32m4(32);   /* = 32 at VLEN=256 SEW=32 */
    size_t vl_m2 = __riscv_vsetvl_e32m2(16);   /* = 16 at VLEN=256 SEW=32 */
    (void)vl_m2;

    for (int h = 0; h < N_HEADS; h++) {
        const bf16_t *qh = &Q[h * HEAD_DIM];
        const bf16_t *kh = &K[h * SEQ_LEN * HEAD_DIM];
        const bf16_t *vh = &V[h * SEQ_LEN * HEAD_DIM];
        bf16_t *oh = &O[h * HEAD_DIM];

        /* SpacemiT-style: cache Q once for the row.  HEAD_DIM=64 = 2 × m4. */
        vfloat32m4_t Q_a = bf16_load_widen_m4(qh + 0,      vl_m4);
        vfloat32m4_t Q_b = bf16_load_widen_m4(qh + vl_m4,  HEAD_DIM - vl_m4);

        /* VKQ accumulator vectors (held in regs across ic iterations). */
        vfloat32m4_t VKQ_a = __riscv_vfmv_v_f_f32m4(0.0f, vl_m4);
        vfloat32m4_t VKQ_b = __riscv_vfmv_v_f_f32m4(0.0f, HEAD_DIM - vl_m4);

        float S = 0.0f;
        float M = -1e30f;

        for (int ic = 0; ic < SEQ_LEN; ic++) {
            const bf16_t *ks = &kh[ic * HEAD_DIM];
            const bf16_t *vs_ptr = &vh[ic * HEAD_DIM];

            /* QK^T: load K, widen, multiply, reduce. */
            vfloat32m4_t K_a = bf16_load_widen_m4(ks + 0,     vl_m4);
            vfloat32m4_t K_b = bf16_load_widen_m4(ks + vl_m4, HEAD_DIM - vl_m4);

            vfloat32m4_t qk_a = __riscv_vfmul_vv_f32m4(K_a, Q_a, vl_m4);
            vfloat32m4_t qk_b = __riscv_vfmul_vv_f32m4(K_b, Q_b, HEAD_DIM - vl_m4);

            vfloat32m1_t s_red = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            s_red = __riscv_vfredusum_vs_f32m4_f32m1(qk_a, s_red, vl_m4);
            s_red = __riscv_vfredusum_vs_f32m4_f32m1(qk_b, s_red, HEAD_DIM - vl_m4);
            float s = __riscv_vfmv_f_s_f32m1_f32(s_red);

            s = s * scale;

            /* Online softmax: track running max M and sum S; rescale VKQ
             * when a new max is seen. */
            const float Mold = M;
            float ms = 1.0f;
            float vs = 1.0f;

            /* Load V before the branch so we can issue it in parallel with
             * the rescale FMUL on hardware that supports it.  Matches the
             * SpacemiT kernel structure. */
            vfloat32m4_t V_a = bf16_load_widen_m4(vs_ptr + 0,     vl_m4);
            vfloat32m4_t V_b = bf16_load_widen_m4(vs_ptr + vl_m4, HEAD_DIM - vl_m4);

            if (s > M) {
                M  = s;
                ms = expf(Mold - M);
                VKQ_a = __riscv_vfmul_vf_f32m4(VKQ_a, ms, vl_m4);
                VKQ_b = __riscv_vfmul_vf_f32m4(VKQ_b, ms, HEAD_DIM - vl_m4);
            } else {
                vs = expf(s - M);
            }

            /* VKQ += vs * V. */
            VKQ_a = __riscv_vfmacc_vf_f32m4(VKQ_a, vs, V_a, vl_m4);
            VKQ_b = __riscv_vfmacc_vf_f32m4(VKQ_b, vs, V_b, HEAD_DIM - vl_m4);

            S = S * ms + vs;
        }

        /* Final normalize: VKQ /= S, store. */
        float S_inv = (S == 0.0f) ? 0.0f : 1.0f / S;
        VKQ_a = __riscv_vfmul_vf_f32m4(VKQ_a, S_inv, vl_m4);
        VKQ_b = __riscv_vfmul_vf_f32m4(VKQ_b, S_inv, HEAD_DIM - vl_m4);

        __riscv_vse32_v_f32m4(VKQ_scratch + 0,     VKQ_a, vl_m4);
        __riscv_vse32_v_f32m4(VKQ_scratch + vl_m4, VKQ_b, HEAD_DIM - vl_m4);

        for (int d = 0; d < HEAD_DIM; d++)
            oh[d] = fp32_to_bf16(VKQ_scratch[d]);
    }

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * HEAD_DIM; i++)
        checksum += bf16_to_fp32(O[i]);

    printf("fa_spacemit_bf16 bench done (SpacemiT m1 algo, BF16, VLEN=256)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f\n", checksum);
    printf("  K bytes       = %lu\n",
           (unsigned long)(K_NUM_ELTS * sizeof(bf16_t)));
    printf("  V bytes       = %lu\n",
           (unsigned long)(K_NUM_ELTS * sizeof(bf16_t)));

    free(K); free(V); free(Q); free(O);
    return 0;
}
