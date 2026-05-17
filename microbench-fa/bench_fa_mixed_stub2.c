/*
 * Track J2 — Flash-attention decode-step, AGGRESSIVE FU-latency stub
 * (per `feedback-fu-stub-brackets` and Track H precedent).
 *
 * Same shape as bench_fa_mixed_stub.c (conservative), but the per-element
 * FMUL with scale is stripped — the FU is assumed to produce a fully-formed
 * scaled BF16/FP32 output.  Scale bytes are still loaded (Saturn fetches
 * them for the FU; we need the BW load), preserved via asm-volatile clobber
 * so the compiler can't dead-eliminate them now that no SW computation
 * depends on the value.
 *
 *   vfconv.nvfp4.bf16.v   ->   uint8->float cast only (no scale FMUL)
 *   vfconv.bf16.fp8.v     ->   1-op trunc only (no pre-scale FMUL)
 *   vfconv.fp8.bf16.v     ->   uint8->float cast only
 *   vfexp.v               ->   inline polynomial expf (same as conservative)
 *
 * The pair (this + stub.c) brackets the cycle answer: stub.c over-counts
 * (scalar FMUL not parallelized by the FU's 16-lane datapath), stub2.c
 * under-counts (an FMUL has to land somewhere in scalar pipeline).  Truth
 * is between, per `feedback-fu-stub-brackets`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bench_fa_common.h"
#include "m5ops.h"

static float e4m3_decode[256];

static void quantize_one_block_kv(int h, int s, int blk,
                                  float (*src_fp32)(int, int, int),
                                  uint8_t *pk_dst,
                                  uint8_t *sc_dst) {
    float block[NVFP4_BLOCK];
    for (int i = 0; i < NVFP4_BLOCK; i++)
        block[i] = src_fp32(h, s, blk * NVFP4_BLOCK + i);
    quantize_nvfp4_block(block, e4m3_decode, pk_dst, sc_dst);
}

static inline float fu_expf(float x) {
    const float LN2_INV = 1.4426950408889634f;
    const float LN2     = 0.6931471805599453f;
    float n_f = floorf(x * LN2_INV + 0.5f);
    float r   = x - n_f * LN2;
    float p   = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f
                  + r * (0.04166667f + r * 0.00833333f))));
    union { float f; uint32_t u; } v;
    int n = (int)n_f;
    if (n >  127) n =  127;
    if (n < -126) n = -126;
    v.u = (uint32_t)((n + 127) << 23);
    return p * v.f;
}

int main(void) {
    e4m3_init_table(e4m3_decode);

    uint8_t *K_pk = (uint8_t *)malloc(K_PACKED_BYTES);
    uint8_t *K_sc = (uint8_t *)malloc(K_SCALE_BYTES);
    uint8_t *V_pk = (uint8_t *)malloc(K_PACKED_BYTES);
    uint8_t *V_sc = (uint8_t *)malloc(K_SCALE_BYTES);
    bf16_t  *Q    = (bf16_t  *)malloc(N_HEADS * HEAD_DIM * sizeof(bf16_t));
    bf16_t  *O    = (bf16_t  *)malloc(N_HEADS * HEAD_DIM * sizeof(bf16_t));
    if (!K_pk || !K_sc || !V_pk || !V_sc || !Q || !O) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    for (int h = 0; h < N_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++)
            Q[h * HEAD_DIM + d] = fp32_to_bf16(init_q_fp32(h, d));

    const int blocks_per_row = HEAD_DIM / NVFP4_BLOCK;
    const int bytes_per_row  = HEAD_DIM / 2;
    for (int h = 0; h < N_HEADS; h++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            uint8_t *kh_pk = &K_pk[((size_t)h * SEQ_LEN + s) * bytes_per_row];
            uint8_t *kh_sc = &K_sc[((size_t)h * SEQ_LEN + s) * blocks_per_row];
            uint8_t *vh_pk = &V_pk[((size_t)h * SEQ_LEN + s) * bytes_per_row];
            uint8_t *vh_sc = &V_sc[((size_t)h * SEQ_LEN + s) * blocks_per_row];
            for (int blk = 0; blk < blocks_per_row; blk++) {
                quantize_one_block_kv(h, s, blk, init_k_fp32,
                                      kh_pk + blk * (NVFP4_BLOCK / 2),
                                      kh_sc + blk);
                quantize_one_block_kv(h, s, blk, init_v_fp32,
                                      vh_pk + blk * (NVFP4_BLOCK / 2),
                                      vh_sc + blk);
            }
        }
    }

    memset(O, 0, N_HEADS * HEAD_DIM * sizeof(bf16_t));

    float *S      = (float *)malloc(SEQ_LEN * sizeof(float));
    float *P_fp32 = (float *)malloc(SEQ_LEN * sizeof(float));
    uint8_t *P_fp8 = (uint8_t *)malloc(SEQ_LEN);
    if (!S || !P_fp32 || !P_fp8) { fprintf(stderr, "alloc failed\n"); return 1; }
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    /* ===================== MEASUREMENT REGION ===================== */
    m5_dump_reset_stats(0, 0);
    uint64_t cyc_start = rdcycle();

    for (int h = 0; h < N_HEADS; h++) {
        const bf16_t  *qh = &Q[h * HEAD_DIM];
        const uint8_t *kh_pk = &K_pk[(size_t)h * SEQ_LEN * (HEAD_DIM / 2)];
        const uint8_t *kh_sc = &K_sc[(size_t)h * SEQ_LEN * (HEAD_DIM / NVFP4_BLOCK)];
        const uint8_t *vh_pk = &V_pk[(size_t)h * SEQ_LEN * (HEAD_DIM / 2)];
        const uint8_t *vh_sc = &V_sc[(size_t)h * SEQ_LEN * (HEAD_DIM / NVFP4_BLOCK)];
        bf16_t *oh = &O[h * HEAD_DIM];

        /* QK^T: aggressive — scale byte load preserved via asm clobber. */
        for (int s = 0; s < SEQ_LEN; s++) {
            const uint8_t *ks_pk = &kh_pk[s * (HEAD_DIM / 2)];
            const uint8_t *ks_sc = &kh_sc[s * (HEAD_DIM / NVFP4_BLOCK)];
            float acc = 0.0f;
            for (int blk = 0; blk < HEAD_DIM / NVFP4_BLOCK; blk++) {
                uint8_t sc_byte = ks_sc[blk];
                asm volatile ("" :: "r"((uint32_t)sc_byte) : );
                for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
                    int d = blk * NVFP4_BLOCK + 2 * b;
                    uint8_t byte = ks_pk[blk * (NVFP4_BLOCK / 2) + b];
                    /* Aggressive: 1-op cast only.  FU output already scaled. */
                    float k0 = (float)(byte & 0x0f);
                    float k1 = (float)((byte >> 4) & 0x0f);
                    acc += bf16_to_fp32(qh[d])     * k0;
                    acc += bf16_to_fp32(qh[d + 1]) * k1;
                }
            }
            S[s] = acc * scale;
        }

        float max_s = S[0];
        for (int s = 1; s < SEQ_LEN; s++) if (S[s] > max_s) max_s = S[s];

        float sum_p = 0.0f;
        for (int s = 0; s < SEQ_LEN; s++) {
            float e = fu_expf(S[s] - max_s);
            P_fp32[s] = e;
            sum_p += e;
        }
        float inv_sum = 1.0f / sum_p;
        const float row_dequant_scale = inv_sum / 448.0f;

        /* vfconv.bf16.fp8.v aggressive stub: 1-op trunc, FU absorbs the
         * pre-scale FMUL into its input stage.  Use asm clobber so the
         * compiler can't dead-elim the P_fp32 read. */
        for (int s = 0; s < SEQ_LEN; s++) {
            float pf = P_fp32[s];
            asm volatile ("" :: "f"(pf) : );
            P_fp8[s] = (uint8_t)((int)pf & 0xff);
        }

        /* P_fp8 * V_nvfp4: aggressive V dequant + FP8 decode = 1 cast each. */
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int s = 0; s < SEQ_LEN; s++) {
                int blk = d / NVFP4_BLOCK;
                int b   = (d - blk * NVFP4_BLOCK) / 2;
                int hi  = (d - blk * NVFP4_BLOCK) & 1;
                uint8_t byte = vh_pk[s * (HEAD_DIM / 2) + blk * (NVFP4_BLOCK / 2) + b];
                int nibble = hi ? ((byte >> 4) & 0xf) : (byte & 0xf);
                uint8_t vs_byte = vh_sc[s * (HEAD_DIM / NVFP4_BLOCK) + blk];
                asm volatile ("" :: "r"((uint32_t)vs_byte) : );

                /* Aggressive V dequant: 1 cast (FU output already scaled). */
                float v = (float)nibble;
                /* Aggressive vfconv.fp8.bf16.v: 1 cast. */
                float p = (float)P_fp8[s];

                acc += p * v;
            }
            oh[d] = fp32_to_bf16(acc * row_dequant_scale);
        }
    }

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * HEAD_DIM; i++)
        checksum += bf16_to_fp32(O[i]);

    printf("fa_mixed_stub2 bench done (AGGRESSIVE FU-latency stub)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f  (NOT a correctness check)\n", checksum);
    printf("  KV total      = %lu\n",
           (unsigned long)(2 * (K_PACKED_BYTES + K_SCALE_BYTES)));

    free(K_pk); free(K_sc); free(V_pk); free(V_sc);
    free(Q); free(O); free(S); free(P_fp32); free(P_fp8);
    return 0;
}
