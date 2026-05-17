/*
 * Track J2 — Flash-attention decode-step, CONSERVATIVE FU-latency stub
 * (per `feedback-fu-stub-brackets` and Track H precedent).
 *
 * Identical layout to bench_fa_mixed.c (J1 unified init, per-row FP8 scale),
 * but the 4 SW-modeled FU lanes + libm expf are stubbed with arithmetic that
 * preserves the per-element scalar FMUL the FU produces a partial result for.
 *
 *   vfconv.nvfp4.bf16.v   LUT2-load + FMUL   ->   uint8->float cast + FMUL with scale stub
 *   vfconv.bf16.fp8.v     fp32->bf16 + RNE10 ->   fp32 * FP8_MAX + 1-op trunc
 *   vfconv.fp8.bf16.v     LUT-load           ->   uint8->float cast
 *   vfexp.v               libm expf          ->   inline degree-5 polynomial expf
 *
 * Memory pattern (byte/scale loads) is preserved verbatim — only the inner
 * scalar ops change.  Numerical output is intentionally different from
 * bench_fa_mixed.c; checksum is dead-elim guard, not a correctness check.
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

/* Inline polynomial expf (~12 ops). Models vfexp.v: range-reduce + degree-5
 * polynomial in [-ln2/2, ln2/2] + 2^N scale.  Accuracy ~2^-18; on Saturn
 * the equivalent is a 10-cycle pipeline + 1 elt/cycle sustained (Track E).
 * On TimingSimpleCPU this models that throughput closely. */
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

    /* Scale-stub coefficient: arbitrary constant that keeps values O(1)
     * without affecting load pattern.  The 0.015625 = 1/64 matches the Track
     * H stub-v1 conventions. */
    const float STUB_SCALE_COEF = 0.015625f;
    const float FP8_E4M3_MAX = 448.0f;

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

        /* QK^T with FU-stubbed NVFP4 K dequant. */
        for (int s = 0; s < SEQ_LEN; s++) {
            const uint8_t *ks_pk = &kh_pk[s * (HEAD_DIM / 2)];
            const uint8_t *ks_sc = &kh_sc[s * (HEAD_DIM / NVFP4_BLOCK)];
            float acc = 0.0f;
            for (int blk = 0; blk < HEAD_DIM / NVFP4_BLOCK; blk++) {
                uint8_t sc_byte = ks_sc[blk];
                float ksc_stub = (float)sc_byte * STUB_SCALE_COEF;
                for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
                    int d = blk * NVFP4_BLOCK + 2 * b;
                    uint8_t byte = ks_pk[blk * (NVFP4_BLOCK / 2) + b];
                    /* Conservative: 1-op cast for FU nibble decode + 1-op
                     * FMUL for scale (FU produces partial; scalar finishes). */
                    float k0 = (float)(byte & 0x0f) * ksc_stub;
                    float k1 = (float)((byte >> 4) & 0x0f) * ksc_stub;
                    acc += bf16_to_fp32(qh[d])     * k0;
                    acc += bf16_to_fp32(qh[d + 1]) * k1;
                }
            }
            S[s] = acc * scale;
        }

        /* Softmax max + exp via inline polynomial expf (vfexp.v stub). */
        float max_s = S[0];
        for (int s = 1; s < SEQ_LEN; s++) if (S[s] > max_s) max_s = S[s];

        float sum_p = 0.0f;
        for (int s = 0; s < SEQ_LEN; s++) {
            float e = fu_expf(S[s] - max_s);     /* vfexp.v stub */
            P_fp32[s] = e;
            sum_p += e;
        }
        float inv_sum = 1.0f / sum_p;
        const float row_dequant_scale = inv_sum / FP8_E4M3_MAX;

        /* vfconv.bf16.fp8.v stub: conservative keeps the FMUL with FP8_MAX
         * (scalar pre-scale work the FU's input multiplexer doesn't absorb)
         * and stubs the bit-level RNE as a single truncation. */
        for (int s = 0; s < SEQ_LEN; s++) {
            float pf = P_fp32[s] * FP8_E4M3_MAX;
            int   pi = (int)pf;
            P_fp8[s] = (uint8_t)(pi & 0xff);
        }

        /* P_fp8 * V_nvfp4 -> O with FU-stubbed V dequant + FP8 decode. */
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int s = 0; s < SEQ_LEN; s++) {
                int blk = d / NVFP4_BLOCK;
                int b   = (d - blk * NVFP4_BLOCK) / 2;
                int hi  = (d - blk * NVFP4_BLOCK) & 1;
                uint8_t byte = vh_pk[s * (HEAD_DIM / 2) + blk * (NVFP4_BLOCK / 2) + b];
                int nibble = hi ? ((byte >> 4) & 0xf) : (byte & 0xf);
                uint8_t vs_byte = vh_sc[s * (HEAD_DIM / NVFP4_BLOCK) + blk];

                /* Conservative V dequant: 1 cast + 1 FMUL with scale stub. */
                float vsc_stub = (float)vs_byte * STUB_SCALE_COEF;
                float v = (float)nibble * vsc_stub;

                /* Conservative vfconv.fp8.bf16.v: 1 cast (no LUT). */
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

    printf("fa_mixed_stub bench done (CONSERVATIVE FU-latency stub)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f  (NOT a correctness check)\n", checksum);
    printf("  KV total      = %lu\n",
           (unsigned long)(2 * (K_PACKED_BYTES + K_SCALE_BYTES)));

    free(K_pk); free(K_sc); free(V_pk); free(V_sc);
    free(Q); free(O); free(S); free(P_fp32); free(P_fp8);
    return 0;
}
