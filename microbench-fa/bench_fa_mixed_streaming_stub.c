/*
 * Track J-2b — SpacemiT's online-softmax algorithm + mixed-precision K/V
 * (NVFP4) + aggressive FU stub for dequant + inline polynomial expf.
 *
 * This is the "Saturn version" of SpacemiT's streaming FA: same algorithm
 * structure (single pass over ic with online softmax rescaling, Q cached,
 * VKQ in vector regs), but K and V are NVFP4-quantized per-block-16 with
 * E4M3 scales — the J3 mixed-prec format.
 *
 * Note: there is no FP8 attn-weight quant in the streaming algorithm
 * because P[s] is never stored; each `vs` is a scalar that multiplies
 * into VKQ.  So `vfconv.bf16.fp8.v` and `vfconv.fp8.bf16.v` are absent
 * from this kernel.  Only `vfconv.nvfp4.bf16.v` (K and V dequant) and
 * `vfexp.v` (the scalar `vs = expf(s - M)`) are the FU lanes that apply.
 *
 * Dequant is FU-stubbed aggressively (Track J2 stub2 pattern): byte load
 * + 1 op per element, no LUT, no scalar FMUL with scale; scale byte is
 * loaded with asm-volatile clobber to preserve BW.  expf is the J2
 * inline polynomial (~12 ops, models vfexp.v's 10-cycle pipeline).
 *
 * Direct Mo 6 head-to-head with bench_fa_spacemit_bf16.c (BF16 K/V baseline,
 * same algorithm) and bench_fa_mixed_rvv_stub.c (mixed K/V, two-pass J3
 * algorithm).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>
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
    /* Underflow / overflow guards (avoid polynomial UB at large |x|).
     * Saturn's vfexp.v has the same special-case handling per Track E. */
    if (x < -87.0f) return 0.0f;
    if (x >  88.0f) return INFINITY;
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

static inline vfloat32m4_t bf16_load_widen_m4(const bf16_t *p, size_t vl) {
    vuint16m2_t bf  = __riscv_vle16_v_u16m2(p, vl);
    vuint32m4_t z   = __riscv_vzext_vf2_u32m4(bf, vl);
    vuint32m4_t shf = __riscv_vsll_vx_u32m4(z, 16, vl);
    return __riscv_vreinterpret_v_u32m4_f32m4(shf);
}

/* Aggressive FU stub for row dequant: byte load + 1 cast per element,
 * scale byte load preserved via asm clobber.  Models vfconv.nvfp4.bf16.v
 * output as FP32 directly into a stack-aligned scratch buffer that the
 * vector pipe then loads. */
static inline void dequant_row_stub(const uint8_t *pk_row,
                                    const uint8_t *sc_row,
                                    float fp32_out[HEAD_DIM]) {
    const int blocks_per_row = HEAD_DIM / NVFP4_BLOCK;
    for (int blk = 0; blk < blocks_per_row; blk++) {
        uint8_t sc_byte = sc_row[blk];
        asm volatile ("" :: "r"((uint32_t)sc_byte) : );
        for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
            int d = blk * NVFP4_BLOCK + 2 * b;
            uint8_t byte = pk_row[blk * (NVFP4_BLOCK / 2) + b];
            fp32_out[d]     = (float)(byte & 0x0f);
            fp32_out[d + 1] = (float)((byte >> 4) & 0x0f);
        }
    }
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

    float K_fp32_row[HEAD_DIM] __attribute__((aligned(64)));
    float V_fp32_row[HEAD_DIM] __attribute__((aligned(64)));
    float VKQ_scratch[HEAD_DIM] __attribute__((aligned(64)));
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    /* ===================== MEASUREMENT REGION ===================== */
    m5_dump_reset_stats(0, 0);
    uint64_t cyc_start = rdcycle();

    size_t vl_m4 = __riscv_vsetvl_e32m4(32);

    for (int h = 0; h < N_HEADS; h++) {
        const bf16_t  *qh = &Q[h * HEAD_DIM];
        const uint8_t *kh_pk = &K_pk[(size_t)h * SEQ_LEN * (HEAD_DIM / 2)];
        const uint8_t *kh_sc = &K_sc[(size_t)h * SEQ_LEN * (HEAD_DIM / NVFP4_BLOCK)];
        const uint8_t *vh_pk = &V_pk[(size_t)h * SEQ_LEN * (HEAD_DIM / 2)];
        const uint8_t *vh_sc = &V_sc[(size_t)h * SEQ_LEN * (HEAD_DIM / NVFP4_BLOCK)];
        bf16_t *oh = &O[h * HEAD_DIM];

        vfloat32m4_t Q_a = bf16_load_widen_m4(qh + 0,     vl_m4);
        vfloat32m4_t Q_b = bf16_load_widen_m4(qh + vl_m4, HEAD_DIM - vl_m4);

        vfloat32m4_t VKQ_a = __riscv_vfmv_v_f_f32m4(0.0f, vl_m4);
        vfloat32m4_t VKQ_b = __riscv_vfmv_v_f_f32m4(0.0f, HEAD_DIM - vl_m4);

        float S = 0.0f;
        float M = -1e30f;

        for (int ic = 0; ic < SEQ_LEN; ic++) {
            /* FU-stubbed K dequant -> K_fp32_row. */
            dequant_row_stub(&kh_pk[ic * (HEAD_DIM / 2)],
                             &kh_sc[ic * (HEAD_DIM / NVFP4_BLOCK)],
                             K_fp32_row);

            vfloat32m4_t K_a = __riscv_vle32_v_f32m4(K_fp32_row + 0,     vl_m4);
            vfloat32m4_t K_b = __riscv_vle32_v_f32m4(K_fp32_row + vl_m4, HEAD_DIM - vl_m4);

            vfloat32m4_t qk_a = __riscv_vfmul_vv_f32m4(K_a, Q_a, vl_m4);
            vfloat32m4_t qk_b = __riscv_vfmul_vv_f32m4(K_b, Q_b, HEAD_DIM - vl_m4);

            vfloat32m1_t s_red = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            s_red = __riscv_vfredusum_vs_f32m4_f32m1(qk_a, s_red, vl_m4);
            s_red = __riscv_vfredusum_vs_f32m4_f32m1(qk_b, s_red, HEAD_DIM - vl_m4);
            float s = __riscv_vfmv_f_s_f32m1_f32(s_red);

            s = s * scale;

            const float Mold = M;
            float ms = 1.0f;
            float vs = 1.0f;

            /* FU-stubbed V dequant -> V_fp32_row. */
            dequant_row_stub(&vh_pk[ic * (HEAD_DIM / 2)],
                             &vh_sc[ic * (HEAD_DIM / NVFP4_BLOCK)],
                             V_fp32_row);

            vfloat32m4_t V_a = __riscv_vle32_v_f32m4(V_fp32_row + 0,     vl_m4);
            vfloat32m4_t V_b = __riscv_vle32_v_f32m4(V_fp32_row + vl_m4, HEAD_DIM - vl_m4);

            if (s > M) {
                M  = s;
                ms = fu_expf(Mold - M);
                VKQ_a = __riscv_vfmul_vf_f32m4(VKQ_a, ms, vl_m4);
                VKQ_b = __riscv_vfmul_vf_f32m4(VKQ_b, ms, HEAD_DIM - vl_m4);
            } else {
                vs = fu_expf(s - M);
            }

            VKQ_a = __riscv_vfmacc_vf_f32m4(VKQ_a, vs, V_a, vl_m4);
            VKQ_b = __riscv_vfmacc_vf_f32m4(VKQ_b, vs, V_b, HEAD_DIM - vl_m4);

            S = S * ms + vs;
        }

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

    printf("fa_mixed_streaming_stub bench done (SpacemiT-algo + NVFP4 + FU stub)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f  (NOT a correctness check)\n", checksum);
    printf("  KV total      = %lu\n",
           (unsigned long)(2 * (K_PACKED_BYTES + K_SCALE_BYTES)));

    free(K_pk); free(K_sc); free(V_pk); free(V_sc);
    free(Q); free(O);
    return 0;
}
