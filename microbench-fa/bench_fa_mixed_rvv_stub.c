/*
 * Track J3+J2 — Combined: RVV-vectorized mixed-prec FA + aggressive FU stub.
 *
 * Most-Saturn-faithful projection on gem5 TimingSimpleCPU:
 *   - RVV intrinsics for vectorized arithmetic (Q*K dot, softmax max/sum,
 *     P*V FMA), matching what Saturn's vector pipe would issue.
 *   - Aggressive FU stubs for the 4 vfconv lanes + vfexp (Track J2 stub2
 *     pattern + Track H feedback-fu-stub-brackets): byte load + 1 cast per
 *     element for the dequant lanes, asm-clobber to preserve scale-byte
 *     loads, inline polynomial expf for vfexp.v.
 *
 * Row-major P*V (s-outer, d-inner) as in bench_fa_mixed_rvv.c so the
 * vector pipe can saturate. Memory pattern preserved from J3.
 *
 * Cycle output projects the Mo 6 Saturn hand-coded number when the FU is
 * physically present in the vector pipe.  The pair (this + a future
 * conservative stub variant) brackets the answer.
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

static inline vfloat32m2_t bf16_load_widen(const bf16_t *p, size_t vl) {
    vuint16m1_t bf  = __riscv_vle16_v_u16m1(p, vl);
    vuint32m2_t z   = __riscv_vzext_vf2_u32m2(bf, vl);
    vuint32m2_t shf = __riscv_vsll_vx_u32m2(z, 16, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(shf);
}

/* Aggressive FU stub for row dequant: byte load + 1 cast per element,
 * scale byte load preserved via asm clobber.  No FMUL — the FU
 * (vfconv.nvfp4.bf16.v) is assumed to output the fully-scaled value. */
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

    float *S      = (float *)malloc(SEQ_LEN * sizeof(float));
    float *P_fp32 = (float *)malloc(SEQ_LEN * sizeof(float));
    uint8_t *P_fp8 = (uint8_t *)malloc(SEQ_LEN);
    float K_fp32_row[HEAD_DIM] __attribute__((aligned(64)));
    float V_fp32_row[HEAD_DIM] __attribute__((aligned(64)));
    float O_fp32[HEAD_DIM]     __attribute__((aligned(64)));
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

        for (int s = 0; s < SEQ_LEN; s++) {
            dequant_row_stub(&kh_pk[s * (HEAD_DIM / 2)],
                             &kh_sc[s * (HEAD_DIM / NVFP4_BLOCK)],
                             K_fp32_row);

            size_t vl_max = __riscv_vsetvl_e32m2(16);
            vfloat32m2_t vacc = __riscv_vfmv_v_f_f32m2(0.0f, vl_max);
            for (int d = 0; d < HEAD_DIM; d += (int)vl_max) {
                size_t vl = __riscv_vsetvl_e32m2(HEAD_DIM - d);
                vfloat32m2_t vq = bf16_load_widen(qh + d, vl);
                vfloat32m2_t vk = __riscv_vle32_v_f32m2(K_fp32_row + d, vl);
                vacc = __riscv_vfmacc_vv_f32m2(vacc, vq, vk, vl);
            }
            vfloat32m1_t vred_init = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t vred = __riscv_vfredusum_vs_f32m2_f32m1(vacc, vred_init, vl_max);
            float acc = __riscv_vfmv_f_s_f32m1_f32(vred);
            S[s] = acc * scale;
        }

        size_t vlm = __riscv_vsetvl_e32m2(16);
        vfloat32m1_t vmax_init = __riscv_vfmv_v_f_f32m1(-1e30f, 1);
        vfloat32m1_t vmax_red = vmax_init;
        for (int s = 0; s < SEQ_LEN; s += (int)vlm) {
            size_t vl = __riscv_vsetvl_e32m2(SEQ_LEN - s);
            vfloat32m2_t vs = __riscv_vle32_v_f32m2(S + s, vl);
            vmax_red = __riscv_vfredmax_vs_f32m2_f32m1(vs, vmax_red, vl);
        }
        float max_s = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

        /* vfexp stub: inline polynomial expf. */
        float sum_p = 0.0f;
        for (int s = 0; s < SEQ_LEN; s++) {
            float e = fu_expf(S[s] - max_s);
            P_fp32[s] = e;
            sum_p += e;
        }
        float inv_sum = 1.0f / sum_p;
        const float row_dequant_scale = inv_sum / 448.0f;

        /* FP8 quant stub: aggressive 1-op trunc. */
        for (int s = 0; s < SEQ_LEN; s++) {
            float pf = P_fp32[s];
            asm volatile ("" :: "f"(pf) : );
            P_fp8[s] = (uint8_t)((int)pf & 0xff);
        }

        for (int d = 0; d < HEAD_DIM; d++) O_fp32[d] = 0.0f;

        for (int s = 0; s < SEQ_LEN; s++) {
            dequant_row_stub(&vh_pk[s * (HEAD_DIM / 2)],
                             &vh_sc[s * (HEAD_DIM / NVFP4_BLOCK)],
                             V_fp32_row);
            float p = (float)P_fp8[s];

            for (int d = 0; d < HEAD_DIM; d += (int)vlm) {
                size_t vl = __riscv_vsetvl_e32m2(HEAD_DIM - d);
                vfloat32m2_t vo = __riscv_vle32_v_f32m2(O_fp32 + d, vl);
                vfloat32m2_t vv = __riscv_vle32_v_f32m2(V_fp32_row + d, vl);
                vo = __riscv_vfmacc_vf_f32m2(vo, p, vv, vl);
                __riscv_vse32_v_f32m2(O_fp32 + d, vo, vl);
            }
        }

        for (int d = 0; d < HEAD_DIM; d++)
            oh[d] = fp32_to_bf16(O_fp32[d] * row_dequant_scale);
    }

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * HEAD_DIM; i++)
        checksum += bf16_to_fp32(O[i]);

    printf("fa_mixed_rvv_stub bench done (RVV + aggressive FU stub)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f  (NOT a correctness check)\n", checksum);
    printf("  KV total      = %lu\n",
           (unsigned long)(2 * (K_PACKED_BYTES + K_SCALE_BYTES)));

    free(K_pk); free(K_sc); free(V_pk); free(V_sc);
    free(Q); free(O); free(S); free(P_fp32); free(P_fp8);
    return 0;
}
