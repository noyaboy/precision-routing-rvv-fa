/*
 * Track J3 — Flash-attention decode-step, mixed precision, RVV-vectorized.
 *
 * Builds on J1 (bench_fa_mixed.c): same unified-FP32 init, same per-row FP8
 * scale, same per-block-16 NVFP4 K/V quant.  Differences:
 *
 * 1. P*V loop restructured to s-outer/d-inner (the only structure that
 *    vectorizes the d=64 reduction over output channels).  V re-reads
 *    collapse from HEAD_DIM=64x to 1x.  This is also the access pattern a
 *    BLAS-tiled FA implementation would use.
 *
 * 2. Hot loops vectorized with RVV intrinsics at VLEN=256, SEW=32, LMUL=2
 *    (16 FP32 elts per vector reg pair):
 *      - QK^T  : vfwmacc-style FMA with BF16->FP32 widen on Q.
 *      - softmax max  : vfredmax over S[SEQ_LEN].
 *      - softmax sum  : vfredusum over P_fp32[SEQ_LEN].
 *      - P*V   : vector-scalar FMA accumulating O_fp32[HEAD_DIM] across s.
 *
 *    BF16 widen avoids Zvfbfwma/Zvfh (gem5 has neither): vle16 -> vzext.vf2
 *    -> vsll 16 -> reinterpret as f32m2.
 *
 * 3. K/V/FP8 dequant + FP8 quant stay scalar.  These model the Saturn
 *    custom instructions (vfconv.{nvfp4,bf16,fp8}.*); on real Saturn they
 *    run inside the vector pipe at 16-lane width.  Combine with the FU
 *    stubs from Track J2 to get the full projection.
 *
 * Numerical validation is checksum-vs-J1.
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

/* BF16 (u16) -> FP32 (f32) via bit shift.  RVV equivalent of
 * (uint32_t)bf16_bits << 16 reinterpreted as float.  Uses vzext.vf2 to widen
 * LMUL=1 -> LMUL=2 alongside the SEW widen. */
static inline vfloat32m2_t bf16_load_widen(const bf16_t *p, size_t vl) {
    vuint16m1_t bf  = __riscv_vle16_v_u16m1(p, vl);
    vuint32m2_t z   = __riscv_vzext_vf2_u32m2(bf, vl);
    vuint32m2_t shf = __riscv_vsll_vx_u32m2(z, 16, vl);
    return __riscv_vreinterpret_v_u32m2_f32m2(shf);
}

/* Dequant K[s,:] or V[s,:] (HEAD_DIM=64) into a FP32 scratch buffer using
 * the SW LUT path.  Each call models 4 invocations of vfconv.nvfp4.bf16.v
 * on Saturn (one per NVFP4 block of 16 lanes); on gem5 we run scalar code. */
static inline void dequant_row(const uint8_t *pk_row,
                               const uint8_t *sc_row,
                               float fp32_out[HEAD_DIM]) {
    const int blocks_per_row = HEAD_DIM / NVFP4_BLOCK;
    for (int blk = 0; blk < blocks_per_row; blk++) {
        float bsc = e4m3_decode[sc_row[blk]];
        for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
            int d = blk * NVFP4_BLOCK + 2 * b;
            uint8_t byte = pk_row[blk * (NVFP4_BLOCK / 2) + b];
            fp32_out[d]     = nvfp4_decode_table[byte & 0xf]        * bsc;
            fp32_out[d + 1] = nvfp4_decode_table[(byte >> 4) & 0xf] * bsc;
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

        /* QK^T: scalar dequant K[s,:] into K_fp32_row, then vectorize the
         * dot product with Q (HEAD_DIM=64 -> 4 iters at vl=16). */
        for (int s = 0; s < SEQ_LEN; s++) {
            dequant_row(&kh_pk[s * (HEAD_DIM / 2)],
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

        /* Softmax max over S[SEQ_LEN] via vfredmax. */
        size_t vlm = __riscv_vsetvl_e32m2(16);
        vfloat32m1_t vmax_init = __riscv_vfmv_v_f_f32m1(-1e30f, 1);
        vfloat32m1_t vmax_red = vmax_init;
        for (int s = 0; s < SEQ_LEN; s += (int)vlm) {
            size_t vl = __riscv_vsetvl_e32m2(SEQ_LEN - s);
            vfloat32m2_t vs = __riscv_vle32_v_f32m2(S + s, vl);
            vmax_red = __riscv_vfredmax_vs_f32m2_f32m1(vs, vmax_red, vl);
        }
        float max_s = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

        /* Softmax exp + sum.  expf is scalar (Saturn would use vfexp.v
         * vector; on gem5 libm is the only available scalar approximation). */
        float sum_p = 0.0f;
        for (int s = 0; s < SEQ_LEN; s++) {
            float e = expf(S[s] - max_s);
            P_fp32[s] = e;
            sum_p += e;
        }
        float inv_sum = 1.0f / sum_p;
        const float row_dequant_scale = inv_sum / FP8_E4M3_MAX;

        /* FP8 quant of attn weights (scalar, models vfconv.bf16.fp8.v). */
        for (int s = 0; s < SEQ_LEN; s++) {
            bf16_t p_bf16 = fp32_to_bf16(P_fp32[s] * FP8_E4M3_MAX);
            P_fp8[s] = bf16_to_e4m3(p_bf16);
        }

        /* P_fp8 * V_nvfp4 restructured s-outer/d-inner.  For each s:
         *   - dequant V[s,:] -> V_fp32_row (scalar LUT)
         *   - p = e4m3_decode[P_fp8[s]]  (scalar)
         *   - vector FMA: O_fp32[d] += p * V_fp32_row[d] across d=0..HEAD_DIM
         * Row-major access of V_pk means one read per (s, d) — no stride
         * penalty.  Vector lanes load 16 V values at once. */
        for (int d = 0; d < HEAD_DIM; d++) O_fp32[d] = 0.0f;

        for (int s = 0; s < SEQ_LEN; s++) {
            dequant_row(&vh_pk[s * (HEAD_DIM / 2)],
                        &vh_sc[s * (HEAD_DIM / NVFP4_BLOCK)],
                        V_fp32_row);
            float p = e4m3_decode[P_fp8[s]];

            for (int d = 0; d < HEAD_DIM; d += (int)vlm) {
                size_t vl = __riscv_vsetvl_e32m2(HEAD_DIM - d);
                vfloat32m2_t vo = __riscv_vle32_v_f32m2(O_fp32 + d, vl);
                vfloat32m2_t vv = __riscv_vle32_v_f32m2(V_fp32_row + d, vl);
                vo = __riscv_vfmacc_vf_f32m2(vo, p, vv, vl);
                __riscv_vse32_v_f32m2(O_fp32 + d, vo, vl);
            }
        }

        /* Final cast O_fp32 * row_scale -> BF16. */
        for (int d = 0; d < HEAD_DIM; d++)
            oh[d] = fp32_to_bf16(O_fp32[d] * row_dequant_scale);
    }

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * HEAD_DIM; i++)
        checksum += bf16_to_fp32(O[i]);

    printf("fa_mixed_rvv bench done (RVV-vectorized, row-major P*V)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f\n", checksum);
    printf("  KV total      = %lu\n",
           (unsigned long)(2 * (K_PACKED_BYTES + K_SCALE_BYTES)));

    free(K_pk); free(K_sc); free(V_pk); free(V_sc);
    free(Q); free(O); free(S); free(P_fp32); free(P_fp8);
    return 0;
}
