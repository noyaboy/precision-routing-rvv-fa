/*
 * Track J-4.5b — Flash-attention decode-step, mixed precision, RVV-vectorized,
 * with native Saturn custom FUs (vfexp.v + vfconv.nvfp4.bf16.v).
 *
 * Built on bench_fa_mixed_rvv_stub.c.  Differences:
 *   1. `dequant_row_native()` replaces `dequant_row_stub()`.  Scalar unpacks
 *      8 packed bytes into 16 u16-nibbles per NVFP4 block, then issues the
 *      native `vfconv.nvfp4.bf16.v` (VFUNCT6=0x13/VS1=0x09, op-class
 *      SimdNvfp4Cvt with opLat=3 wired in FuncUnitConfig.py).  Per-block
 *      E4M3 scale applied via vector vfmul on the widened FP32 values.
 *   2. The softmax exp loop swaps scalar `fu_expf()` for native vector
 *      `vfexp.v` (VFUNCT6=0x13/VS1=0x06, op-class SimdFloatExp, opLat=10).
 *      Vectorized over SEQ_LEN at LMUL=2.
 *
 * Bench shape and all other kernel structure preserved from
 * bench_fa_mixed_rvv_stub.c.  Direct cycle comparison: J3+stub (scalar
 * dequant + polynomial expf) vs J4 native (vfconv + vfexp).
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

static inline vfloat32m2_t bf16_load_widen(const bf16_t *p, size_t vl) {
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

/* Native vfconv.nvfp4.bf16.v dequant: 4 blocks of 16 nibbles per row.
 * Self-contained asm block to avoid vsetvli-state mismatch between asm
 * and surrounding intrinsics.  Scalar unpack to u16-nibble buffer, then
 * single asm: vle16 -> vfconv.nvfp4.bf16.v -> vzext+vsll widen to FP32 ->
 * vfmul with E4M3 scale -> vse32 to output. */
static inline void dequant_row_native(const uint8_t *pk_row,
                                      const uint8_t *sc_row,
                                      float fp32_out[HEAD_DIM]) {
    uint16_t nibbles[NVFP4_BLOCK] __attribute__((aligned(64)));

    const int blocks_per_row = HEAD_DIM / NVFP4_BLOCK;
    for (int blk = 0; blk < blocks_per_row; blk++) {
        for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
            uint8_t byte = pk_row[blk * (NVFP4_BLOCK / 2) + b];
            nibbles[2 * b + 0] = (uint16_t)(byte & 0xF);
            nibbles[2 * b + 1] = (uint16_t)((byte >> 4) & 0xF);
        }
        float scale = e4m3_decode[sc_row[blk]];
        asm volatile (
            "vsetivli x0, 16, e16, m1, ta, ma\n\t"
            "vle16.v   v0, (%[in])\n\t"
            ".4byte    0x4E049057\n\t"   /* vfconv.nvfp4.bf16.v v0, v0 */
            "vsetivli  x0, 16, e32, m2, ta, ma\n\t"
            "vzext.vf2 v8, v0\n\t"       /* u16 -> u32, LMUL 1->2 */
            "vsll.vi   v8, v8, 16\n\t"   /* shift bf16 bits to FP32 */
            "vfmul.vf  v8, v8, %[scale]\n\t"
            "vse32.v   v8, (%[out])\n\t"
            :
            : [in]"r"(nibbles), [scale]"f"(scale),
              [out]"r"(fp32_out + blk * NVFP4_BLOCK)
            : "memory", "v0", "v8", "v9"
        );
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
            dequant_row_native(&kh_pk[s * (HEAD_DIM / 2)],
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

        /* Vector softmax exp via native vfexp.v.  All in asm so we can
         * issue the .4byte custom encoding alongside RVV ops on explicit
         * registers.  Reduce-sum into a scratch fp32 (sum_p). */
        float sum_p = 0.0f;
        {
            float local_sum = 0.0f;
            int n = SEQ_LEN;
            const float *Sp = S;
            float *Pp = P_fp32;
            asm volatile (
                "vsetvli   t0, %[n], e32, m2, ta, ma\n\t"
                "vmv.v.i   v0, 0\n\t"        /* reduction accumulator (m1) */
                "vmv.v.i   v8, 0\n\t"        /* clear work reg */
                "1:\n\t"
                "vsetvli   t0, %[n], e32, m2, ta, ma\n\t"
                "vle32.v   v8, (%[s])\n\t"
                "vfsub.vf  v8, v8, %[mx]\n\t"
                ".4byte    0x4E831457\n\t"   /* vfexp.v v8, v8 */
                "vse32.v   v8, (%[p])\n\t"
                "vfredusum.vs v0, v8, v0\n\t"
                "slli      t1, t0, 2\n\t"     /* bytes consumed */
                "add       %[s], %[s], t1\n\t"
                "add       %[p], %[p], t1\n\t"
                "sub       %[n], %[n], t0\n\t"
                "bnez      %[n], 1b\n\t"
                "vsetivli  x0, 1, e32, m1, ta, ma\n\t"
                "vfmv.f.s  %[sum], v0\n\t"
                : [sum]"=f"(local_sum), [s]"+r"(Sp), [p]"+r"(Pp), [n]"+r"(n)
                : [mx]"f"(max_s)
                : "t0", "t1", "memory", "v0", "v8", "v9"
            );
            sum_p = local_sum;
        }
        float inv_sum = 1.0f / sum_p;
        const float row_dequant_scale = inv_sum / 448.0f;

        /* FP8 quant stub (unchanged from J3+stub). */
        for (int s = 0; s < SEQ_LEN; s++) {
            float pf = P_fp32[s];
            asm volatile ("" :: "f"(pf) : );
            P_fp8[s] = (uint8_t)((int)pf & 0xff);
        }

        for (int d = 0; d < HEAD_DIM; d++) O_fp32[d] = 0.0f;

        for (int s = 0; s < SEQ_LEN; s++) {
            dequant_row_native(&vh_pk[s * (HEAD_DIM / 2)],
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

    printf("fa_mixed_rvv_native bench done (RVV + native Saturn customs)\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f\n", checksum);
    printf("  KV total      = %lu\n",
           (unsigned long)(2 * (K_PACKED_BYTES + K_SCALE_BYTES)));

    free(K_pk); free(K_sc); free(V_pk); free(V_sc);
    free(Q); free(O); free(S); free(P_fp32); free(P_fp8);
    (void)bf16_load_widen_m4;
    return 0;
}
