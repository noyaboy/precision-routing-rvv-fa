/*
 * Track J-4.5d — Variant of bench_fa_mixed_rvv_native.c that ALSO issues
 * the FP8-quant via the native `vfconv.bf16.fp8.v` lane (VS1=0x08,
 * op-class SimdBf16Fp8Cvt, opLat=2).  This is the "all 4 Saturn customs
 * integrated end-to-end" experiment.
 *
 * Pipeline per 16-element batch in the FP8 quant loop:
 *   vle32 P_fp32 -> vfmul.vf by 448 -> vnsrl FP32->BF16 (low 16 bits
 *   dropped, truncation; BF16 narrowing error << downstream FP8-E4M3
 *   quant resolution) -> vfconv.bf16.fp8.v -> vnsrl u16 -> u8
 *   (extract low byte) -> vse8 P_fp8.
 *
 * Finding (RiscvO3CPU @ SEQ_LEN=2048): correctness preserved (checksum
 * -20.375 vs reference -20.27 -> 0.4 % drift, well inside the 5 % NVFP4
 * quantization-noise budget), but cycles INCREASE from native (SW FP8
 * quant) 7.00 M -> 7.56 M.  IPC drops 2.19 -> 1.91 while instruction
 * count drops 15.36 M -> 14.41 M.  The 5-deep vector dependency chain
 * at LMUL=1 / mf2 limits ILP; gem5's single-instruction-2-cycle stub
 * for the vfconv.bf16.fp8.v lane doesn't model the 16-lane parallelism
 * a real Saturn FU would provide.  Kept as a sidebar variant; the
 * primary bench (bench_fa_mixed_rvv_native.c) uses SW FP8 quant.
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

/* Native vfconv.nvfp4.bf16.v dequant: all 4 blocks of 16 nibbles in one
 * asm block.  Bug-2 fix: collapse 8 vsetvli switches/row (2 per block) to
 * 2 vsetvli switches/row (one e16-m4 pass for the vfconv, one e32-m2
 * config that persists across the 4 widen+scale+store blocks).  Saves
 * ~6 vsetvli + 4 SEW-transition penalties per row x 16 K rows. */
static inline void dequant_row_native(const uint8_t *pk_row,
                                      const uint8_t *sc_row,
                                      float fp32_out[HEAD_DIM]) {
    uint16_t nibbles[HEAD_DIM] __attribute__((aligned(64)));

    const int blocks_per_row = HEAD_DIM / NVFP4_BLOCK;
    for (int blk = 0; blk < blocks_per_row; blk++) {
        for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
            uint8_t byte = pk_row[blk * (NVFP4_BLOCK / 2) + b];
            nibbles[blk * NVFP4_BLOCK + 2 * b + 0] = (uint16_t)(byte & 0xF);
            nibbles[blk * NVFP4_BLOCK + 2 * b + 1] = (uint16_t)((byte >> 4) & 0xF);
        }
    }

    float s0 = e4m3_decode[sc_row[0]];
    float s1 = e4m3_decode[sc_row[1]];
    float s2 = e4m3_decode[sc_row[2]];
    float s3 = e4m3_decode[sc_row[3]];

    asm volatile (
        "li        t0, 64\n\t"
        "vsetvli   x0, t0, e16, m4, ta, ma\n\t"
        "vle16.v   v0, (%[in])\n\t"
        ".4byte    0x4E049057\n\t"          /* vfconv.nvfp4.bf16.v v0, v0 */

        "vsetivli  x0, 16, e32, m2, ta, ma\n\t"

        "vzext.vf2 v8, v0\n\t"
        "vsll.vi   v8, v8, 16\n\t"
        "vfmul.vf  v8, v8, %[s0]\n\t"
        "vse32.v   v8, (%[o0])\n\t"

        "vzext.vf2 v8, v1\n\t"
        "vsll.vi   v8, v8, 16\n\t"
        "vfmul.vf  v8, v8, %[s1]\n\t"
        "vse32.v   v8, (%[o1])\n\t"

        "vzext.vf2 v8, v2\n\t"
        "vsll.vi   v8, v8, 16\n\t"
        "vfmul.vf  v8, v8, %[s2]\n\t"
        "vse32.v   v8, (%[o2])\n\t"

        "vzext.vf2 v8, v3\n\t"
        "vsll.vi   v8, v8, 16\n\t"
        "vfmul.vf  v8, v8, %[s3]\n\t"
        "vse32.v   v8, (%[o3])\n\t"
        :
        : [in]"r"(nibbles),
          [s0]"f"(s0), [s1]"f"(s1), [s2]"f"(s2), [s3]"f"(s3),
          [o0]"r"(fp32_out +  0), [o1]"r"(fp32_out + 16),
          [o2]"r"(fp32_out + 32), [o3]"r"(fp32_out + 48)
        : "memory", "t0",
          "v0", "v1", "v2", "v3", "v8", "v9"
    );
}

int main(void) {
    e4m3_init_table(e4m3_decode);

    uint8_t *K_pk = (uint8_t *)malloc(K_PACKED_BYTES);
    uint8_t *K_sc = (uint8_t *)malloc(K_SCALE_BYTES);
    uint8_t *V_pk = (uint8_t *)malloc(K_PACKED_BYTES);
    uint8_t *V_sc = (uint8_t *)malloc(K_SCALE_BYTES);
    bf16_t  *Q    = (bf16_t  *)malloc(N_HEADS * HEAD_DIM * sizeof(bf16_t));
    bf16_t  *O    = (bf16_t  *)malloc(N_HEADS * HEAD_DIM * sizeof(bf16_t));
    /* Pre-widened Q in FP32.  Avoids needing bf16_load_widen (vle16 +
     * vzext.vf2 + vsll.vi) inside the QK inner loop, which GCC 13.2's
     * RVV vsetvli pass mis-optimizes after the dequant asm block (no
     * e32-m2 vsetvli emitted between the vle16 and the vzext, so vzext
     * runs at e16 m1 reading bytes instead of u16s).  Q is small
     * (8 x 64 x 4B = 2 KiB) so the extra storage is negligible. */
    float   *Q_fp32 = (float *)malloc(N_HEADS * HEAD_DIM * sizeof(float));
    if (!K_pk || !K_sc || !V_pk || !V_sc || !Q || !O || !Q_fp32) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    for (int h = 0; h < N_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++) {
            float qv = init_q_fp32(h, d);
            Q[h * HEAD_DIM + d]      = fp32_to_bf16(qv);
            Q_fp32[h * HEAD_DIM + d] = bf16_to_fp32(Q[h * HEAD_DIM + d]);
        }

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
        const float   *qh_fp32 = &Q_fp32[h * HEAD_DIM];
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
                vfloat32m2_t vq = __riscv_vle32_v_f32m2(qh_fp32 + d, vl);
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

        /* Native FP8 quant via vfconv.bf16.fp8.v (see file header). */
        {
            const float P_SCALE = 448.0f;
            int n = SEQ_LEN;
            const float *Pp = P_fp32;
            uint8_t *Fp = P_fp8;
            asm volatile (
                "1:\n\t"
                "vsetivli  x0, 16, e32, m2, ta, ma\n\t"
                "vle32.v   v0, (%[Pp])\n\t"
                "vfmul.vf  v0, v0, %[scale]\n\t"
                "vsetivli  x0, 16, e16, m1, ta, ma\n\t"
                "vnsrl.wi  v4, v0, 16\n\t"
                ".4byte    0x4E441257\n\t"   /* vfconv.bf16.fp8.v v4, v4 */
                "vsetivli  x0, 16, e8, mf2, ta, ma\n\t"
                "vnsrl.wi  v6, v4, 0\n\t"
                "vse8.v    v6, (%[Fp])\n\t"
                "addi      %[Pp], %[Pp], 64\n\t"
                "addi      %[Fp], %[Fp], 16\n\t"
                "addi      %[n], %[n], -16\n\t"
                "bnez      %[n], 1b\n\t"
                : [Pp]"+r"(Pp), [Fp]"+r"(Fp), [n]"+r"(n)
                : [scale]"f"(P_SCALE)
                : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6"
            );
        }

        for (int d = 0; d < HEAD_DIM; d++) O_fp32[d] = 0.0f;

        for (int s = 0; s < SEQ_LEN; s++) {
            dequant_row_native(&vh_pk[s * (HEAD_DIM / 2)],
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
    free(Q); free(Q_fp32); free(O); free(S); free(P_fp32); free(P_fp8);
    (void)bf16_load_widen;
    (void)bf16_load_widen_m4;
    return 0;
}
