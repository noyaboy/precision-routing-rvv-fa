/*
 * Mo 8 step 4c — bench harness for the Exo-generated FA kernel.
 *
 * Calls fa_kernel_decode_naive() (paper/exo_schedule_fa.py composition)
 * with the same FP32 source data as bench_fa_mixed_rvv_native.c. Two
 * deferred-from-step-4b-3 phases handled in C scaffolding here:
 *
 *   1. FP8 quant of P — SKIPPED. fa_kernel uses P_fp32[s] directly as the
 *      per-key scalar in P·V instead of the bench's
 *      e4m3_decode[bf16_to_e4m3(fp32_to_bf16(P[s] * 448))] round-trip.
 *      Cycle measurement of the kernel excludes this FP8-quant phase
 *      (which the bench measures inside its main loop).
 *   2. Output BF16 conversion + row_dequant_scale — SKIPPED. The
 *      kernel returns O_fp32 (unscaled). The bench multiplies by
 *      inv_sum/448 and narrows to BF16 (lines 281-282).
 *
 * The cycle delta vs bench_fa_mixed_rvv_native_g14_l2k thus measures
 * just the cycle-dominating phases (dequant + QK^T + softmax + P·V),
 * which are bit-for-bit structurally identical to the bench's hot loop
 * modulo the asm-volatile macro spill/reload overhead flagged in
 * step 4a's intrinsic-vs-macro cost analysis.
 *
 * Also note: this bench pre-unpacks the NVFP4 nibbles outside the
 * measured region (the bench unpacks inside the QK and PV loops via
 * dequant_64elt_chunk). Step 4d will add an inline-unpack variant for
 * apples-to-apples comparison; for step 4c first measurement, expect
 * our kernel to slightly underestimate cycles by the nibble-unpack cost.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bench_fa_common.h"
#include "m5ops.h"
#include "exo_schedule_fa.h"

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

int main(void) {
    e4m3_init_table(e4m3_decode);

    /* ----- Allocate the bench's packed-NVFP4 source buffers ----- */
    const int blocks_per_row = HEAD_DIM / NVFP4_BLOCK;       /* 4 for head_dim=64 */
    const int bytes_per_row  = HEAD_DIM / 2;                 /* 32 for head_dim=64 */

    uint8_t *K_pk_src = (uint8_t *)malloc(K_PACKED_BYTES);
    uint8_t *K_sc_src = (uint8_t *)malloc(K_SCALE_BYTES);
    uint8_t *V_pk_src = (uint8_t *)malloc(K_PACKED_BYTES);
    uint8_t *V_sc_src = (uint8_t *)malloc(K_SCALE_BYTES);
    if (!K_pk_src || !K_sc_src || !V_pk_src || !V_sc_src) {
        fprintf(stderr, "alloc failed (packed buffers)\n"); return 1;
    }

    /* ----- Allocate the Exo kernel's input buffers ----- */
    /* K_nvfp4 layout: ui16[N_HEADS][SEQ_LEN][4 blocks][16 ui16 nibbles] */
    const size_t k_nvfp4_words = (size_t)N_HEADS * SEQ_LEN * blocks_per_row * NVFP4_BLOCK;
    const size_t k_scale_words = (size_t)N_HEADS * SEQ_LEN * blocks_per_row;
    uint16_t *K_nvfp4  = (uint16_t *)malloc(k_nvfp4_words * sizeof(uint16_t));
    float    *K_scale  = (float    *)malloc(k_scale_words * sizeof(float));
    uint16_t *V_nvfp4  = (uint16_t *)malloc(k_nvfp4_words * sizeof(uint16_t));
    float    *V_scale  = (float    *)malloc(k_scale_words * sizeof(float));
    float    *Q_fp32   = (float    *)malloc(N_HEADS * HEAD_DIM * sizeof(float));
    float    *O_fp32   = (float    *)malloc(N_HEADS * HEAD_DIM * sizeof(float));
    float     qk_scale[1];
    qk_scale[0] = 1.0f / sqrtf((float)HEAD_DIM);
    if (!K_nvfp4 || !K_scale || !V_nvfp4 || !V_scale || !Q_fp32 || !O_fp32) {
        fprintf(stderr, "alloc failed (Exo kernel buffers)\n"); return 1;
    }
    memset(O_fp32, 0, N_HEADS * HEAD_DIM * sizeof(float));

    /* ----- Init Q: BF16 round-trip from FP32 (matches bench) ----- */
    for (int h = 0; h < N_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float qv = init_q_fp32(h, d);
            Q_fp32[h * HEAD_DIM + d] = bf16_to_fp32(fp32_to_bf16(qv));
        }
    }

    /* ----- Init K, V packed buffers via bench's quantize path ----- */
    for (int h = 0; h < N_HEADS; h++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            uint8_t *kh_pk = &K_pk_src[((size_t)h * SEQ_LEN + s) * bytes_per_row];
            uint8_t *kh_sc = &K_sc_src[((size_t)h * SEQ_LEN + s) * blocks_per_row];
            uint8_t *vh_pk = &V_pk_src[((size_t)h * SEQ_LEN + s) * bytes_per_row];
            uint8_t *vh_sc = &V_sc_src[((size_t)h * SEQ_LEN + s) * blocks_per_row];
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

    /* ----- Pre-unpack NVFP4 nibbles into the Exo kernel's layout ----- */
    /* Step 4d will inline this into dequant_row for apples-to-apples
     * with the bench's dequant_64elt_chunk. */
    for (int h = 0; h < N_HEADS; h++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            const uint8_t *kh_pk = &K_pk_src[((size_t)h * SEQ_LEN + s) * bytes_per_row];
            const uint8_t *kh_sc = &K_sc_src[((size_t)h * SEQ_LEN + s) * blocks_per_row];
            const uint8_t *vh_pk = &V_pk_src[((size_t)h * SEQ_LEN + s) * bytes_per_row];
            const uint8_t *vh_sc = &V_sc_src[((size_t)h * SEQ_LEN + s) * blocks_per_row];
            size_t k_base = ((size_t)h * SEQ_LEN + s) * blocks_per_row * NVFP4_BLOCK;
            size_t sc_base = ((size_t)h * SEQ_LEN + s) * blocks_per_row;
            for (int blk = 0; blk < blocks_per_row; blk++) {
                /* Decode block's E4M3 scale to FP32 */
                K_scale[sc_base + blk] = e4m3_decode[kh_sc[blk]];
                V_scale[sc_base + blk] = e4m3_decode[vh_sc[blk]];
                /* Unpack 8 bytes -> 16 ui16 nibbles (low nibble first) */
                for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
                    uint8_t kbyte = kh_pk[blk * (NVFP4_BLOCK / 2) + b];
                    uint8_t vbyte = vh_pk[blk * (NVFP4_BLOCK / 2) + b];
                    size_t dst_idx = k_base + blk * NVFP4_BLOCK + 2 * b;
                    K_nvfp4[dst_idx + 0] = (uint16_t)(kbyte & 0xF);
                    K_nvfp4[dst_idx + 1] = (uint16_t)((kbyte >> 4) & 0xF);
                    V_nvfp4[dst_idx + 0] = (uint16_t)(vbyte & 0xF);
                    V_nvfp4[dst_idx + 1] = (uint16_t)((vbyte >> 4) & 0xF);
                }
            }
        }
    }

    /* Packed source buffers no longer needed once unpacked. */
    free(K_pk_src); free(K_sc_src); free(V_pk_src); free(V_sc_src);

    /* ===================== MEASUREMENT REGION ===================== */
    m5_dump_reset_stats(0, 0);
    uint64_t cyc_start = rdcycle();

    fa_kernel_decode_naive(
        NULL,                    /* ctxt */
        SEQ_LEN,
        qk_scale,
        Q_fp32,
        K_nvfp4, K_scale,
        V_nvfp4, V_scale,
        O_fp32);

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    /* ----- FP32 checksum (no row_dequant_scale; bench checksums BF16) ----- */
    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * HEAD_DIM; i++) checksum += O_fp32[i];

    printf("fa_kernel_decode_naive (Mo 8 step 4c, Exo-generated)\n");
    printf("  seq_len       = %d\n", SEQ_LEN);
    printf("  head_dim      = %d\n", HEAD_DIM);
    printf("  n_heads       = %d\n", N_HEADS);
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f  (FP32, no row_dequant_scale)\n", checksum);
    printf("  notes         = nibble-unpack done outside measurement region;\n");
    printf("                  FP8 quant + output BF16 narrow deferred to step 4d.\n");

    free(K_nvfp4); free(K_scale); free(V_nvfp4); free(V_scale);
    free(Q_fp32);  free(O_fp32);
    return 0;
}
