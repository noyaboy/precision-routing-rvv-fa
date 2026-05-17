/*
 * Mo 2 microbench — NVFP4 K-cache dequant + GEMV.
 *
 * Same kernel shape as bench_fp16.c, but K is NVFP4-quantized:
 *   - 4-bit FP4 E2M1 value per element (packed two per byte)
 *   - 8-bit E4M3 scale per block-16
 *   - (We omit the per-tensor FP32 second-level scale; constant 1.0 in this
 *     microbench. It would be loaded once per pass — negligible BW.)
 *
 * NVFP4 spec source: NVIDIA Blackwell whitepaper + ARCQuant (arXiv:2601.07475).
 * E4M3 follows OCP FP8 spec (sign / 4-bit exp bias 7 / 3-bit mantissa).
 *
 * Per-element memory: 4 bits + 8 bits / 16 = 4.5 bits = 0.5625 bytes/elt.
 * vs BF16 16 bits/elt = 2 bytes/elt. Compression: 3.56x.
 *
 * We use lookup tables for the 4-bit and 8-bit decodes so the inner loop is
 * pure load + table-lookup + FP32 FMA. This matches what a software emulator
 * for the future vfconv.nvfp4.bf16.v instruction would do.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "bench_common.h"
#include "m5ops.h"

/* FP4 E2M1 (sign-magnitude) decode table.
 * Encoding: 1 sign bit, 2 exp bits, 1 mantissa bit, bias 1.
 * Normal: (-1)^s * 2^(e-1) * (1 + m/2)   for e in {1,2,3}
 * Subnormal: (-1)^s * m/2 * 2^0          for e = 0  (i.e. m=0 → ±0, m=1 → ±0.5)
 * No infinity or NaN in E2M1.
 *
 * Indexing: bit3=sign, bit2:1=exp, bit0=mantissa.
 *   0000 →  +0           1000 → -0
 *   0001 → +0.5          1001 → -0.5
 *   0010 → +1            1010 → -1
 *   0011 → +1.5          1011 → -1.5
 *   0100 → +2            1100 → -2
 *   0101 → +3            1101 → -3
 *   0110 → +4            1110 → -4
 *   0111 → +6            1111 → -6
 */
static const float nvfp4_decode[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/* E4M3 (sign / 4-bit exp / 3-bit mantissa, bias 7, no inf, NaN at S.1111.111). */
static float e4m3_decode[256];

static void init_e4m3_table(void) {
    for (int i = 0; i < 256; i++) {
        int sign = (i >> 7) & 0x1;
        int exp  = (i >> 3) & 0xf;
        int man  = i & 0x7;
        float v;
        if (exp == 0xf && man == 0x7) {
            v = 0.0f;  /* NaN — treat as 0 for microbench */
        } else if (exp == 0) {
            /* Subnormal: 2^(-6) * m/8 */
            v = (float)man / 8.0f / 64.0f;
        } else {
            /* Normal: 2^(exp-7) * (1 + m/8) */
            int e = exp - 7;
            float scale;
            if (e >= 0) {
                scale = (float)(1ULL << e);
            } else {
                scale = 1.0f / (float)(1ULL << (-e));
            }
            v = scale * (1.0f + (float)man / 8.0f);
        }
        e4m3_decode[i] = sign ? -v : v;
    }
}

int main(void) {
    init_e4m3_table();

    /* Packed K: 2 elements per byte → K_NUM_ELTS/2 bytes. */
    uint8_t *K_packed = (uint8_t *)malloc(K_NUM_ELTS / 2);
    uint8_t *scales   = (uint8_t *)malloc(K_NUM_BLOCKS);
    float   *Q   = (float *)malloc(N_HEADS * HEAD_DIM * sizeof(float));
    float   *out = (float *)malloc(N_HEADS * SEQ_LEN  * sizeof(float));
    if (!K_packed || !scales || !Q || !out) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    /* Deterministic init (matches FP16 init pattern so the comparison is fair). */
    for (int i = 0; i < (int)(K_NUM_ELTS / 2); i++)
        K_packed[i] = (uint8_t)((i * 13 + 7) & 0xff);
    for (int i = 0; i < (int)K_NUM_BLOCKS; i++) {
        /* Bias E4M3 scales into the normal range (exp ∈ [4..10]) so the
         * dequant values are sensible and the table stays in L1D. */
        scales[i] = (uint8_t)(0x20 + ((i * 31 + 11) & 0x3f));
    }
    for (int h = 0; h < N_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++)
            Q[h * HEAD_DIM + d] =
                ((float)((h * 31 + d * 17) % 7) / 7.0f) - 0.5f;

    memset(out, 0, N_HEADS * SEQ_LEN * sizeof(float));

    /* ===================== MEASUREMENT REGION ===================== */
    m5_dump_reset_stats(0, 0);
    uint64_t cyc_start = rdcycle();

    for (int h = 0; h < N_HEADS; h++) {
        const float *qh = &Q[h * HEAD_DIM];
        float *outh = &out[h * SEQ_LEN];
        /* Per-row pointer math: each row has HEAD_DIM/2 bytes packed and
         * HEAD_DIM/NVFP4_BLOCK scale bytes. */
        const uint8_t *kh_packed =
            &K_packed[(size_t)h * SEQ_LEN * (HEAD_DIM / 2)];
        const uint8_t *sh =
            &scales[(size_t)h * SEQ_LEN * (HEAD_DIM / NVFP4_BLOCK)];

        for (int s = 0; s < SEQ_LEN; s++) {
            const uint8_t *ks_packed = &kh_packed[s * (HEAD_DIM / 2)];
            const uint8_t *ss        = &sh[s * (HEAD_DIM / NVFP4_BLOCK)];
            float acc = 0.0f;

            for (int blk = 0; blk < HEAD_DIM / NVFP4_BLOCK; blk++) {
                float scale = e4m3_decode[ss[blk]];
                /* 16 elements per block = 8 packed bytes. */
                for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
                    int d = blk * NVFP4_BLOCK + 2 * b;
                    uint8_t byte = ks_packed[blk * (NVFP4_BLOCK / 2) + b];
                    float k0 = nvfp4_decode[byte & 0xf] * scale;
                    float k1 = nvfp4_decode[(byte >> 4) & 0xf] * scale;
                    acc += qh[d]     * k0;
                    acc += qh[d + 1] * k1;
                }
            }
            outh[s] = acc;
        }
    }

    uint64_t cyc_end = rdcycle();
    m5_dump_stats(0, 0);
    /* ================== END MEASUREMENT REGION ==================== */

    float checksum = 0.0f;
    for (int i = 0; i < N_HEADS * SEQ_LEN; i++) checksum += out[i];

    size_t k_bytes  = K_NUM_ELTS / 2;       /* packed nibbles */
    size_t sc_bytes = K_NUM_BLOCKS;         /* E4M3 scales */

    printf("nvfp4 bench done\n");
    printf("  rdcycle delta = %lu\n", (unsigned long)(cyc_end - cyc_start));
    printf("  checksum      = %f\n", checksum);
    printf("  K bytes       = %lu  (packed)\n", (unsigned long)k_bytes);
    printf("  scale bytes   = %lu  (E4M3 per block-%d)\n",
           (unsigned long)sc_bytes, NVFP4_BLOCK);
    printf("  K + scale     = %lu\n", (unsigned long)(k_bytes + sc_bytes));

    free(K_packed); free(scales); free(Q); free(out);
    return 0;
}
