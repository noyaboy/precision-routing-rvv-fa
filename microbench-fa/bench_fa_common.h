#ifndef BENCH_FA_COMMON_H
#define BENCH_FA_COMMON_H

/*
 * Flash-attention decode-step microbench common header.
 *
 * Shape: one query token attending to seq_len KV entries, repeated across
 * N_HEADS independent heads.  Matches Llama-3.2-1B GQA decode shape used in
 * Mo 2: n_kv_heads=8, head_dim=64, seq_len=2048.
 *
 *   Q [N_HEADS, HEAD_DIM]               BF16 (per precision_config.md)
 *   K [N_HEADS, SEQ_LEN, HEAD_DIM]      BF16 baseline / NVFP4 mixed-prec
 *   V [N_HEADS, SEQ_LEN, HEAD_DIM]      BF16 baseline / NVFP4 mixed-prec
 *   S [N_HEADS, SEQ_LEN]                BF16 (logits)
 *   P [N_HEADS, SEQ_LEN]                BF16 baseline / FP8-E4M3 mixed-prec
 *   O [N_HEADS, HEAD_DIM]               BF16
 *
 * Per-head decode is independent (no cross-head reductions), so the loop
 * structure is for-head -> { QK^T, softmax, P*V }.
 */

#include <stdint.h>
#include <math.h>

#define N_HEADS      8
#ifndef HEAD_DIM
#define HEAD_DIM     64
#endif
#ifndef SEQ_LEN
#define SEQ_LEN      2048
#endif
#define NVFP4_BLOCK  16

#define K_NUM_ELTS    (N_HEADS * SEQ_LEN * HEAD_DIM)
#define K_NUM_BLOCKS  (K_NUM_ELTS / NVFP4_BLOCK)
#define K_PACKED_BYTES (K_NUM_ELTS / 2)
#define K_SCALE_BYTES  (K_NUM_BLOCKS)

typedef uint16_t bf16_t;

static inline float bf16_to_fp32(bf16_t b) {
    union { uint32_t u; float f; } u;
    u.u = ((uint32_t)b) << 16;
    return u.f;
}

static inline bf16_t fp32_to_bf16(float f) {
    /* Round-to-nearest-even on the 16 low bits. */
    union { uint32_t u; float f; } u;
    u.f = f;
    uint32_t bits = u.u;
    uint32_t lower = bits & 0xFFFFu;
    uint32_t upper = (bits >> 16) & 0xFFFFu;
    uint32_t rne = (lower > 0x8000u) ? 1u
                  : (lower < 0x8000u) ? 0u
                  : (upper & 1u);
    return (bf16_t)((upper + rne) & 0xFFFFu);
}

/* ============================================================
 * Shared FP32 ground truth.  Both BF16 and mixed-precision paths derive
 * their K/V tensors from these generators so the kernels compute over the
 * same underlying numerical reality.  Per-path quantization differs but
 * the FP32 source is identical.
 *
 * Range: K and V values lie in [-0.5, 0.5].  Stable softmax inputs and a
 * good fit for NVFP4 with per-block-16 E4M3 scale (typical scale ~0.08).
 * ============================================================ */
static inline float init_q_fp32(int h, int d) {
    return ((float)((h * 31 + d * 17) % 7) / 7.0f) - 0.5f;
}
static inline float init_k_fp32(int h, int s, int d) {
    int idx = (h * SEQ_LEN + s) * HEAD_DIM + d;
    return ((float)((idx * 13 + 7) % 11) / 11.0f) - 0.5f;
}
static inline float init_v_fp32(int h, int s, int d) {
    int idx = (h * SEQ_LEN + s) * HEAD_DIM + d;
    return ((float)((idx * 17 + 13) % 13) / 13.0f) - 0.5f;
}

/* ============================================================
 * NVFP4 (E2M1, OCP) codepoint table.  Index = 4-bit code.
 * ============================================================ */
static const float nvfp4_decode_table[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/* ============================================================
 * E4M3 (OCP FN) decode + encode.
 * ============================================================ */
static inline void e4m3_init_table(float table[256]) {
    for (int i = 0; i < 256; i++) {
        int sign = (i >> 7) & 1;
        int exp  = (i >> 3) & 0xF;
        int man  = i & 0x7;
        float v;
        if (exp == 0xF && man == 0x7) v = 0.0f;            /* NaN -> 0 */
        else if (exp == 0)            v = (float)man * 0.001953125f;  /* m * 2^-9 */
        else {
            int e = exp - 7;
            float scale = (e >= 0) ? (float)(1ULL << e) : 1.0f / (float)(1ULL << (-e));
            v = scale * (1.0f + (float)man / 8.0f);
        }
        table[i] = sign ? -v : v;
    }
}

/* BF16 -> FP8-E4M3 (OCP FN) RNE with saturation. Mirrors
 * VFConvBf16Fp8Lane.scala semantics; bit-exact across roundtrippable
 * values per Track F2. */
static inline uint8_t bf16_to_e4m3(bf16_t bits) {
    int sign  = (bits >> 15) & 1;
    int bexp  = (bits >> 7)  & 0xFF;
    int bmant = bits & 0x7F;

    if (bexp == 0xFF)  return (sign << 7) | (bmant ? 0x7F : 0x7E);  /* NaN / Inf */
    if (bexp == 0)     return sign << 7;                            /* zero / subn -> 0 */

    int target = bexp - 120;
    int full = 128 + bmant;

    if (target >= 16) return (sign << 7) | 0x7E;
    if (target <= -4) return sign << 7;

    int em_pre, guard, sticky;
    if (target >= 1) {
        em_pre = (bmant >> 4) & 0x7;
        guard  = (bmant >> 3) & 1;
        sticky = (bmant & 0x7) != 0;
    } else if (target == 0) {
        em_pre = (full >> 5) & 0x7;
        guard  = (full >> 4) & 1;
        sticky = (full & 0xF) != 0;
    } else if (target == -1) {
        em_pre = (full >> 6) & 0x7;
        guard  = (full >> 5) & 1;
        sticky = (full & 0x1F) != 0;
    } else if (target == -2) {
        em_pre = (full >> 7) & 0x7;
        guard  = (full >> 6) & 1;
        sticky = (full & 0x3F) != 0;
    } else { /* target == -3 */
        em_pre = 0;
        guard  = 1;
        sticky = (bmant & 0x7F) != 0;
    }
    int round_up = guard && (sticky || (em_pre & 1));
    int em_full = em_pre + round_up;
    int cascade = (em_full >> 3) & 1;
    int em_out = em_full & 0x7;

    if (target >= 1) {
        int e_final = target + cascade;
        if (e_final > 15 || (e_final == 15 && em_out == 7))
            return (sign << 7) | 0x7E;
        return (sign << 7) | (e_final << 3) | em_out;
    } else {
        int e_final = cascade ? 1 : 0;
        return (sign << 7) | (e_final << 3) | em_out;
    }
}

static inline uint8_t fp32_to_e4m3(float f) {
    return bf16_to_e4m3(fp32_to_bf16(f));
}

/* FP32 -> NVFP4 (E2M1 OCP).  Round-to-nearest with round-half-up on the
 * 7 magnitude midpoints (init data won't hit ties exactly; RNE on these
 * would only flip ~0% of outputs and isn't worth the branchy code). */
static inline uint8_t fp32_to_nvfp4(float x) {
    int sign = (x < 0.0f) ? 1 : 0;
    float a = fabsf(x);
    uint8_t mag;
    if      (a < 0.25f) mag = 0;       /* 0.0 */
    else if (a < 0.75f) mag = 1;       /* 0.5 */
    else if (a < 1.25f) mag = 2;       /* 1.0 */
    else if (a < 1.75f) mag = 3;       /* 1.5 */
    else if (a < 2.5f)  mag = 4;       /* 2.0 */
    else if (a < 3.5f)  mag = 5;       /* 3.0 */
    else if (a < 5.0f)  mag = 6;       /* 4.0 */
    else                mag = 7;       /* 6.0 */
    return (uint8_t)((sign << 3) | mag);
}

/* ============================================================
 * Per-block-16 NVFP4+E4M3 quantization helper.
 *
 * For each block of 16 contiguous FP32 values:
 *   amax = max |x_i|
 *   scale = amax / 6                (so the largest mapped to NVFP4 +6)
 *   scale_e4m3 = round-to-E4M3(scale)
 *   each x_i / scale_decoded -> NVFP4 codepoint (round-to-nearest)
 *
 * Caller provides packed-byte and scale-byte output buffers and the FP32
 * input buffer of NVFP4_BLOCK elements.
 * ============================================================ */
static inline void quantize_nvfp4_block(const float in_fp32[NVFP4_BLOCK],
                                        const float e4m3_decode[256],
                                        uint8_t out_packed[NVFP4_BLOCK / 2],
                                        uint8_t *out_scale) {
    float amax = 0.0f;
    for (int i = 0; i < NVFP4_BLOCK; i++) {
        float a = fabsf(in_fp32[i]);
        if (a > amax) amax = a;
    }
    /* Guard the all-zeros case: pick a tiny non-zero scale so the decode
     * still produces zero (each x_i / scale_decoded < 0.25 -> code 0). */
    float scale_target = (amax > 1e-7f) ? (amax / 6.0f) : (1.0f / 64.0f);
    uint8_t scale_byte = fp32_to_e4m3(scale_target);
    *out_scale = scale_byte;
    float scale_decoded = e4m3_decode[scale_byte];
    if (scale_decoded <= 0.0f) scale_decoded = 1.0f / 64.0f;

    for (int b = 0; b < NVFP4_BLOCK / 2; b++) {
        float v0 = in_fp32[2 * b]     / scale_decoded;
        float v1 = in_fp32[2 * b + 1] / scale_decoded;
        uint8_t n0 = fp32_to_nvfp4(v0);
        uint8_t n1 = fp32_to_nvfp4(v1);
        out_packed[b] = (uint8_t)((n1 << 4) | n0);
    }
}

#endif
