
#pragma once
#ifndef EXO_SCHEDULE_FA_H
#define EXO_SCHEDULE_FA_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


#ifndef EXO_WIN_1F32
#define EXO_WIN_1F32
struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
#endif
#ifndef EXO_WIN_1F32C
#define EXO_WIN_1F32C
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
#endif
#ifndef EXO_WIN_1UI16
#define EXO_WIN_1UI16
struct exo_win_1ui16{
    uint16_t * const data;
    const int_fast32_t strides[1];
};
#endif
#ifndef EXO_WIN_1UI16C
#define EXO_WIN_1UI16C
struct exo_win_1ui16c{
    const uint16_t * const data;
    const int_fast32_t strides[1];
};
#endif
// fa_kernel_decode_naive(
//     seq_len : size,
//     qk_scale : f32[1] @DRAM,
//     Q_fp32 : f32[8, 64] @DRAM,
//     K_nvfp4 : ui16[8, seq_len, 4, 16] @DRAM,
//     K_scale : f32[8, seq_len, 4] @DRAM,
//     V_nvfp4 : ui16[8, seq_len, 4, 16] @DRAM,
//     V_scale : f32[8, seq_len, 4] @DRAM,
//     O_fp32 : f32[8, 64] @DRAM
// )
void fa_kernel_decode_naive( void *ctxt, int_fast32_t seq_len, const float* qk_scale, const float* Q_fp32, const uint16_t* K_nvfp4, const float* K_scale, const uint16_t* V_nvfp4, const float* V_scale, float* O_fp32 );



#ifdef __cplusplus
}
#endif
#endif  // EXO_SCHEDULE_FA_H
