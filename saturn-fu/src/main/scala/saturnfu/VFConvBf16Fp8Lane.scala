// SPDX-License-Identifier: Apache-2.0
//
// VFConvBf16Fp8Lane — one lane of `vfconv.bf16.fp8.v`.
//
// Quantises a BF16 attention-weight scalar to an 8-bit FP8-E4M3 (OCP FN
// variant: exp=15 & m=7 is the only NaN, no Inf — same convention as the E4M3
// scale handling in VFConvNvfp4Bf16Lane so the two converters are exact
// inverses on round-trippable values).
//
// 2-stage pipeline (matching fu_sketch.md):
//   s1: decode BF16 (sign / 8-bit biased exp / 7-bit mantissa)
//       compute signed target E4M3 biased exp = bf_biased - 120
//       classify input (zero / subn / Inf / NaN / normal)
//   s2: dispatch by target into one of {overflow, normal-output, subn-output,
//       underflow}.  Apply RNE rounding (with cascade) on the full 8-bit
//       mantissa.  Saturate at +/-max (0x7E / 0xFE); pre-empt NaN-encoding
//       (exp=15 & m=7) at the normal/post-cascade boundary.  Pack the byte.
//
// Latency = 2 cycles fill, then 1 elt/cycle sustained.  Backpressure: skeleton
// pass-through `io.in.ready := io.out.ready` (skid buffer is Mo 4 work, same
// as the other two lanes).
//
// Numerical target (Track F2): bit-exact vs the FP32-and-round-to-E4M3-FN
// golden across all 65536 BF16 inputs.  Includes saturation at the
// NaN-encoding boundary, RNE ties, subnormal-output ranges, BF16 NaN
// pass-through and BF16 Inf saturation.
//
// References: fu_sketch.md (2-cycle BF16->FP8 quant lane sketch),
// precision_config.md (FP8-E4M3 attn-weights convention), OCP MXFP8 spec
// (E4M3 FN: bias 7, single NaN encoding at exp=15&m=7, max normal = 1.110 *
// 2^8 = 448).

package saturnfu

import chisel3._
import chisel3.util._

class VFConvBf16Fp8Lane extends Module {
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(UInt(16.W)))   // BF16
    val out = Decoupled(UInt(8.W))             // FP8-E4M3 (OCP FN)
  })

  // Skeleton pass-through backpressure — same convention as VFConvNvfp4Bf16Lane.
  io.in.ready := io.out.ready

  val LAT = 2
  val stage_valid = RegInit(VecInit(Seq.fill(LAT)(false.B)))
  stage_valid(0) := io.in.valid && io.in.ready
  for (i <- 1 until LAT) stage_valid(i) := stage_valid(i - 1)

  // ==========================================================================
  // s1: decode BF16, classify, compute signed target E4M3 biased exp
  // ==========================================================================
  val bf16  = io.in.bits
  val sign  = bf16(15)
  val bexp  = bf16(14, 7)         // 8-bit biased BF16 exp
  val bmant = bf16(6, 0)          // 7-bit BF16 mantissa (no implicit 1)

  val is_bf16_zero = (bexp === 0.U)    && (bmant === 0.U)
  val is_bf16_subn = (bexp === 0.U)    && (bmant =/= 0.U)
  val is_bf16_inf  = (bexp === 0xff.U) && (bmant === 0.U)
  val is_bf16_nan  = (bexp === 0xff.U) && (bmant =/= 0.U)

  // BF16 subnormals are below 2^-126, way under E4M3's smallest subnormal
  // (2^-9), so RNE rounds them to signed zero.  Fold into the zero-or-subn
  // override flag.
  val is_zero_in = is_bf16_zero || is_bf16_subn

  // target = bexp_unbiased_BF16 + bias_E4M3 = (bexp - 127) + 7 = bexp - 120.
  // Use SInt(10.W) so the subtraction never wraps even for bexp = 0.
  val target_c = bexp.zext - 120.S(10.W)

  // --- s1 -> s2 boundary registers ---
  val s1_sign       = RegNext(sign,       false.B)
  val s1_bmant      = RegNext(bmant,      0.U(7.W))
  val s1_target     = RegNext(target_c,   0.S(10.W))
  val s1_is_zero_in = RegNext(is_zero_in, false.B)
  val s1_is_inf     = RegNext(is_bf16_inf, false.B)
  val s1_is_nan     = RegNext(is_bf16_nan, false.B)

  // ==========================================================================
  // s2: regime dispatch + RNE round + saturation + pack
  // ==========================================================================

  // Output-regime predicates (in terms of the signed target).
  //   target >= 16              -> overflow (saturate +/-max)
  //   target in [1, 15]         -> normal output
  //   target in [-3, 0]         -> subnormal output
  //   target <= -4              -> underflow (round to signed zero)
  val is_overflow  = s1_target > 15.S
  val is_normal_o  = (s1_target > 0.S)  && (s1_target <= 15.S)
  val is_subn_o    = (s1_target > -4.S) && (s1_target <= 0.S)
  // is_underflow = !is_overflow && !is_normal_o && !is_subn_o (target <= -4)

  // --- Build (em_pre, guard, sticky) for RNE.  em_pre is the 3-bit
  //     pre-rounding mantissa; guard is the bit immediately below; sticky is
  //     the OR of all bits below the guard.
  //
  // Normal output (target in [1, 15]): keep bmant[6:4]; guard = bmant[3];
  //   sticky = bmant[2:0].orR.  (The BF16 implicit 1 sits above bmant[6] and
  //   becomes the implicit 1 of the E4M3 normal — discarded for storage.)
  //
  // Subnormal output: the implicit 1 of BF16 gets folded into the explicit
  //   3-bit subnormal mantissa.  Per target value:
  //     target = 0  : right-shift-by-5  -> em_pre = {1, bm[6], bm[5]},
  //                   guard = bm[4], sticky = bm[3:0].orR
  //     target = -1 : right-shift-by-6  -> em_pre = {0, 1, bm[6]},
  //                   guard = bm[5], sticky = bm[4:0].orR
  //     target = -2 : right-shift-by-7  -> em_pre = {0, 0, 1},
  //                   guard = bm[6], sticky = bm[5:0].orR
  //     target = -3 : right-shift-by-8  -> em_pre = 0,
  //                   guard = 1 (the BF16 implicit 1), sticky = bm[6:0].orR
  //   (For target = -3, guard is unconditionally 1 because the BF16 implicit
  //   1 is the only bit at position shift_amount-1 = 7.)
  val em_pre = Wire(UInt(3.W))
  val guard  = Wire(Bool())
  val sticky = Wire(Bool())

  em_pre := 0.U
  guard  := false.B
  sticky := false.B

  when (is_normal_o) {
    em_pre := s1_bmant(6, 4)
    guard  := s1_bmant(3)
    sticky := s1_bmant(2, 0).orR
  } .elsewhen (is_subn_o) {
    when (s1_target === 0.S) {
      em_pre := Cat(1.U(1.W), s1_bmant(6, 5))
      guard  := s1_bmant(4)
      sticky := s1_bmant(3, 0).orR
    } .elsewhen (s1_target === (-1).S(10.W)) {
      em_pre := Cat(0.U(1.W), 1.U(1.W), s1_bmant(6))
      guard  := s1_bmant(5)
      sticky := s1_bmant(4, 0).orR
    } .elsewhen (s1_target === (-2).S(10.W)) {
      em_pre := Cat(0.U(2.W), 1.U(1.W))
      guard  := s1_bmant(6)
      sticky := s1_bmant(5, 0).orR
    } .otherwise { // target === -3
      em_pre := 0.U
      guard  := true.B
      sticky := s1_bmant(6, 0).orR
    }
  }

  val round_up  = guard && (sticky || em_pre(0))
  val em_full   = Cat(0.U(1.W), em_pre) +& round_up.asUInt   // 4 bits
  val cascade   = em_full(3)
  val em_out    = em_full(2, 0)

  // Output exponent for the two regimes.
  //   Normal regime: starts at s1_target, may increment on cascade.  If post-
  //     cascade exp exceeds 15 OR (exp==15 && em==7), saturate to (15, 6).
  //   Subnormal regime: starts at 0; on cascade becomes 1 (=> min normal
  //     1.000 * 2^-6).
  val exp_norm_raw      = s1_target + Mux(cascade, 1.S(2.W), 0.S(2.W))
  val overflow_after    = (exp_norm_raw > 15.S) || ((exp_norm_raw === 15.S) && (em_out === 7.U))
  val exp_norm_fld      = exp_norm_raw(3, 0).asUInt
  val em_norm_fld       = em_out

  val normal_packed     = Mux(overflow_after,
                              Cat(s1_sign, 0xF.U(4.W), 6.U(3.W)),
                              Cat(s1_sign, exp_norm_fld, em_norm_fld))

  val subn_exp_fld      = Mux(cascade, 1.U(4.W), 0.U(4.W))
  val subn_packed       = Cat(s1_sign, subn_exp_fld, em_out)

  // Saturated / special outputs.
  val sat_max           = Cat(s1_sign, 0xF.U(4.W), 6.U(3.W))   // +/-max = 448
  val sat_zero          = Cat(s1_sign, 0.U(7.W))               // +/-0
  val nan_out           = Cat(s1_sign, 0xF.U(4.W), 7.U(3.W))   // sign-preserving NaN

  // Final select.  BF16 NaN comes first so it doesn't get misclassified as
  // an overflow (target >= 16 also holds for BF16 NaN inputs, since bexp =
  // 0xff -> target = 135).
  val result = Mux(s1_is_nan,                       nan_out,
               Mux(s1_is_inf || is_overflow,        sat_max,
               Mux(s1_is_zero_in,                   sat_zero,
               Mux(is_subn_o,                       subn_packed,
               Mux(is_normal_o,                     normal_packed,
                                                    sat_zero)))))  // underflow

  val s2_out = RegNext(result, 0.U(8.W))

  // ==========================================================================
  // Output
  // ==========================================================================
  io.out.valid := stage_valid(LAT - 1)
  io.out.bits  := s2_out
}
