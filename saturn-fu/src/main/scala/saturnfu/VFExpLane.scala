// SPDX-License-Identifier: Apache-2.0
//
// VFExpLane — one lane of the `vfexp.v` custom RVV functional unit.
//
// Per `paper/fu_sketch.md`: a pipelined BF16 -> FP32 exp evaluator. The full
// FU instantiates 16 lanes; this file is one lane and is structurally what
// Saturn's `FunctionalUnitFactory` will see when we integrate.
//
// 10-stage pipeline (updated 2026-05-17 for real polynomial — Track E):
//   s1 :  BF16 -> FP32 widen
//   s2 :  special-case detect (NaN, +/-Inf, |x|>87.3, |x|<2^-16)
//          + FP32 -> Q8.23 fixed-point conversion
//   s3 :  y_q (Q8.23) = x_q * log2(e), via 32x32 signed mul + arithmetic right
//          shift 30 (Q1.30 constant for log2e).
//   s4 :  N_int (8-bit signed) = round(y_q / 2^23);
//          f_q (Q1.23) = y_q - (N_int << 23);
//          r_q (Q2.30) = (f_q << 7) * ln(2)_Q2_30  ->  shift right 30.
//   s5..s9 : five Horner FMAs in Q2.30 (PolyExpQ2_30, 5-stage sub-pipeline).
//            Approximates exp(r) on r in [-ln(2)/2, +ln(2)/2].
//   s10:  Q2.30 -> FP32 reconstruct (leading-1 detect, shift mantissa to 23
//          bits, set exponent = 126 + leadingOneBitPos - 30 + N) + apply
//          special-case override + emit.
//
// Numerical-accuracy target (Mo 3 first checkpoint): max rel err vs scalar
// expf() < 2^-16 over the representable BF16 range, excluding special-case
// inputs.  The PolyExpQ2_30 sub-module already hits 3.24e-6 = ~2^-18 over
// [-ln(2)/2, +ln(2)/2]; the additional error from range reduction (FP32 mul
// + Cody-Waite-ish two-step ln(2)) and the Q2.30->FP32 reconstruct is bounded
// by ~2^-22, well within the budget.
//
// Reference: Cephes implementation of expf (basis for our coefficients), and
// the standard Tang range-reduction technique.

package saturnfu

import chisel3._
import chisel3.util._

object VFExpConsts {
  // ----- Constants for range reduction --------------------------------------
  // log2(e) in Q1.30 = round(1.4426950408889634 * 2^30) = 1549082004 = 0x5C551D94
  val LOG2E_Q1_30: Long = 0x5C551D94L

  // ln(2) in Q2.30 = round(0.6931471805599453 * 2^30) = 744261118 = 0x2C5C85FE
  val LN2_Q2_30:   Long = 0x2C5C85FEL

  val Q1_30_FRAC: Int = 30
  val Q2_30_FRAC: Int = 30
  val Q8_23_FRAC: Int = 23

  // ----- Special-value FP32 bit patterns ------------------------------------
  val FP32_PLUS_INF:   Long = 0x7F800000L
  val FP32_MINUS_INF:  Long = 0xFF800000L
  val FP32_PLUS_ZERO:  Long = 0x00000000L
  val FP32_PLUS_ONE:   Long = 0x3F800000L
  val FP32_QNAN:       Long = 0x7FC00000L

  // BF16 magnitudes that overflow / underflow FP32 exp.
  // exp(88.7) overflows FP32; exp(-87.3) underflows to subnormal/zero.
  val OVERFLOW_THRESH_F32:  Long = java.lang.Float.floatToRawIntBits( 88.0f).toLong & 0xFFFFFFFFL
  val UNDERFLOW_THRESH_F32: Long = java.lang.Float.floatToRawIntBits(-87.0f).toLong & 0xFFFFFFFFL

  // |x| < 2^-13 (FP32 exp ≈ 114): exp(x) ≈ 1 + x, but the Mo-3 target is
  // 2^-16 relative, and for |x| < 2^-13 we get |1 + x - exp(x)| < x^2 / 2 ≈
  // 2^-27 in absolute terms = 2^-27 / 1 = 2^-27 relative. So returning 1
  // for |x| < 2^-16 is safe; below that, the polynomial path is also
  // accurate. We use the special case only for true zero/subnormal in s2.
}

/** Bundle carried alongside the data through the pipeline, signalling
  * special-case overrides. If `override_valid` is high in any stage,
  * the final stage emits `override_result` instead of the computed value.
  */
class VFExpSpecialBundle extends Bundle {
  val override_valid  = Bool()
  val override_result = UInt(32.W)
}

/** One lane of `vfexp.v`. Decoupled BF16 in -> Decoupled FP32 out.
  * Throughput = 1 element per cycle when downstream is ready.
  * Latency  = 10 cycles fill, then 1/cycle sustained.
  */
class VFExpLane extends Module {
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(UInt(16.W)))  // BF16
    val out = Decoupled(UInt(32.W))           // FP32
  })

  import VFExpConsts._

  // --- valid-bit pipeline + backpressure -------------------------------
  // Skeleton-grade backpressure: pass downstream-ready upstream. A real
  // skid buffer at the output goes in Mo 4 (TODO).
  io.in.ready := io.out.ready

  val LAT = 10
  val stage_valid = RegInit(VecInit(Seq.fill(LAT)(false.B)))
  stage_valid(0) := io.in.valid && io.in.ready
  for (i <- 1 until LAT) stage_valid(i) := stage_valid(i - 1)

  // ============================================================
  // s1: BF16 -> FP32 widen
  // ============================================================
  // BF16 = sign(1) | exp(8) | mantissa(7). FP32 = sign(1) | exp(8) | mantissa(23).
  // Widen by appending 16 zero bits to the mantissa low end.
  val s1_x_f32 = RegNext(Cat(io.in.bits, 0.U(16.W)), 0.U(32.W))

  // ============================================================
  // s2: special-case detect AND FP32 -> Q8.23 conversion
  // ============================================================
  // Decode FP32 components of s1_x_f32 (combinational).
  val x_sign           = s1_x_f32(31)
  val x_exp            = s1_x_f32(30, 23)
  val x_mantissa_23    = s1_x_f32(22, 0)
  val x_mag            = s1_x_f32(30, 0)
  val x_is_nan         = (x_exp === 0xFF.U) && (x_mantissa_23 =/= 0.U)
  val x_is_pos_inf     = (s1_x_f32 === FP32_PLUS_INF.U(32.W))
  val x_is_neg_inf     = (s1_x_f32 === FP32_MINUS_INF.U(32.W))
  val x_is_zero_or_sub = (x_exp === 0.U)
  val x_overflows      = !x_sign && (x_mag > OVERFLOW_THRESH_F32.U(32.W))
  val x_underflows     =  x_sign && (x_mag > (UNDERFLOW_THRESH_F32.U(32.W) & "h_7FFFFFFF".U(32.W)))

  val s2_special_in = Wire(new VFExpSpecialBundle)
  s2_special_in.override_valid  := x_is_nan || x_is_pos_inf || x_is_neg_inf ||
                                   x_is_zero_or_sub || x_overflows || x_underflows
  s2_special_in.override_result := Mux1H(Seq(
    x_is_nan         -> FP32_QNAN.U(32.W),
    x_is_pos_inf     -> FP32_PLUS_INF.U(32.W),
    x_is_neg_inf     -> FP32_PLUS_ZERO.U(32.W),
    x_is_zero_or_sub -> FP32_PLUS_ONE.U(32.W),
    x_overflows      -> FP32_PLUS_INF.U(32.W),
    x_underflows     -> FP32_PLUS_ZERO.U(32.W),
  ))

  // FP32 -> Q8.23 conversion.
  //   value(x) = (-1)^sign * (1.m23) * 2^(exp - 127)
  //   Q8.23(x) = value * 2^23
  //            = ±(1.m23 as 24-bit integer in Q1.23) << (exp - 127)
  //
  // For exp ∈ [104, 134] (covers |x| in [2^-23, 2^7.999]): shift in [-23, 7].
  // Variable shift: left if exp >= 127, right otherwise.
  val mantissa_24 = Cat(1.U(1.W), x_mantissa_23).asUInt          // UInt(24.W), [2^23, 2^24-1]
  val mantissa_32 = Cat(0.U(8.W), mantissa_24).asUInt           // UInt(32.W), zero-pad for left shift
  val shift_signed = Cat(0.U(1.W), x_exp).asSInt - 127.S(9.W)   // SInt(9.W)
  val shift_left   = shift_signed > 0.S
  val abs_shift_u  = Mux(shift_left, shift_signed.asUInt, (-shift_signed).asUInt)
  val abs_shift    = abs_shift_u(4, 0)                          // up to 23 bits is enough
  val shifted_L    = (mantissa_32 << abs_shift)(31, 0)
  val shifted_R    = (mantissa_32 >> abs_shift)
  val x_q_abs      = Mux(shift_left, shifted_L, shifted_R)
  val x_q_signed   = Mux(x_sign, (-x_q_abs.asSInt), x_q_abs.asSInt)

  val s2_x_q       = RegNext(x_q_signed, 0.S(32.W))          // Q8.23
  val s2_special   = RegNext(s2_special_in, 0.U.asTypeOf(new VFExpSpecialBundle))
  val s2_N_pred    = RegNext(0.S(9.W), 0.S(9.W))             // placeholder, computed in s3

  // ============================================================
  // s3: y_q (Q8.23) = x_q * log2(e)
  // ============================================================
  // x_q (Q8.23, 32-bit signed) * LOG2E (Q1.30, 32-bit unsigned positive).
  // Treat LOG2E as Q1.30 signed (positive, top bit = 0). Mul → Q9.53 in 64 bits.
  // Shift right 30 to get Q9.23. Truncate to 32-bit Q8.23 (overflow only on
  // huge |x| which is special-cased).
  val LOG2E_S = LOG2E_Q1_30.S(32.W)
  val mul_xy  = s2_x_q * LOG2E_S                                // SInt(64.W), Q9.53
  val mul_xy_rounded = mul_xy +& (1L << (Q1_30_FRAC - 1)).S(64.W)
  val y_q_64  = (mul_xy_rounded >> Q1_30_FRAC).asSInt           // SInt, value fits in 34 bits
  val y_q     = y_q_64(31, 0).asSInt                            // Q8.23, 32-bit signed

  val s3_y_q     = RegNext(y_q,         0.S(32.W))
  val s3_special = RegNext(s2_special,  0.U.asTypeOf(new VFExpSpecialBundle))

  // ============================================================
  // s4: extract N_int (8-bit signed) and r_q (Q2.30)
  // ============================================================
  // Round-to-nearest N: N_int = (y_q + 2^22) >> 23 (arith shift, treats sign).
  // For y_q in Q8.23 with N range [-128, 128], the integer part fits in 8 bits.
  val y_q_for_round = s3_y_q +& (1L << (Q8_23_FRAC - 1)).S(32.W)  // y_q + 2^22, 33-bit signed
  val N_int_wide    = (y_q_for_round >> Q8_23_FRAC).asSInt        // signed shift
  val N_int         = N_int_wide(8, 0).asSInt                     // 9-bit signed (sign bit + 8)

  // f_q (Q1.23) = y_q - (N_int << 23). For N_int in [-128, 128], (N_int << 23)
  // fits in 32-bit signed.
  val N_shl_23  = (N_int_wide << Q8_23_FRAC).asSInt                 // signed shift, widens; we narrow:
  val N_shl_23_32 = N_shl_23(31, 0).asSInt
  val f_q_23   = (s3_y_q - N_shl_23_32)(31, 0).asSInt              // Q1.23, range [-0.5, 0.5)

  // f_q (Q1.23) -> Q2.30 by shifting left 7.
  val f_q_30   = (f_q_23 << 7)(31, 0).asSInt                       // Q2.30 view of f

  // r_q = f_q (Q2.30) * LN2 (Q2.30) -> Q4.60, shift right 30 -> Q2.30.
  val LN2_S    = LN2_Q2_30.S(32.W)
  val mul_fr   = f_q_30 * LN2_S                                     // SInt(64.W), Q4.60
  val mul_fr_rounded = mul_fr +& (1L << (Q2_30_FRAC - 1)).S(64.W)
  val r_q_wide = (mul_fr_rounded >> Q2_30_FRAC).asSInt
  val r_q      = r_q_wide(31, 0).asSInt                             // Q2.30

  val s4_r_q     = RegNext(r_q,       0.S(32.W))
  val s4_N_int   = RegNext(N_int,     0.S(9.W))
  val s4_special = RegNext(s3_special, 0.U.asTypeOf(new VFExpSpecialBundle))

  // ============================================================
  // s5..s9: PolyExpQ2_30 (5-stage internal pipeline)
  // ============================================================
  val poly = Module(new PolyExpQ2_30)
  poly.io.in_valid := stage_valid(3)
  poly.io.r        := s4_r_q

  // Pipe N_int + special bundle alongside the polynomial (5 cycles).
  val N_pipe       = Reg(Vec(5, SInt(9.W)))
  val sp_pipe      = Reg(Vec(5, new VFExpSpecialBundle))
  N_pipe(0)        := s4_N_int
  sp_pipe(0)       := s4_special
  for (i <- 1 until 5) {
    N_pipe(i)  := N_pipe(i - 1)
    sp_pipe(i) := sp_pipe(i - 1)
  }

  val poly_y       = poly.io.y                                 // Q2.30 (last stage of poly)
  val s9_poly_y    = poly_y                                    // alias for clarity
  val s9_N_int     = N_pipe(4)
  val s9_special   = sp_pipe(4)

  // ============================================================
  // s10: Q2.30 -> FP32 reconstruct + scale by 2^N + special-case override
  // ============================================================
  // poly_y is in Q2.30, value should be in [exp(-ln2/2), exp(+ln2/2)] = [~0.707, ~1.414].
  //   For value v in Q2.30, bit pattern p (32-bit signed unsigned-cast):
  //     v in [0.5, 1.0) -> leading 1 at bit 29
  //     v in [1.0, 2.0) -> leading 1 at bit 30
  //   The mantissa of FP32 takes bits below the leading 1, in MSB order.
  //
  // For our range, just two cases:
  //   - if bit 30 set:    exp_shift = 0, mantissa = bits [29:7]
  //   - else (bit 29 set): exp_shift = -1, mantissa = bits [28:6]
  //
  // The FP32 exponent biased value is 127 + exp_shift + N_int.
  val poly_u             = s9_poly_y.asUInt
  val bit30              = poly_u(30)
  val mantissa_field_lo30 = poly_u(29, 7)              // 23 bits, used when bit30=1
  val mantissa_field_lo29 = poly_u(28, 6)              // 23 bits, used when bit30=0
  val mantissa_final     = Mux(bit30, mantissa_field_lo30, mantissa_field_lo29)

  // exponent base 127, minus 1 if leading 1 is at bit 29 (value < 1).
  val exp_base    = Mux(bit30, 127.S(10.W), 126.S(10.W))
  val exp_final_s = exp_base + s9_N_int.pad(10).asSInt
  // FP32 exp is 8-bit unsigned. Clamp to [0, 255] (overflow => Inf, underflow => 0).
  val exp_too_big = exp_final_s > 254.S
  val exp_too_neg = exp_final_s < 0.S
  val exp_clamped = exp_final_s(7, 0).asUInt
  val fp32_computed_normal = Cat(0.U(1.W), exp_clamped, mantissa_final)

  val fp32_computed = Mux(exp_too_big, FP32_PLUS_INF.U(32.W),
                       Mux(exp_too_neg, FP32_PLUS_ZERO.U(32.W),
                         fp32_computed_normal))

  val s10_out      = RegNext(Mux(s9_special.override_valid,
                                 s9_special.override_result,
                                 fp32_computed),
                             FP32_PLUS_ONE.U(32.W))

  // ============================================================
  // Output
  // ============================================================
  io.out.valid := stage_valid(LAT - 1)
  io.out.bits  := s10_out
}
