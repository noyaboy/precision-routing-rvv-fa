// SPDX-License-Identifier: Apache-2.0
//
// VFConvNvfp4Bf16Lane — one lane of `vfconv.nvfp4.bf16.v`.
//
// Dequantises a 4-bit NVFP4 element (E2M1, OCP spec) by an 8-bit E4M3 block
// scale (NVIDIA NVFP4 / OCP FN variant: exp=15&m=7 is the only NaN, no Inf)
// and emits a BF16 result. The full FU instantiates 16 of these in parallel
// behind one shared E4M3 scale broadcast; this file is the per-lane primitive.
//
// 3-stage pipeline (matching fu_sketch.md):
//   s1: decode NVFP4 -> BF16  (16-entry LUT)
//       decode E4M3  -> BF16  (combinational: normal / subnormal / NaN / zero)
//       precompute override flags (zero, NaN scale)
//   s2: 8x8 mantissa multiply (with hidden 1s) + 9-bit exp-sum-minus-bias +
//       sign XOR. The product is 16 bits unsigned with the leading 1 at bit
//       14 or bit 15.
//   s3: normalize (shift product left by 1 if MSB is at bit 14), RNE on the
//       discarded 8 bits, exp clamp, apply zero/NaN override, pack BF16.
//
// Latency = 3 cycles fill, then 1 elt/cycle sustained. Throughput 1/cycle
// when downstream is ready. Backpressure: simple `ready := ready` passthrough,
// same skeleton as VFExpLane; the skid buffer is Mo 4 work.
//
// Numerical target (Mo 3 second checkpoint): bit-exact vs the BF16 round of
// the FP64-computed product across all 256 x 16 = 4096 input combinations,
// excluding NaN scale (NaN encoding is implementation-defined; we return a
// QNaN with the multiply-result sign).
//
// References: fu_sketch.md (Saturn FU block diagram), precision_config.md
// (NVFP4 block=16, E4M3 scale, BF16 output convention), NVIDIA Blackwell
// NVFP4 spec (E2M1 element + E4M3 scale + FP32 per-tensor scale, where the
// per-tensor scale is folded outside this FU).

package saturnfu

import chisel3._
import chisel3.util._

/** Per-lane input: one 4-bit NVFP4 element + one 8-bit E4M3 scale. */
class VFConvNvfp4Bf16In extends Bundle {
  val elt   = UInt(4.W)   // NVFP4 (E2M1) — 16 codepoints, values in +/-{0, 0.5, 1, 1.5, 2, 3, 4, 6}
  val scale = UInt(8.W)   // E4M3 (OCP FN variant — exp=15 & m=7 == NaN, no Inf)
}

object VFConvNvfp4Bf16Consts {
  // ----- NVFP4 (E2M1) -> BF16 LUT --------------------------------------------
  // Index by raw 4-bit codepoint. Sign bit is the top bit of the codepoint,
  // so the lower 8 entries are non-negative and the upper 8 mirror them with
  // the BF16 sign bit set.
  val E2M1_TO_BF16: Seq[Int] = Seq(
    0x0000,  // 0x0  +0.0
    0x3F00,  // 0x1  +0.5
    0x3F80,  // 0x2  +1.0
    0x3FC0,  // 0x3  +1.5
    0x4000,  // 0x4  +2.0
    0x4040,  // 0x5  +3.0
    0x4080,  // 0x6  +4.0
    0x40C0,  // 0x7  +6.0
    0x8000,  // 0x8  -0.0
    0xBF00,  // 0x9  -0.5
    0xBF80,  // 0xA  -1.0
    0xBFC0,  // 0xB  -1.5
    0xC000,  // 0xC  -2.0
    0xC040,  // 0xD  -3.0
    0xC080,  // 0xE  -4.0
    0xC0C0,  // 0xF  -6.0
  )

  // BF16 constants
  val BF16_POS_INF: Int = 0x7F80
  val BF16_QNAN:    Int = 0x7FC0   // canonical positive QNaN; sign is OR'd in at use
}

class VFConvNvfp4Bf16Lane extends Module {
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new VFConvNvfp4Bf16In))
    val out = Decoupled(UInt(16.W))    // BF16
  })

  import VFConvNvfp4Bf16Consts._

  // Skeleton backpressure — same convention as VFExpLane.
  io.in.ready := io.out.ready

  val LAT = 3
  val stage_valid = RegInit(VecInit(Seq.fill(LAT)(false.B)))
  stage_valid(0) := io.in.valid && io.in.ready
  for (i <- 1 until LAT) stage_valid(i) := stage_valid(i - 1)

  // ==========================================================================
  // s1: decode NVFP4 -> BF16 and E4M3 -> BF16; compute override flags
  // ==========================================================================
  val in_elt   = io.in.bits.elt
  val in_scale = io.in.bits.scale

  // --- NVFP4 (E2M1) -> BF16 via 16-entry vector LUT ---
  val e2m1_lut    = VecInit(E2M1_TO_BF16.map(_.U(16.W)))
  val elt_bf16_c  = e2m1_lut(in_elt)
  // True iff codepoint is +0.0 (0x0) or -0.0 (0x8). Element is never NaN/Inf.
  val elt_is_zero_c = (in_elt(2, 0) === 0.U)

  // --- E4M3 (OCP FN) -> BF16 ---
  val sc_sign = in_scale(7)
  val sc_exp  = in_scale(6, 3)   // 4 bits
  val sc_mant = in_scale(2, 0)   // 3 bits

  val sc_is_nan   = (sc_exp === 0xF.U) && (sc_mant === 0x7.U)
  val sc_is_zero  = (sc_exp === 0.U)   && (sc_mant === 0.U)
  val sc_is_subn  = (sc_exp === 0.U)   && (sc_mant =/= 0.U)
  val sc_is_norm  = !sc_is_nan && !sc_is_zero && !sc_is_subn

  // Normal: BF16 exp = E4M3 exp + 120, BF16 mantissa = E4M3 mantissa << 4.
  // (E4M3 bias = 7, BF16 bias = 127, so the unbiased value lines up: +120.)
  val sc_norm_exp  = sc_exp +& 120.U(8.W)                          // 9-bit add; result fits in 8 bits
  val sc_norm_bf16 = Cat(sc_sign, sc_norm_exp(7, 0), sc_mant, 0.U(4.W))

  // Subnormal: value = m * 2^-9. Find MSB position in 3-bit mantissa.
  //   m=1 (001): MSB at bit 0 -> BF16 exp = 118, mant = 0
  //   m=2 (010): MSB at bit 1 -> BF16 exp = 119, mant = 0
  //   m=3 (011): MSB at bit 1 -> BF16 exp = 119, mant = 0x40 (bit 6 from m[0])
  //   m=4 (100): MSB at bit 2 -> BF16 exp = 120, mant = 0
  //   m=5 (101): MSB at bit 2 -> BF16 exp = 120, mant = 0x20 (bit 5 from m[0])
  //   m=6 (110): MSB at bit 2 -> BF16 exp = 120, mant = 0x40 (bit 6 from m[1])
  //   m=7 (111): MSB at bit 2 -> BF16 exp = 120, mant = 0x60 (bits 6:5 from m[1:0])
  // Inline priority encoder.
  val sc_subn_msbpos = Mux(sc_mant(2), 2.U(2.W),
                       Mux(sc_mant(1), 1.U(2.W),
                                       0.U(2.W)))
  val sc_subn_exp = 118.U(8.W) + sc_subn_msbpos
  // Bits below MSB, left-aligned into the 7-bit BF16 mantissa.
  val sc_subn_mant = MuxLookup(sc_subn_msbpos, 0.U(7.W))(Seq(
    0.U(2.W) -> 0.U(7.W),
    1.U(2.W) -> Cat(sc_mant(0), 0.U(6.W)),
    2.U(2.W) -> Cat(sc_mant(1, 0), 0.U(5.W)),
  ))
  val sc_subn_bf16 = Cat(sc_sign, sc_subn_exp, sc_subn_mant)

  // Zero -> signed zero in BF16.
  val sc_zero_bf16 = Cat(sc_sign, 0.U(15.W))
  // NaN -> sign | 0x7FC0 (positive-pattern QNaN with input scale's sign bit).
  val sc_nan_bf16  = Cat(sc_sign, 0xFF.U(8.W), 0x40.U(7.W))

  val scale_bf16_c = Mux(sc_is_nan,  sc_nan_bf16,
                      Mux(sc_is_zero, sc_zero_bf16,
                       Mux(sc_is_subn, sc_subn_bf16,
                                       sc_norm_bf16)))

  // --- s1 -> s2 boundary registers ---
  val s1_elt_bf16    = RegNext(elt_bf16_c,    0.U(16.W))
  val s1_scale_bf16  = RegNext(scale_bf16_c,  0.U(16.W))
  val s1_elt_is_zero = RegNext(elt_is_zero_c, false.B)
  val s1_sc_is_zero  = RegNext(sc_is_zero,    false.B)
  val s1_sc_is_nan   = RegNext(sc_is_nan,     false.B)

  // ==========================================================================
  // s2: 8x8 mantissa multiply, exp sum, sign XOR
  // ==========================================================================
  val a_sign = s1_elt_bf16(15)
  val a_exp  = s1_elt_bf16(14, 7)
  val a_mant = s1_elt_bf16(6, 0)

  val b_sign = s1_scale_bf16(15)
  val b_exp  = s1_scale_bf16(14, 7)
  val b_mant = s1_scale_bf16(6, 0)

  // Hidden-1 mantissas. Both decoders always produce normal BF16 for the
  // mantissa stage (subnormals and zeros are handled via the zero-override
  // flag set in s1), so a_exp and b_exp are guaranteed non-zero when the
  // multiply is "live".
  val a_mant_full = Cat(1.U(1.W), a_mant)        // 8 bits
  val b_mant_full = Cat(1.U(1.W), b_mant)        // 8 bits
  val mant_product_c = a_mant_full * b_mant_full // 16 bits unsigned

  // Biased exp sum minus one bias. Widen each operand to 10-bit signed before
  // adding — a 9-bit signed accumulator silently wraps for a_exp+b_exp > 255
  // (e.g. NVFP4 +4.0 (exp 129) * E4M3 +1.0 (exp 127) = 256), and the wrap then
  // trips the exp_too_small clamp downstream.
  val a_exp_s = Cat(0.U(2.W), a_exp).asSInt           // SInt(10.W), unsigned-extended
  val b_exp_s = Cat(0.U(2.W), b_exp).asSInt
  val exp_sum_c = a_exp_s + b_exp_s - 127.S(10.W)

  val sign_c = a_sign ^ b_sign

  // Special-case overrides (consolidate s1 flags into a single override).
  val override_valid_c = s1_elt_is_zero || s1_sc_is_zero || s1_sc_is_nan
  val override_result_c = Mux(s1_sc_is_nan,
                              Cat(sign_c, 0xFF.U(8.W), 0x40.U(7.W)),  // QNaN, sign = a XOR b
                              Cat(sign_c, 0.U(15.W)))                  // signed zero

  // --- s2 -> s3 boundary registers ---
  val s2_mant_product   = RegNext(mant_product_c,   0.U(16.W))
  val s2_exp_sum        = RegNext(exp_sum_c,        0.S(10.W))
  val s2_sign           = RegNext(sign_c,           false.B)
  val s2_override_valid = RegNext(override_valid_c, false.B)
  val s2_override_res   = RegNext(override_result_c, 0.U(16.W))

  // ==========================================================================
  // s3: normalize, RNE round, exp clamp, apply override, pack BF16
  // ==========================================================================
  // Normalize so the leading 1 sits at bit 15 of a 16-bit value.
  //   product[15] = 1  -> already there (effective value in [2, 4)); exp += 1
  //   product[15] = 0  -> leading 1 at bit 14; left-shift by 1
  val norm_value = Mux(s2_mant_product(15),
                       s2_mant_product,
                       (s2_mant_product << 1)(15, 0))
  val exp_norm_adjust = Mux(s2_mant_product(15), 1.S(2.W), 0.S(2.W))

  // RNE on the 8 bits below the kept mantissa.
  val mantissa_pre = norm_value(14, 8)        // 7 bits
  val guard        = norm_value(7)
  val sticky       = norm_value(6, 0).orR
  val round_up     = guard && (sticky || mantissa_pre(0))

  // Adding 1 to a 7-bit field can carry into an 8th bit; capture and feed
  // into the exponent.
  val mantissa_p1 = Cat(0.U(1.W), mantissa_pre) +& 1.U                  // 8 bits
  val mantissa_rounded = Mux(round_up, mantissa_p1, Cat(0.U(1.W), mantissa_pre))
  val round_carry     = mantissa_rounded(7)
  val mantissa_final  = mantissa_rounded(6, 0)

  val exp_adjusted = s2_exp_sum + exp_norm_adjust.pad(10) +
                     Mux(round_carry, 1.S(2.W), 0.S(2.W)).pad(10)

  // Clamp to BF16 normal exponent range [1, 254]. Outside this, emit
  // saturated +/-Inf (overflow) or signed zero (underflow). For our NVFP4
  // dequant domain neither path fires in practice.
  val exp_too_big   = exp_adjusted > 254.S
  val exp_too_small = exp_adjusted < 1.S
  val exp_8b        = exp_adjusted(7, 0).asUInt

  val computed_normal   = Cat(s2_sign, exp_8b, mantissa_final)
  val computed_inf      = Cat(s2_sign, 0xFF.U(8.W), 0.U(7.W))
  val computed_zero     = Cat(s2_sign, 0.U(15.W))
  val computed_clamped  = Mux(exp_too_big,   computed_inf,
                            Mux(exp_too_small, computed_zero,
                                              computed_normal))

  val s3_out = RegNext(Mux(s2_override_valid, s2_override_res, computed_clamped),
                       0.U(16.W))

  // ==========================================================================
  // Output
  // ==========================================================================
  io.out.valid := stage_valid(LAT - 1)
  io.out.bits  := s3_out
}
