// SPDX-License-Identifier: Apache-2.0
//
// VFConvFp8Bf16Lane — one lane of `vfconv.fp8.bf16.v`.
//
// Dequantises an 8-bit FP8-E4M3 (OCP FN variant) element to BF16.  Same E4M3
// convention as VFConvNvfp4Bf16Lane's scale-decoder and VFConvBf16Fp8Lane's
// output (exp=15 & m=7 is the only NaN, max normal = 1.110 * 2^8 = 448), so
// this lane is the exact inverse of vfconv.bf16.fp8.v on round-trippable
// values.
//
// The conversion is mathematically exact: BF16 has 7 mantissa bits and an 8-
// bit biased exponent, both strictly wider than E4M3-FN's 3 mantissa bits +
// 4-bit exponent.  Every finite E4M3 representable maps to a unique BF16
// without rounding loss.  Hence no guard/round/sticky logic; no cascade; one
// pipeline stage suffices.
//
// 1-stage pipeline (matching fu_sketch.md):
//   s1: combinational decode (sign + exp+120 + mantissa<<4 for normals;
//       priority-encoder + shift for subnormals; signed-zero and sign-
//       preserving QNaN paths for the special cases) -> output register.
//
// Latency = 1 cycle fill, then 1 elt/cycle sustained.  Backpressure: same
// skeleton pass-through used by the other three lanes (skid buffer is Mo 4
// work).
//
// Numerical target (Track F3): bit-exact vs the e4m3ToDouble-then-BF16-round
// golden across all 256 FP8 inputs.  Coverage includes the round-trip
// inverse of VFConvBf16Fp8Lane on round-trippable values.
//
// References: fu_sketch.md (1-cycle FP8->BF16 dequant lane), exo_instr_decls
// .md §5.3 (compiler-side @instr decl), VFConvNvfp4Bf16Lane.scala (shared
// E4M3 decoder pattern), VFConvBf16Fp8Lane.scala (the forward direction).

package saturnfu

import chisel3._
import chisel3.util._

class VFConvFp8Bf16Lane extends Module {
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(UInt(8.W)))    // FP8-E4M3 (OCP FN)
    val out = Decoupled(UInt(16.W))            // BF16
  })

  // Skeleton pass-through backpressure — same convention as the other lanes.
  io.in.ready := io.out.ready

  val LAT = 1
  val stage_valid = RegInit(VecInit(Seq.fill(LAT)(false.B)))
  stage_valid(0) := io.in.valid && io.in.ready
  for (i <- 1 until LAT) stage_valid(i) := stage_valid(i - 1)

  // ==========================================================================
  // s1: combinational FP8-E4M3 -> BF16 decode (no rounding required)
  // ==========================================================================
  val fp8        = io.in.bits
  val sign       = fp8(7)
  val exp_field  = fp8(6, 3)        // 4 bits
  val mant_field = fp8(2, 0)        // 3 bits

  val is_nan  = (exp_field === 0xF.U) && (mant_field === 0x7.U)
  val is_zero = (exp_field === 0.U)   && (mant_field === 0.U)
  val is_subn = (exp_field === 0.U)   && (mant_field =/= 0.U)
  // Implicit: is_normal = !(is_nan || is_zero || is_subn).

  // Normal: BF16 exp = E4M3 exp + 120, BF16 mantissa = E4M3 mantissa << 4.
  // (E4M3 bias 7, BF16 bias 127, delta = 120.  Mantissa widening pads low
  // bits with zero, so the BF16 value is bit-exact.)
  val norm_exp  = exp_field +& 120.U(8.W)   // 9-bit result fits in 8 bits
  val norm_bf16 = Cat(sign, norm_exp(7, 0), mant_field, 0.U(4.W))

  // Subnormal: value = mant_field * 2^-9, mant_field in {1..7}.
  // Priority-encode the MSB position of the 3-bit mantissa to recover the
  // BF16 normal form 1.bbbbbbb * 2^(118 + msbpos):
  //   m=1 (001): MSB at bit 0 -> BF16 exp = 118, mant = 0
  //   m=2 (010): MSB at bit 1 -> BF16 exp = 119, mant = 0
  //   m=3 (011): MSB at bit 1 -> BF16 exp = 119, mant = m[0] << 6
  //   m=4 (100): MSB at bit 2 -> BF16 exp = 120, mant = 0
  //   m=5 (101): MSB at bit 2 -> BF16 exp = 120, mant = m[0] << 5
  //   m=6 (110): MSB at bit 2 -> BF16 exp = 120, mant = m[1] << 6
  //   m=7 (111): MSB at bit 2 -> BF16 exp = 120, mant = m[1:0] << 5
  val subn_msbpos = Mux(mant_field(2), 2.U(2.W),
                    Mux(mant_field(1), 1.U(2.W),
                                       0.U(2.W)))
  val subn_exp = 118.U(8.W) + subn_msbpos
  val subn_mant = MuxLookup(subn_msbpos, 0.U(7.W))(Seq(
    0.U(2.W) -> 0.U(7.W),
    1.U(2.W) -> Cat(mant_field(0),    0.U(6.W)),
    2.U(2.W) -> Cat(mant_field(1, 0), 0.U(5.W)),
  ))
  val subn_bf16 = Cat(sign, subn_exp, subn_mant)

  // Zero -> signed zero in BF16.
  val zero_bf16 = Cat(sign, 0.U(15.W))
  // NaN -> sign-preserving BF16 QNaN.
  val nan_bf16  = Cat(sign, 0xFF.U(8.W), 0x40.U(7.W))

  val result_c = Mux(is_nan,  nan_bf16,
                  Mux(is_zero, zero_bf16,
                   Mux(is_subn, subn_bf16,
                                norm_bf16)))

  val s1_out = RegNext(result_c, 0.U(16.W))

  // ==========================================================================
  // Output
  // ==========================================================================
  io.out.valid := stage_valid(LAT - 1)
  io.out.bits  := s1_out
}
