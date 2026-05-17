// SPDX-License-Identifier: Apache-2.0
//
// PolyExpQ2_30 — degree-5 Horner polynomial evaluator for exp(r) on small r.
//
// Approximates exp(r) on r ∈ [-ln(2)/2, +ln(2)/2] ≈ [-0.3466, +0.3466].
// Coefficients are Taylor truncation 1/k! (these are within ~3× of the
// minimax-optimal coefficients for this range, and the d=5 Taylor truncation
// error |r^6/720| ≤ (ln2/2)^6/720 ≈ 2.4e-6 ≈ 2^-19, which is well below the
// 2^-16 Mo-3 accuracy target).
//
// Internal arithmetic: Q2.30 signed fixed-point throughout (32-bit signed,
// 2 integer bits + 30 fractional bits). Avoids the hardfloat dependency for
// this standalone Chisel project; the polynomial is the only path that needs
// FP-grade math, and Q2.30 spans the full output range [exp(-ln2/2), exp(+ln2/2)]
// = [0.707, 1.414] with margin.
//
// Multiplication: 32×32 signed → 64-bit signed (Q4.60), add coefficient
// shifted left by 30, round-to-nearest (add 2^29), arithmetic right-shift 30
// → 32-bit signed Q2.30.
//
// Pipeline: 5 stages (one Horner FMA per stage), 1 elt/cycle throughput,
// fixed 5-cycle latency. No backpressure (caller manages pacing).
//
// Coefficient quantization error in Q2.30 is bounded by 2^-30 per coeff;
// over the polynomial this accumulates to well under 2^-20. Round-to-nearest
// at each FMA prevents bias accumulation. Net polynomial error budget vs.
// true exp(r): ≈ 2^-18 worst-case (Taylor truncation 2^-19 + roundoff 2^-21
// + coeff quant 2^-20). Comfortable headroom against the 2^-16 target.

package saturnfu

import chisel3._
import chisel3.util._

object PolyExpQ2_30Consts {
  // Coefficients = round(c_k * 2^30) for c_k = 1/k!.
  //   2^30 = 1073741824
  //     c0 = 1.0           → 1073741824 = 0x40000000
  //     c1 = 1.0           → 1073741824 = 0x40000000
  //     c2 = 1/2           →  536870912 = 0x20000000
  //     c3 = 1/6           →  178956971 = 0x0AAAAAAB  (round-up of 178956970.667)
  //     c4 = 1/24          →   44739243 = 0x02AAAAAB
  //     c5 = 1/120         →    8947849 = 0x00888889  (round-up of 8947848.533)
  val C0: Long = 0x40000000L
  val C1: Long = 0x40000000L
  val C2: Long = 0x20000000L
  val C3: Long = 0x0AAAAAABL
  val C4: Long = 0x02AAAAABL
  val C5: Long = 0x00888889L

  val FRAC_BITS: Int = 30
  val ROUND_BIAS: Long = 1L << (FRAC_BITS - 1)  // 2^29 for round-to-nearest

  /** Reference Scala-double evaluator using the SAME quantized coefficients,
    * for unit-test cross-checking. */
  def evalDouble(r: Double): Double = {
    val c0 = C0.toDouble / (1L << FRAC_BITS)
    val c1 = C1.toDouble / (1L << FRAC_BITS)
    val c2 = C2.toDouble / (1L << FRAC_BITS)
    val c3 = C3.toDouble / (1L << FRAC_BITS)
    val c4 = C4.toDouble / (1L << FRAC_BITS)
    val c5 = C5.toDouble / (1L << FRAC_BITS)
    var y = c5
    y = y * r + c4
    y = y * r + c3
    y = y * r + c2
    y = y * r + c1
    y = y * r + c0
    y
  }
}

class PolyExpQ2_30 extends Module {
  val io = IO(new Bundle {
    val in_valid  = Input(Bool())
    val r         = Input(SInt(32.W))
    val out_valid = Output(Bool())
    val y         = Output(SInt(32.W))
  })

  import PolyExpQ2_30Consts._

  // Wrap each constant as a 32-bit SInt literal.
  val c0 = C0.S(32.W)
  val c1 = C1.S(32.W)
  val c2 = C2.S(32.W)
  val c3 = C3.S(32.W)
  val c4 = C4.S(32.W)
  val c5 = C5.S(32.W)

  /** One Horner FMA in Q2.30: returns (t * r + c) in Q2.30. Round-to-nearest. */
  def horner(t: SInt, r: SInt, c: SInt): SInt = {
    val mul   = t * r                                // SInt(64.W), Q4.60
    val rnd   = mul +& ROUND_BIAS.S(64.W)            // +& keeps width safe
    val q230w = (rnd >> FRAC_BITS).asSInt            // SInt, value fits in 34 bits
    val q230  = q230w(31, 0).asSInt                  // truncate to 32 bits (safe for our range)
    q230 +& c                                        // result widened by 1, then truncate
  }

  // 5-stage pipeline. Each stage carries (r, t, valid).
  // After stage k (k=0..4), t_stage(k) holds the Horner result through c_{4-k}.
  val r_stage = Reg(Vec(5, SInt(32.W)))
  val t_stage = Reg(Vec(5, SInt(32.W)))
  val v_stage = RegInit(VecInit(Seq.fill(5)(false.B)))

  // Stage 0: t1 = c5 * r + c4
  v_stage(0) := io.in_valid
  r_stage(0) := io.r
  t_stage(0) := horner(c5, io.r, c4)(31, 0).asSInt   // FMA output may be 33-bit; truncate

  // Stages 1..4
  for (k <- 1 until 5) {
    v_stage(k) := v_stage(k - 1)
    r_stage(k) := r_stage(k - 1)
  }
  // Stage 1: t2 = t1 * r + c3
  t_stage(1) := horner(t_stage(0), r_stage(0), c3)(31, 0).asSInt
  // Stage 2: t3 = t2 * r + c2
  t_stage(2) := horner(t_stage(1), r_stage(1), c2)(31, 0).asSInt
  // Stage 3: t4 = t3 * r + c1
  t_stage(3) := horner(t_stage(2), r_stage(2), c1)(31, 0).asSInt
  // Stage 4: t5 = t4 * r + c0  (final result)
  t_stage(4) := horner(t_stage(3), r_stage(3), c0)(31, 0).asSInt

  io.out_valid := v_stage(4)
  io.y         := t_stage(4)
}
