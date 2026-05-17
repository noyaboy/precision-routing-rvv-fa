// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for PolyExpQ2_30 — verifies the degree-5 Horner evaluator hits
// the Mo-3 numerical-accuracy target (< 2^-16 max rel err) against
// scala.math.exp on r ∈ [-ln(2)/2, +ln(2)/2].

package saturnfu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

object Q230Utils {
  val ONE: Long = 1L << 30  // 2^30

  /** Encode a real value v ∈ [-2, 2) as a 32-bit signed Q2.30 BigInt. */
  def encode(v: Double): BigInt = {
    val scaled = math.round(v * ONE.toDouble).toLong
    // Saturate just to be safe.
    val sat = math.max(math.min(scaled, (1L << 31) - 1), -(1L << 31))
    BigInt(sat)
  }

  /** Decode a 32-bit BigInt (treated as signed Q2.30) back to a Double. */
  def decode(b: BigInt): Double = {
    // Treat as signed 32-bit
    val masked = b & ((BigInt(1) << 32) - 1)
    val signed = if (masked >= (BigInt(1) << 31)) masked - (BigInt(1) << 32) else masked
    signed.toDouble / ONE.toDouble
  }
}

class PolyExpQ2_30Spec extends AnyFlatSpec with ChiselScalatestTester {
  import Q230Utils._

  behavior of "PolyExpQ2_30"

  it should "evaluate exp(0) = 1 exactly (within ULP)" in {
    test(new PolyExpQ2_30) { dut =>
      dut.io.in_valid.poke(true.B)
      dut.io.r.poke(encode(0.0).S(32.W))
      dut.clock.step(1)
      dut.io.in_valid.poke(false.B)

      // 5-cycle latency
      var got: Option[BigInt] = None
      for (_ <- 0 until 10 if got.isEmpty) {
        if (dut.io.out_valid.peek().litToBoolean)
          got = Some(dut.io.y.peek().litValue)
        dut.clock.step(1)
      }
      assert(got.isDefined, "no output within 10 cycles")
      val gotD = decode(got.get)
      val err = math.abs(gotD - 1.0)
      assert(err < math.pow(2.0, -20),
        s"exp(0) should be 1.0; got $gotD (abs err $err)")
    }
  }

  it should "respect 5-cycle latency for the first valid output" in {
    test(new PolyExpQ2_30) { dut =>
      dut.io.in_valid.poke(true.B)
      dut.io.r.poke(encode(0.1).S(32.W))
      // Sample output starting next cycle.
      var latency = -1
      for (cyc <- 1 to 10 if latency < 0) {
        dut.clock.step(1)
        if (dut.io.out_valid.peek().litToBoolean) latency = cyc
        dut.io.in_valid.poke(false.B)  // only first input is valid
      }
      assert(latency == 5, s"expected 5-cycle latency, got $latency")
    }
  }

  it should "stream a sequence of inputs with one elt/cycle throughput" in {
    test(new PolyExpQ2_30) { dut =>
      val inputs = Seq(-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)

      // Issue: feed input every cycle, sample output every cycle starting cycle 5.
      val outputs = scala.collection.mutable.ArrayBuffer.empty[Double]
      val totalCycles = inputs.length + 10
      var inIdx = 0

      for (cyc <- 0 until totalCycles) {
        if (inIdx < inputs.length) {
          dut.io.in_valid.poke(true.B)
          dut.io.r.poke(encode(inputs(inIdx)).S(32.W))
          inIdx += 1
        } else {
          dut.io.in_valid.poke(false.B)
        }
        if (dut.io.out_valid.peek().litToBoolean) {
          outputs += decode(dut.io.y.peek().litValue)
        }
        dut.clock.step(1)
      }

      assert(outputs.length == inputs.length,
        s"expected ${inputs.length} outputs, got ${outputs.length}")

      // Each output should match math.exp(input) to < 2^-16.
      val threshold = math.pow(2.0, -16)
      for ((in, out) <- inputs.zip(outputs)) {
        val expected = math.exp(in)
        val absErr = math.abs(out - expected)
        val relErr = absErr / math.abs(expected)
        assert(relErr < threshold,
          f"r=$in%.3f exp(r)=$expected%.6f got=$out%.6f relErr=$relErr%.3e " +
          f"(threshold ${threshold}%.3e)")
      }
    }
  }

  it should "stay within 2^-16 max rel err over a dense sweep of [-ln(2)/2, +ln(2)/2]" in {
    test(new PolyExpQ2_30) { dut =>
      val ln2half = math.log(2.0) / 2.0
      // 257 points across the range
      val nPts = 257
      val inputs = (0 until nPts).map(i =>
        -ln2half + (2 * ln2half) * i.toDouble / (nPts - 1)
      )

      val outputs = scala.collection.mutable.ArrayBuffer.empty[Double]
      val totalCycles = inputs.length + 12
      var inIdx = 0
      for (cyc <- 0 until totalCycles) {
        if (inIdx < inputs.length) {
          dut.io.in_valid.poke(true.B)
          dut.io.r.poke(encode(inputs(inIdx)).S(32.W))
          inIdx += 1
        } else {
          dut.io.in_valid.poke(false.B)
        }
        if (dut.io.out_valid.peek().litToBoolean) {
          outputs += decode(dut.io.y.peek().litValue)
        }
        dut.clock.step(1)
      }
      assert(outputs.length == inputs.length,
        s"streamed ${outputs.length} outputs, expected ${inputs.length}")

      val threshold = math.pow(2.0, -16)
      var maxRelErr = 0.0
      var maxAt: Double = Double.NaN
      for ((in, out) <- inputs.zip(outputs)) {
        val expected = math.exp(in)
        val relErr = math.abs(out - expected) / math.abs(expected)
        if (relErr > maxRelErr) { maxRelErr = relErr; maxAt = in }
      }
      info(f"max rel err = $maxRelErr%.4e at r = $maxAt%.4f (threshold ${threshold}%.3e)")
      assert(maxRelErr < threshold,
        f"max rel err $maxRelErr%.4e exceeds threshold ${threshold}%.4e (at r=$maxAt%.4f)")
    }
  }
}
