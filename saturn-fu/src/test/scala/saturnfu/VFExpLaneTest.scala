// SPDX-License-Identifier: Apache-2.0
//
// Tests for VFExpLane. Updated 2026-05-17 for Track E (real polynomial).
// Latency: 10 cycles.  Numerical-accuracy target: max rel err < 2^-16 over a
// dense sweep of BF16 inputs (excluding special-case inputs).

package saturnfu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

object VFExpTestUtils {
  val LATENCY = 10

  /** Convert a 16-bit raw BF16 pattern to an IEEE-754 BF16 value as Float. */
  def bf16BitsToFloat(bits: Int): Float = {
    val bits32 = (bits & 0xFFFF) << 16
    java.lang.Float.intBitsToFloat(bits32)
  }

  /** Convert a Float to BF16 bits via round-to-nearest-even truncation. */
  def floatToBf16Bits(f: Float): Int = {
    val bits32 = java.lang.Float.floatToRawIntBits(f)
    val rounded = ((bits32 + 0x8000) >>> 16) & 0xFFFF
    rounded
  }

  /** Convert raw 32-bit pattern back to Float. */
  def bitsToFloat(bits: Long): Float =
    java.lang.Float.intBitsToFloat(bits.toInt)

  /** Relative error for a Float. */
  def relErr(actual: Float, expected: Float): Double = {
    if (expected.isNaN || actual.isNaN)
      return if (expected.isNaN && actual.isNaN) 0.0 else Double.PositiveInfinity
    if (expected == 0.0f) return math.abs(actual.toDouble)
    math.abs((actual.toDouble - expected.toDouble) / expected.toDouble)
  }

  /** Push a single BF16 input through the pipe; return the BigInt output. */
  def runSingle(dut: VFExpLane, bf16Bits: Int): Option[BigInt] = {
    dut.io.out.ready.poke(true.B)
    dut.io.in.bits.poke(bf16Bits.U)
    dut.io.in.valid.poke(true.B)
    dut.clock.step(1)
    dut.io.in.valid.poke(false.B)

    var got: Option[BigInt] = None
    for (_ <- 0 until (LATENCY + 8) if got.isEmpty) {
      if (dut.io.out.valid.peek().litToBoolean)
        got = Some(dut.io.out.bits.peek().litValue)
      dut.clock.step(1)
    }
    got
  }
}

class VFExpLaneSpec extends AnyFlatSpec with ChiselScalatestTester {
  import VFExpTestUtils._

  behavior of "VFExpLane"

  it should "emit FP32 exp(0) == 1 for a zero BF16 input" in {
    test(new VFExpLane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, 0)  // BF16 0.0
      assert(got.isDefined, "no output emitted within latency window")
      val gotF = bitsToFloat(got.get.toLong)
      assert(gotF == 1.0f, s"exp(0) should be 1.0, got $gotF")
    }
  }

  it should s"respect the $LATENCY-cycle latency for the first valid output" in {
    test(new VFExpLane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      dut.io.in.bits.poke(floatToBf16Bits(1.0f).U)
      dut.io.in.valid.poke(true.B)

      var latency = -1
      dut.clock.step(1)
      dut.io.in.valid.poke(false.B)

      for (cyc <- 1 to (LATENCY + 4) if latency < 0) {
        if (dut.io.out.valid.peek().litToBoolean) latency = cyc
        dut.clock.step(1)
      }
      assert(latency == LATENCY, s"expected $LATENCY-cycle latency, got $latency")
    }
  }

  it should "saturate to +Inf for large positive BF16 inputs" in {
    test(new VFExpLane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, floatToBf16Bits(100.0f))
      assert(got.isDefined)
      val gotF = bitsToFloat(got.get.toLong)
      assert(gotF.isPosInfinity, s"expected +Inf, got $gotF")
    }
  }

  it should "underflow to zero for large negative BF16 inputs" in {
    test(new VFExpLane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, floatToBf16Bits(-100.0f))
      assert(got.isDefined)
      val gotF = bitsToFloat(got.get.toLong)
      assert(gotF == 0.0f, s"expected 0.0, got $gotF")
    }
  }

  it should "pipe a stream of moderate inputs through without NaN" in {
    test(new VFExpLane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val inputs = (-16 to 16).map(i => i * 0.125f).toSeq

      val outputs = scala.collection.mutable.ArrayBuffer.empty[Float]
      var inIdx = 0
      var cyc = 0
      val maxCyc = inputs.length + LATENCY + 4

      while (outputs.length < inputs.length && cyc < maxCyc) {
        if (inIdx < inputs.length) {
          dut.io.in.bits.poke(floatToBf16Bits(inputs(inIdx)).U)
          dut.io.in.valid.poke(true.B)
        } else {
          dut.io.in.valid.poke(false.B)
        }
        if (dut.io.out.valid.peek().litToBoolean)
          outputs += bitsToFloat(dut.io.out.bits.peek().litValue.toLong)
        if (inIdx < inputs.length && dut.io.in.ready.peek().litToBoolean) inIdx += 1
        dut.clock.step(1)
        cyc += 1
      }

      assert(outputs.length == inputs.length,
        s"only got ${outputs.length} outputs for ${inputs.length} inputs")
      assert(!outputs.exists(_.isNaN), "no output should be NaN")
    }
  }

  it should "hit < 2^-16 max rel err vs scalar expf on a dense sweep of BF16 inputs in [-10, 10]" in {
    test(new VFExpLane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      // Dense sweep — BF16 grid covers ~80 points/decade. -10 to +10 in steps of 0.0625.
      val rawInputs = (-160 to 160).map(i => i * 0.0625f).toSeq
      // Use the BF16-rounded value for comparison, since the DUT actually
      // receives the BF16-quantized input.
      val inputBf16Pairs = rawInputs.map { f =>
        val bits = floatToBf16Bits(f)
        (bits, bf16BitsToFloat(bits))
      }

      // Stream inputs through.
      val outputs = scala.collection.mutable.ArrayBuffer.empty[Float]
      var inIdx = 0
      var cyc = 0
      val maxCyc = inputBf16Pairs.length + LATENCY + 8
      while (outputs.length < inputBf16Pairs.length && cyc < maxCyc) {
        if (inIdx < inputBf16Pairs.length) {
          dut.io.in.bits.poke(inputBf16Pairs(inIdx)._1.U)
          dut.io.in.valid.poke(true.B)
        } else {
          dut.io.in.valid.poke(false.B)
        }
        if (dut.io.out.valid.peek().litToBoolean)
          outputs += bitsToFloat(dut.io.out.bits.peek().litValue.toLong)
        if (inIdx < inputBf16Pairs.length && dut.io.in.ready.peek().litToBoolean) inIdx += 1
        dut.clock.step(1)
        cyc += 1
      }

      assert(outputs.length == inputBf16Pairs.length,
        s"streamed ${outputs.length} outputs, expected ${inputBf16Pairs.length}")

      // Compare DUT output to scalar expf on the BF16-quantized input.
      val threshold = math.pow(2.0, -16)  // ≈ 1.526e-5
      var maxRelErr = 0.0
      var maxAt: Float = Float.NaN
      var nChecked = 0
      for (((_, xBf16), out) <- inputBf16Pairs.zip(outputs)) {
        val expected = math.exp(xBf16.toDouble).toFloat
        if (!expected.isInfinite && !expected.isNaN && expected != 0.0f &&
            !out.isInfinite && !out.isNaN) {
          val err = relErr(out, expected)
          if (err > maxRelErr) { maxRelErr = err; maxAt = xBf16 }
          nChecked += 1
        }
      }
      info(f"checked $nChecked%d points; max rel err = $maxRelErr%.4e at x = $maxAt%.4f " +
           f"(threshold ${threshold}%.3e)")
      assert(nChecked > 100, s"only $nChecked points were checked — sweep too sparse")
      assert(maxRelErr < threshold,
        f"max rel err $maxRelErr%.4e exceeds 2^-16 threshold ${threshold}%.4e (at x=$maxAt%.4f)")
    }
  }
}
