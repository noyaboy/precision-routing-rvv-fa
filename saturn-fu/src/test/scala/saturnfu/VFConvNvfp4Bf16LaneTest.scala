// SPDX-License-Identifier: Apache-2.0
//
// Tests for VFConvNvfp4Bf16Lane (Track F, Mo 3 follow-up).
//
// Latency: 3 cycles.  Correctness target: bit-exact vs FP64-and-round-to-BF16
// golden across all 256 x 16 = 4096 (scale, element) combinations, except
// NaN-scale outputs where the golden encodes a sign-preserving QNaN.

package saturnfu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

object VFConvNvfp4Bf16TestUtils {
  val LATENCY = 3

  // ----- Reference decoders (Scala double precision) ------------------------

  /** Decode 4-bit NVFP4 (E2M1, OCP spec) to a Double.
    * Codepoints: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}. No NaN, no Inf. */
  def nvfp4ToDouble(code: Int): Double = {
    val sign = if ((code & 0x8) != 0) -1.0 else 1.0
    val exp  = (code >> 1) & 0x3
    val mant = code & 0x1
    val mag = if (exp == 0) {
      if (mant == 0) 0.0 else 0.5            // subnormal: only ±0.5
    } else {
      (1.0 + 0.5 * mant) * math.pow(2.0, exp - 1)
    }
    sign * mag
  }

  /** Decode 8-bit E4M3 (OCP FN variant, bias 7) to a Double or NaN.
    * exp=15 & m=7 is the only NaN encoding; all other exp=15 values are
    * normals with the usual (1 + m/8) * 2^(exp-7) formula. */
  def e4m3ToDouble(code: Int): Double = {
    val sign = if ((code & 0x80) != 0) -1.0 else 1.0
    val exp  = (code >> 3) & 0xF
    val mant = code & 0x7
    if (exp == 15 && mant == 7) Double.NaN
    else if (exp == 0) {
      if (mant == 0) 0.0
      else sign * (mant.toDouble / 8.0) * math.pow(2.0, -6)
    } else {
      sign * (1.0 + mant.toDouble / 8.0) * math.pow(2.0, exp - 7)
    }
  }

  /** Round-to-nearest-even from FP32 bits to BF16 (16-bit) bits. */
  def floatToBf16Bits(f: Float): Int = {
    if (java.lang.Float.isNaN(f)) {
      val bits = java.lang.Float.floatToRawIntBits(f)
      val sign = (bits >>> 31) & 1
      (sign << 15) | 0x7FC0
    } else {
      val bits  = java.lang.Float.floatToRawIntBits(f).toLong & 0xFFFFFFFFL
      val lower = bits & 0xFFFFL
      val upper = (bits >>> 16) & 0xFFFFL
      val rne = if (lower > 0x8000L) 1L
                else if (lower < 0x8000L) 0L
                else upper & 1L              // tie -> round to even
      ((upper + rne) & 0xFFFFL).toInt
    }
  }

  /** Hardware-matching golden output for (elt, scale).
    * Special-case path is identical to the DUT:
    *   - scale NaN  -> QNaN with sign = elt.sign XOR scale.sign
    *   - elt zero or scale zero -> signed zero with sign = XOR */
  def golden(eltCode: Int, scaleCode: Int): Int = {
    val eltSign   = (eltCode >> 3) & 0x1
    val scaleSign = (scaleCode >> 7) & 0x1
    val outSign   = eltSign ^ scaleSign

    val scaleExp  = (scaleCode >> 3) & 0xF
    val scaleMant = scaleCode & 0x7
    val scaleIsNan  = (scaleExp == 0xF) && (scaleMant == 0x7)
    val scaleIsZero = (scaleExp == 0)   && (scaleMant == 0)
    val eltIsZero   = (eltCode & 0x7)   == 0

    if (scaleIsNan) {
      (outSign << 15) | 0x7FC0
    } else if (eltIsZero || scaleIsZero) {
      outSign << 15
    } else {
      val eltVal   = nvfp4ToDouble(eltCode)
      val scaleVal = e4m3ToDouble(scaleCode)
      val product  = eltVal * scaleVal
      floatToBf16Bits(product.toFloat)
    }
  }

  /** Convert raw BF16 bits to a Float for human-readable error reporting. */
  def bf16BitsToFloat(bits: Int): Float = {
    val bits32 = (bits & 0xFFFF) << 16
    java.lang.Float.intBitsToFloat(bits32)
  }

  /** Push one (elt, scale) through the lane; return the BigInt output. */
  def runSingle(dut: VFConvNvfp4Bf16Lane, eltCode: Int, scaleCode: Int): Option[BigInt] = {
    dut.io.out.ready.poke(true.B)
    dut.io.in.bits.elt.poke(eltCode.U)
    dut.io.in.bits.scale.poke(scaleCode.U)
    dut.io.in.valid.poke(true.B)
    dut.clock.step(1)
    dut.io.in.valid.poke(false.B)

    var got: Option[BigInt] = None
    for (_ <- 0 until (LATENCY + 4) if got.isEmpty) {
      if (dut.io.out.valid.peek().litToBoolean)
        got = Some(dut.io.out.bits.peek().litValue)
      dut.clock.step(1)
    }
    got
  }
}

class VFConvNvfp4Bf16LaneSpec extends AnyFlatSpec with ChiselScalatestTester {
  import VFConvNvfp4Bf16TestUtils._

  behavior of "VFConvNvfp4Bf16Lane"

  it should s"respect the $LATENCY-cycle latency for the first valid output" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      dut.io.in.bits.elt.poke(0x2.U)        // +1.0
      dut.io.in.bits.scale.poke(0x38.U)     // E4M3 +1.0 (exp=7, m=0)
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

  it should "emit BF16 +1.0 for NVFP4(+1.0) * E4M3(+1.0)" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, 0x2, 0x38).getOrElse(BigInt(-1))
      assert(got == BigInt(0x3F80),
        f"expected 0x3F80 (+1.0 BF16); got 0x${got.toLong}%04X")
    }
  }

  it should "emit BF16 +6.0 for NVFP4(+6.0) * E4M3(+1.0)" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, 0x7, 0x38).getOrElse(BigInt(-1))
      assert(got == BigInt(0x40C0),
        f"expected 0x40C0 (+6.0 BF16); got 0x${got.toLong}%04X")
    }
  }

  it should "emit BF16 -3.0 for NVFP4(-3.0) * E4M3(+1.0)" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, 0xD, 0x38).getOrElse(BigInt(-1))
      assert(got == BigInt(0xC040),
        f"expected 0xC040 (-3.0 BF16); got 0x${got.toLong}%04X")
    }
  }

  it should "emit signed zero when element is +0.0 (NVFP4 0x0)" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, 0x0, 0x40).getOrElse(BigInt(-1))   // 0 * 2.0
      assert(got == BigInt(0x0000), f"expected 0x0000; got 0x${got.toLong}%04X")
    }
  }

  it should "emit -0.0 when element is +0.0 but scale is negative" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, 0x0, 0xC0).getOrElse(BigInt(-1))   // +0 * -2.0
      assert(got == BigInt(0x8000), f"expected 0x8000 (-0.0); got 0x${got.toLong}%04X")
    }
  }

  it should "emit signed zero when scale is +0.0" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, 0x7, 0x00).getOrElse(BigInt(-1))   // 6.0 * 0
      assert(got == BigInt(0x0000), f"expected 0x0000; got 0x${got.toLong}%04X")
    }
  }

  it should "propagate sign-preserving QNaN for NaN scale (exp=15 & m=7)" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, 0x2, 0x7F).getOrElse(BigInt(-1))   // +1.0 * NaN(+)
      assert(got == BigInt(0x7FC0),
        f"expected 0x7FC0 (+QNaN); got 0x${got.toLong}%04X")
    }
  }

  it should "propagate negative-sign QNaN for negative NaN scale" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val got = runSingle(dut, 0x2, 0xFF).getOrElse(BigInt(-1))   // +1.0 * NaN(-)
      assert(got == BigInt(0xFFC0),
        f"expected 0xFFC0 (-QNaN); got 0x${got.toLong}%04X")
    }
  }

  it should "handle subnormal E4M3 scales correctly" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      // E4M3 subnormal: exp=0, m=4 -> value = 4/8 * 2^-6 = 2^-7
      // NVFP4 +1.0 * 2^-7 = 2^-7 ≈ 0.0078125
      // BF16 of 2^-7: exp = 127-7 = 120 = 0x78, mantissa = 0 -> 0x3C00
      val got = runSingle(dut, 0x2, 0x04).getOrElse(BigInt(-1))
      assert(got == BigInt(0x3C00),
        f"expected 0x3C00 (2^-7 BF16); got 0x${got.toLong}%04X")
    }
  }

  it should "stream a sequence of inputs with 1-elt/cycle throughput" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      // Test pattern: every NVFP4 codepoint x E4M3 scale 0x38 (+1.0)
      val inputs = (0 until 16).map(c => (c, 0x38))

      val outputs = scala.collection.mutable.ArrayBuffer.empty[BigInt]
      var inIdx = 0
      var cyc = 0
      val maxCyc = inputs.length + LATENCY + 4
      while (outputs.length < inputs.length && cyc < maxCyc) {
        if (inIdx < inputs.length) {
          dut.io.in.bits.elt.poke(inputs(inIdx)._1.U)
          dut.io.in.bits.scale.poke(inputs(inIdx)._2.U)
          dut.io.in.valid.poke(true.B)
        } else {
          dut.io.in.valid.poke(false.B)
        }
        if (dut.io.out.valid.peek().litToBoolean)
          outputs += dut.io.out.bits.peek().litValue
        if (inIdx < inputs.length && dut.io.in.ready.peek().litToBoolean) inIdx += 1
        dut.clock.step(1)
        cyc += 1
      }

      assert(outputs.length == inputs.length,
        s"streamed ${outputs.length} outputs, expected ${inputs.length}")
      // Expected: BF16 of nvfp4 element value (since scale is +1.0).
      val expected = VFConvNvfp4Bf16Consts.E2M1_TO_BF16
      for (i <- inputs.indices) {
        assert(outputs(i) == BigInt(expected(i)),
          f"i=$i%d: got 0x${outputs(i).toLong}%04X, expected 0x${expected(i)}%04X")
      }
    }
  }

  it should "match the FP64-and-BF16-round golden bit-exactly across all 4096 (elt, scale) combinations" in {
    test(new VFConvNvfp4Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      // Generate every combination of 4-bit element x 8-bit scale.
      val inputs = (for {
        sc <- 0 until 256
        e  <- 0 until 16
      } yield (e, sc)).toSeq

      val outputs = scala.collection.mutable.ArrayBuffer.empty[BigInt]
      var inIdx = 0
      var cyc = 0
      val maxCyc = inputs.length + LATENCY + 16
      while (outputs.length < inputs.length && cyc < maxCyc) {
        if (inIdx < inputs.length) {
          dut.io.in.bits.elt.poke(inputs(inIdx)._1.U)
          dut.io.in.bits.scale.poke(inputs(inIdx)._2.U)
          dut.io.in.valid.poke(true.B)
        } else {
          dut.io.in.valid.poke(false.B)
        }
        if (dut.io.out.valid.peek().litToBoolean)
          outputs += dut.io.out.bits.peek().litValue
        if (inIdx < inputs.length && dut.io.in.ready.peek().litToBoolean) inIdx += 1
        dut.clock.step(1)
        cyc += 1
      }

      assert(outputs.length == inputs.length,
        s"streamed ${outputs.length} outputs, expected ${inputs.length}")

      var mismatches = 0
      var firstMismatch: Option[(Int, Int, Int, Int)] = None
      for (((e, sc), out) <- inputs.zip(outputs)) {
        val expected = golden(e, sc)
        if (out.toInt != expected) {
          mismatches += 1
          if (firstMismatch.isEmpty) firstMismatch = Some((e, sc, out.toInt, expected))
        }
      }
      val (e0, sc0, got0, exp0) = firstMismatch.getOrElse((0, 0, 0, 0))
      info(f"exhaustive sweep: ${inputs.length}%d cases, $mismatches%d mismatches")
      assert(mismatches == 0,
        f"$mismatches%d / ${inputs.length}%d mismatches; first at elt=0x$e0%X scale=0x$sc0%02X " +
        f"got=0x$got0%04X expected=0x$exp0%04X " +
        f"(elt=${nvfp4ToDouble(e0)}%.4f scale=${e4m3ToDouble(sc0)}%.4e product=${nvfp4ToDouble(e0) * e4m3ToDouble(sc0)}%.4e)")
    }
  }
}
