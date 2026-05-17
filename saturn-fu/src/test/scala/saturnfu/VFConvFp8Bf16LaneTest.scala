// SPDX-License-Identifier: Apache-2.0
//
// Tests for VFConvFp8Bf16Lane (Track F3).
//
// Latency: 1 cycle.  Correctness target: bit-exact vs the
// e4m3ToDouble-then-BF16-round golden across all 256 FP8 inputs.  Includes
// the round-trip inverse of VFConvBf16Fp8Lane on round-trippable values.

package saturnfu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

object VFConvFp8Bf16TestUtils {
  val LATENCY = 1

  /** Same decoder used by F2's golden — kept in sync with the OCP-FN
    * convention also used by VFConvNvfp4Bf16Lane's scale decode. */
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

  /** Same RNE BF16 packer used by VFConvNvfp4Bf16LaneTest. */
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
                else upper & 1L
      ((upper + rne) & 0xFFFFL).toInt
    }
  }

  /** Golden: FP8-E4M3 (OCP FN) -> BF16 via Double round-trip, with the same
    * sign-preserving QNaN convention the DUT emits.  NaN gets special-cased
    * before the Double conversion to preserve the FP8 sign (Java's
    * Float.toFloat on Double.NaN may not). */
  def fp8E4M3FnToBf16(fp8: Int): Int = {
    val sign = (fp8 >> 7) & 1
    val exp  = (fp8 >> 3) & 0xF
    val mant = fp8 & 0x7

    if (exp == 0xF && mant == 7) return (sign << 15) | 0x7FC0          // ±NaN
    if (exp == 0  && mant == 0) return sign << 15                      // ±0

    // All other E4M3 values fit exactly in BF16; Double round-trip is exact.
    val v = e4m3ToDouble(fp8).toFloat
    floatToBf16Bits(v)
  }

  /** Convert raw BF16 bits to a Float for human-readable error reporting. */
  def bf16BitsToFloat(bits: Int): Float = {
    val bits32 = (bits & 0xFFFF) << 16
    java.lang.Float.intBitsToFloat(bits32)
  }

  /** Push one FP8 through the lane; return the BF16 output if produced. */
  def runSingle(dut: VFConvFp8Bf16Lane, fp8: Int): Option[BigInt] = {
    dut.io.out.ready.poke(true.B)
    dut.io.in.bits.poke(fp8.U)
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

class VFConvFp8Bf16LaneSpec extends AnyFlatSpec with ChiselScalatestTester {
  import VFConvFp8Bf16TestUtils._

  behavior of "VFConvFp8Bf16Lane"

  it should s"respect the $LATENCY-cycle latency for the first valid output" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      dut.io.in.bits.poke(0x38.U)             // FP8 +1.0
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

  it should "emit BF16 +1.0 (0x3F80) for FP8 +1.0 (0x38)" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x38).getOrElse(BigInt(-1))
      assert(got == BigInt(0x3F80), f"expected 0x3F80; got 0x${got.toLong}%04X")
    }
  }

  it should "emit BF16 -1.0 (0xBF80) for FP8 -1.0 (0xB8)" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0xB8).getOrElse(BigInt(-1))
      assert(got == BigInt(0xBF80), f"expected 0xBF80; got 0x${got.toLong}%04X")
    }
  }

  it should "emit BF16 +2.0 (0x4000) for FP8 +2.0 (0x40)" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x40).getOrElse(BigInt(-1))
      assert(got == BigInt(0x4000), f"expected 0x4000; got 0x${got.toLong}%04X")
    }
  }

  it should "emit BF16 +max (448 = 0x43E0) for FP8 +max (0x7E)" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x7E).getOrElse(BigInt(-1))
      assert(got == BigInt(0x43E0), f"expected 0x43E0; got 0x${got.toLong}%04X")
    }
  }

  it should "emit BF16 -max (-448 = 0xC3E0) for FP8 -max (0xFE)" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0xFE).getOrElse(BigInt(-1))
      assert(got == BigInt(0xC3E0), f"expected 0xC3E0; got 0x${got.toLong}%04X")
    }
  }

  it should "preserve signed zero (0x00 -> 0x0000, 0x80 -> 0x8000)" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val gotP = runSingle(dut, 0x00).getOrElse(BigInt(-1))
      assert(gotP == BigInt(0x0000), f"expected 0x0000; got 0x${gotP.toLong}%04X")
      val gotN = runSingle(dut, 0x80).getOrElse(BigInt(-1))
      assert(gotN == BigInt(0x8000), f"expected 0x8000; got 0x${gotN.toLong}%04X")
    }
  }

  it should "emit sign-preserving BF16 QNaN for FP8 NaN (0x7F -> 0x7FC0, 0xFF -> 0xFFC0)" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val gotP = runSingle(dut, 0x7F).getOrElse(BigInt(-1))
      assert(gotP == BigInt(0x7FC0), f"expected 0x7FC0; got 0x${gotP.toLong}%04X")
      val gotN = runSingle(dut, 0xFF).getOrElse(BigInt(-1))
      assert(gotN == BigInt(0xFFC0), f"expected 0xFFC0; got 0x${gotN.toLong}%04X")
    }
  }

  it should "decode subnormal FP8 0x01 (= 2^-9) to BF16 0x3B00" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x01).getOrElse(BigInt(-1))
      assert(got == BigInt(0x3B00), f"expected 0x3B00; got 0x${got.toLong}%04X")
    }
  }

  it should "decode subnormal FP8 0x04 (= 4*2^-9 = 2^-7) to BF16 0x3C00" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x04).getOrElse(BigInt(-1))
      assert(got == BigInt(0x3C00), f"expected 0x3C00; got 0x${got.toLong}%04X")
    }
  }

  it should "decode max-subnormal FP8 0x07 (= 7*2^-9) to BF16 0x3C60" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x07).getOrElse(BigInt(-1))
      assert(got == BigInt(0x3C60), f"expected 0x3C60; got 0x${got.toLong}%04X")
    }
  }

  it should "stream a sequence of inputs with 1-elt/cycle throughput" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val inputs: Seq[Int] = Seq(
        0x38, 0xB8, 0x40, 0xC0, 0x30, 0x04, 0x07, 0x01,
        0x7E, 0xFE, 0x00, 0x80, 0x7F, 0xFF, 0x02, 0x03,
      )
      val expected = inputs.map(fp8E4M3FnToBf16)

      val outputs = scala.collection.mutable.ArrayBuffer.empty[BigInt]
      var inIdx = 0
      var cyc = 0
      val maxCyc = inputs.length + LATENCY + 4
      while (outputs.length < inputs.length && cyc < maxCyc) {
        if (inIdx < inputs.length) {
          dut.io.in.bits.poke(inputs(inIdx).U)
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
      for (i <- inputs.indices) {
        assert(outputs(i) == BigInt(expected(i)),
          f"i=$i%d in=0x${inputs(i)}%02X: got 0x${outputs(i).toLong}%04X, " +
          f"expected 0x${expected(i)}%04X")
      }
    }
  }

  it should "match the e4m3-via-Double-to-BF16 golden bit-exactly across all 256 FP8 inputs" in {
    test(new VFConvFp8Bf16Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val inputs: Seq[Int] = 0 until 256
      val outputs = scala.collection.mutable.ArrayBuffer.empty[BigInt]
      var inIdx = 0
      var cyc = 0
      val maxCyc = inputs.length + LATENCY + 16
      while (outputs.length < inputs.length && cyc < maxCyc) {
        if (inIdx < inputs.length) {
          dut.io.in.bits.poke(inputs(inIdx).U)
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
      var firstMismatch: Option[(Int, Int, Int)] = None
      for ((fp8, out) <- inputs.zip(outputs)) {
        val expected = fp8E4M3FnToBf16(fp8)
        if (out.toInt != expected) {
          mismatches += 1
          if (firstMismatch.isEmpty) firstMismatch = Some((fp8, out.toInt, expected))
        }
      }
      val (b0, g0, e0) = firstMismatch.getOrElse((0, 0, 0))
      info(f"exhaustive sweep: ${inputs.length}%d cases, $mismatches%d mismatches")
      assert(mismatches == 0,
        f"$mismatches%d / ${inputs.length}%d mismatches; first at FP8=0x$b0%02X " +
        f"got=0x$g0%04X expected=0x$e0%04X")
    }
  }

  it should "be the inverse of VFConvBf16Fp8Lane on round-trippable FP8 inputs" in {
    // Compose Track F3 and Track F2 in software: FP8 -> BF16 (via F3 golden),
    // then BF16 -> FP8 (via F2 golden).  Round-trip on every byte except NaN
    // should return the original byte (zero patterns return canonical 0x00 /
    // 0x80 if the input was 0x00 / 0x80; alternative-zero patterns don't exist
    // in E4M3 FN).  NaN patterns may round-trip to the canonical NaN encoding.
    import VFConvBf16Fp8TestUtils.bf16ToFp8E4M3Fn

    var mismatches = 0
    var firstMismatch: Option[(Int, Int)] = None
    for (fp8 <- 0 until 256) {
      val bf16  = fp8E4M3FnToBf16(fp8)
      val round = bf16ToFp8E4M3Fn(bf16)
      // NaN inputs round-trip to NaN encoding (0x7F/0xFF) regardless of sign.
      val ok = (round == fp8)
      if (!ok) {
        mismatches += 1
        if (firstMismatch.isEmpty) firstMismatch = Some((fp8, round))
      }
    }
    val (in0, out0) = firstMismatch.getOrElse((0, 0))
    info(f"round-trip: 256 cases, $mismatches%d non-identity")
    assert(mismatches == 0,
      f"$mismatches%d round-trip drifts; first FP8=0x$in0%02X -> BF16 -> 0x$out0%02X")
  }
}
