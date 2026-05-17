// SPDX-License-Identifier: Apache-2.0
//
// Tests for VFConvBf16Fp8Lane (Track F2, follow-up to Track F).
//
// Latency: 2 cycles.  Correctness target: bit-exact vs the FP-RNE golden
// (distance-based nearest E4M3 representable, banker's tie-break) across all
// 65536 BF16 inputs.  Includes BF16 NaN / Inf / zero / subnormal handling and
// the E4M3-FN saturation-at-NaN-encoding boundary.

package saturnfu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

object VFConvBf16Fp8TestUtils {
  val LATENCY = 2

  /** Decode 8-bit E4M3 (OCP FN variant, bias 7) to a Double or NaN.
    * Same convention as VFConvNvfp4Bf16TestUtils.e4m3ToDouble — exp=15 & m=7
    * is the only NaN, all other exp=15 patterns are normal. */
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

  /** Decode raw BF16 bits to a Float. */
  def bf16BitsToFloat(bits: Int): Float = {
    val bits32 = (bits & 0xFFFF) << 16
    java.lang.Float.intBitsToFloat(bits32)
  }

  /** Sorted positive-side E4M3 representables (byte, value).  Excludes the
    * single NaN encoding (0x7F).  Used by the distance-based golden. */
  val POS_REPS: Array[(Int, Double)] = {
    (0 to 0x7E).map(b => (b, e4m3ToDouble(b)))
               .toArray
               .sortBy(_._2)
  }

  /** Distance-based RNE golden: for a BF16 input, return the bit-exact E4M3-FN
    * byte after RNE rounding with banker's tie-break and saturation at the
    * NaN-encoding boundary.  Independent of the DUT's bit-level shift logic. */
  def bf16ToFp8E4M3Fn(bits: Int): Int = {
    val sign  = (bits >> 15) & 1
    val bexp  = (bits >> 7)  & 0xFF
    val bmant = bits & 0x7F

    if (bexp == 0xFF) {
      // BF16 Inf -> saturate to ±max; BF16 NaN -> sign-preserving NaN.
      return if (bmant == 0) (sign << 7) | 0x7E
             else            (sign << 7) | 0x7F
    }
    if (bexp == 0) {
      // BF16 zero or subnormal (< 2^-126) -> underflows to ±0.
      return sign << 7
    }

    val v    = bf16BitsToFloat(bits).toDouble
    val absV = math.abs(v)
    val maxNormal = POS_REPS.last._2   // 448.0

    // Saturation: any value strictly above max-normal saturates.  Exactly
    // max-normal hits the distance-based branch below and resolves to 0x7E.
    if (absV > maxNormal) {
      return (sign << 7) | 0x7E
    }

    // Find the nearest representable in POS_REPS to absV.  Binary-search for
    // the smallest index with value >= absV, then compare against index-1.
    var lo = 0
    var hi = POS_REPS.length - 1
    while (lo < hi) {
      val mid = (lo + hi) / 2
      if (POS_REPS(mid)._2 < absV) lo = mid + 1 else hi = mid
    }
    val (bHi, vHi) = POS_REPS(lo)
    val (bLo, vLo) = if (lo == 0) POS_REPS(0) else POS_REPS(lo - 1)

    val dHi = vHi - absV
    val dLo = absV - vLo
    val chosenByte =
      if (lo == 0)        bHi               // absV == vLo == 0
      else if (dLo < dHi) bLo
      else if (dHi < dLo) bHi
      else {
        // RNE tie -> even byte LSB (= even E4M3 mantissa LSB).
        if ((bLo & 1) == 0) bLo else bHi
      }

    // For ±0, drop the sign-only-byte zero pattern and just emit sign << 7.
    if (chosenByte == 0) sign << 7
    else                 (sign << 7) | chosenByte
  }

  /** Push one BF16 through the lane; return the byte output if produced. */
  def runSingle(dut: VFConvBf16Fp8Lane, bf16: Int): Option[BigInt] = {
    dut.io.out.ready.poke(true.B)
    dut.io.in.bits.poke(bf16.U)
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

class VFConvBf16Fp8LaneSpec extends AnyFlatSpec with ChiselScalatestTester {
  import VFConvBf16Fp8TestUtils._

  behavior of "VFConvBf16Fp8Lane"

  it should s"respect the $LATENCY-cycle latency for the first valid output" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      dut.io.in.bits.poke(0x3F80.U)         // BF16 +1.0
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

  it should "emit FP8 +1.0 (0x38) for BF16 +1.0 (0x3F80)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x3F80).getOrElse(BigInt(-1))
      assert(got == BigInt(0x38), f"expected 0x38; got 0x${got.toLong}%02X")
    }
  }

  it should "emit FP8 -1.0 (0xB8) for BF16 -1.0 (0xBF80)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0xBF80).getOrElse(BigInt(-1))
      assert(got == BigInt(0xB8), f"expected 0xB8; got 0x${got.toLong}%02X")
    }
  }

  it should "emit FP8 +0.5 (0x30) for BF16 +0.5 (0x3F00)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x3F00).getOrElse(BigInt(-1))
      assert(got == BigInt(0x30), f"expected 0x30; got 0x${got.toLong}%02X")
    }
  }

  it should "emit FP8 +2.0 (0x40) for BF16 +2.0 (0x4000)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x4000).getOrElse(BigInt(-1))
      assert(got == BigInt(0x40), f"expected 0x40; got 0x${got.toLong}%02X")
    }
  }

  it should "preserve signed +0 (0x3F80 sign flip etc.)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val gotP = runSingle(dut, 0x0000).getOrElse(BigInt(-1))
      assert(gotP == BigInt(0x00), f"expected 0x00 for +0; got 0x${gotP.toLong}%02X")
      val gotN = runSingle(dut, 0x8000).getOrElse(BigInt(-1))
      assert(gotN == BigInt(0x80), f"expected 0x80 for -0; got 0x${gotN.toLong}%02X")
    }
  }

  it should "emit FP8 +max (0x7E) for BF16 +max-normal (448 = 0x43E0)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      // BF16 of 448: bexp = 8 + 127 = 135 = 0x87; mantissa 1.110 -> bmant = 0x60
      // packed: 0_10000111_1100000 = 0x43E0
      val got = runSingle(dut, 0x43E0).getOrElse(BigInt(-1))
      assert(got == BigInt(0x7E), f"expected 0x7E; got 0x${got.toLong}%02X")
    }
  }

  it should "saturate BF16 +Inf to FP8 +max (0x7E)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x7F80).getOrElse(BigInt(-1))
      assert(got == BigInt(0x7E), f"expected 0x7E; got 0x${got.toLong}%02X")
    }
  }

  it should "saturate BF16 -Inf to FP8 -max (0xFE)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0xFF80).getOrElse(BigInt(-1))
      assert(got == BigInt(0xFE), f"expected 0xFE; got 0x${got.toLong}%02X")
    }
  }

  it should "pass BF16 +NaN through as FP8 +NaN (0x7F)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0x7FC0).getOrElse(BigInt(-1))
      assert(got == BigInt(0x7F), f"expected 0x7F; got 0x${got.toLong}%02X")
    }
  }

  it should "pass BF16 -NaN through as FP8 -NaN (0xFF)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val got = runSingle(dut, 0xFFC0).getOrElse(BigInt(-1))
      assert(got == BigInt(0xFF), f"expected 0xFF; got 0x${got.toLong}%02X")
    }
  }

  it should "saturate BF16 +480 (NaN-encoding boundary, banker's) to 0x7E" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      // BF16 of 480 = 1.111 * 2^8.  bexp = 135 = 0x87; bmant = 111_0000 = 0x70.
      // Packed: 0_10000111_1110000 = 0x43F0.
      // Closest E4M3: 448 (byte 0x7E, even) or 480 (byte 0x7F, NaN, excluded).
      // Distance-based: 448 wins as the only available choice on that side.
      val got = runSingle(dut, 0x43F0).getOrElse(BigInt(-1))
      assert(got == BigInt(0x7E), f"expected 0x7E; got 0x${got.toLong}%02X")
    }
  }

  it should "saturate BF16 +1024 to 0x7E (overflow far above max)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      // BF16 of 1024 = 1.000 * 2^10.  bexp = 137 = 0x89; bmant = 0. = 0x4480.
      val got = runSingle(dut, 0x4480).getOrElse(BigInt(-1))
      assert(got == BigInt(0x7E), f"expected 0x7E; got 0x${got.toLong}%02X")
    }
  }

  it should "emit FP8 +4 (subnormal 0x04) for BF16 2^-7 (0x3C00)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      // BF16 of 2^-7 = 0x3C00.  Value as multiple of 2^-9 = 4 -> E4M3 0x04.
      val got = runSingle(dut, 0x3C00).getOrElse(BigInt(-1))
      assert(got == BigInt(0x04), f"expected 0x04; got 0x${got.toLong}%02X")
    }
  }

  it should "emit FP8 max-subnormal (0x07) for BF16 7*2^-9 (1.110*2^-7 = 0x3C60)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      // BF16 of 7 * 2^-9 = 1.110 * 2^-7.  bexp = 120 = 0x78; bmant = 110_0000 = 0x60.
      // Packed: 0_01111000_1100000 = 0x3C60.
      val got = runSingle(dut, 0x3C60).getOrElse(BigInt(-1))
      assert(got == BigInt(0x07), f"expected 0x07; got 0x${got.toLong}%02X")
    }
  }

  it should "cascade BF16 15/1024 (tie at subn-7.5) to FP8 min-normal 0x08" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      // 15/1024 = 1.111 * 2^-7.  bexp = 120 = 0x78; bmant = 111_0000 = 0x70 -> 0x3C70.
      // RNE tie between 7*2^-9 (0x07, odd) and 8*2^-9 = 2^-6 (0x08, even) -> 0x08.
      val got = runSingle(dut, 0x3C70).getOrElse(BigInt(-1))
      assert(got == BigInt(0x08), f"expected 0x08; got 0x${got.toLong}%02X")
    }
  }

  it should "underflow BF16 2^-11 (below half-subn-grid) to 0x00" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      // 2^-11 = 0.5 * 2^-10.  Below midpoint 2^-10; rounds to 0.
      // BF16 of 2^-11: bexp = 116 = 0x74; bmant = 0 -> 0x3A00.
      val got = runSingle(dut, 0x3A00).getOrElse(BigInt(-1))
      assert(got == BigInt(0x00), f"expected 0x00; got 0x${got.toLong}%02X")
    }
  }

  it should "RNE-tie BF16 2^-10 (exactly at midpoint to 2^-9) to 0x00 (banker's)" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      // 2^-10 = midpoint between 0 (byte 0x00 even) and 2^-9 (byte 0x01 odd).
      // Banker's -> even -> 0x00.
      // BF16 of 2^-10: bexp = 117 = 0x75; bmant = 0 -> 0x3A80.
      val got = runSingle(dut, 0x3A80).getOrElse(BigInt(-1))
      assert(got == BigInt(0x00), f"expected 0x00; got 0x${got.toLong}%02X")
    }
  }

  it should "RNE-up BF16 just above 2^-10 to subnormal 0x01" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      // First BF16 above 2^-10 has bmant = 1 -> 0x3A81.
      val got = runSingle(dut, 0x3A81).getOrElse(BigInt(-1))
      assert(got == BigInt(0x01), f"expected 0x01; got 0x${got.toLong}%02X")
    }
  }

  it should "stream a sequence of inputs with 1-elt/cycle throughput" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      // 16 inputs covering normals, signs, +/-max, BF16 NaN/Inf.
      val inputs: Seq[Int] = Seq(
        0x3F80, 0xBF80, 0x4000, 0xC040, 0x3F00, 0x3C00, 0x3BE0, 0x3BF0,
        0x43E0, 0x7F80, 0xFF80, 0x7FC0, 0x0000, 0x8000, 0x3A00, 0x3A81,
      )
      val expected = inputs.map(bf16ToFp8E4M3Fn)

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
          f"i=$i%d in=0x${inputs(i)}%04X: got 0x${outputs(i).toLong}%02X, " +
          f"expected 0x${expected(i)}%02X")
      }
    }
  }

  it should "match the FP-RNE golden bit-exactly across all 65536 BF16 inputs" in {
    test(new VFConvBf16Fp8Lane) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val inputs: Seq[Int] = 0 until 65536
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
      for ((bits, out) <- inputs.zip(outputs)) {
        val expected = bf16ToFp8E4M3Fn(bits)
        if (out.toInt != expected) {
          mismatches += 1
          if (firstMismatch.isEmpty) firstMismatch = Some((bits, out.toInt, expected))
        }
      }
      val (b0, g0, e0) = firstMismatch.getOrElse((0, 0, 0))
      info(f"exhaustive sweep: ${inputs.length}%d cases, $mismatches%d mismatches")
      assert(mismatches == 0,
        f"$mismatches%d / ${inputs.length}%d mismatches; first at BF16=0x$b0%04X " +
        f"(${bf16BitsToFloat(b0)}%.6g) got=0x$g0%02X expected=0x$e0%02X")
    }
  }
}
