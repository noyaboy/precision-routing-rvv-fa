// SPDX-License-Identifier: Apache-2.0
//
// Emit SystemVerilog for the 5 Saturn-FU modules (Track I synthesis prep).
// Run: `cd saturn-fu && PATH=...:$PATH sbt "runMain saturnfu.Emit"`.
// Output: ./build/<ModuleName>.sv (one file per module).

package saturnfu

import circt.stage.ChiselStage

object Emit {
  val outDir = "build"

  val firtoolArgs = Array(
    "-disable-all-randomization",
    "-strip-debug-info",
    "-strip-fir-debug-info",
    // Emit older Verilog dialect Yosys 0.9 can parse: no `automatic` local
    // vars, no packed arrays, no SV-style `logic`.
    "-lowering-options=disallowLocalVariables,disallowPackedArrays",
  )

  def main(cliArgs: Array[String]): Unit = {
    new java.io.File(outDir).mkdirs()

    val tops = Seq[(String, () => chisel3.Module)](
      ("PolyExpQ2_30",          () => new PolyExpQ2_30),
      ("VFExpLane",             () => new VFExpLane),
      ("VFConvNvfp4Bf16Lane",   () => new VFConvNvfp4Bf16Lane),
      ("VFConvBf16Fp8Lane",     () => new VFConvBf16Fp8Lane),
      ("VFConvFp8Bf16Lane",     () => new VFConvFp8Bf16Lane),
    )

    tops.foreach { case (name, gen) =>
      println(s"==== Emitting Verilog for $name ====")
      val modOutDir = s"$outDir/$name"
      new java.io.File(modOutDir).mkdirs()
      ChiselStage.emitSystemVerilogFile(
        gen(),
        args = Array(
          "--target",     "systemverilog",
          "--target-dir", modOutDir,
          "--split-verilog",
        ),
        firtoolOpts = firtoolArgs,
      )
      println(s"     -> $modOutDir/")
    }

    println("==== Emit complete ====")
  }
}
