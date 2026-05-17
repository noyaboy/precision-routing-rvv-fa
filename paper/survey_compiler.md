# Compiler-First Alternatives to Saturn-`vfexp`+MLIR Baseline

Survey of three compiler-heavy RISC-V LLM-acceleration directions for PhD application signal.
Cut-off: May 2026. Baseline = Saturn `vfexp` + fused FA RTL + MLIR pattern-match attention pass.

---

## 1. Triton-CPU on RVV (+ minimal Saturn extension)

**(a) OSS state.** `triton-lang/triton-cpu` is officially "experimental, work in progress" with no RISC-V target in the upstream backend folder. Latest main-branch commits (April 2026) target AMD CDNA/Blackwell layouts, not RVV. The only public RVV story comes from the **Terapines AI-Benchmark** fork (https://github.com/Terapines/AI-Benchmark) and the November 2024 RISC-V Foundation blog post: a private patch series (0001-patch, 0002-autotuning) on the SpacemiT K1 (256-bit RVV 1.0) running rope/matmul/softmax/layernorm. None of this has been upstreamed; no Triton-CPU PR mentions RVV/scalable vectors in the past 6 months. Custom-instruction support in Triton is essentially absent — Triton tile-level ops lower through the MLIR vector dialect, and there is no upstream RVV vector dialect (RFC is still open on Discourse from 2021).

**(b) Most-recent papers.** "Triton kernel performance on RISC-V CPU," RISC-V International blog (Nov 2024, Terapines). "ML-Triton: A Multi-Level Compilation and Language Extension to Triton" (arXiv:2503.14985, 2025, Intel). "The Anatomy of a Triton Attention Kernel" (arXiv:2511.11581, Nov 2025). None target RVV. "Compiling and Optimizing Triton Kernels onto RISC-V Targets Based on MLIR" — RISC-V Summit NA 2024 deck (Terapines, not peer-reviewed).

**(c) OSS gap (1 sentence).** No upstream Triton-CPU RVV backend exists, no MLIR vector→RVV scalable-vector lowering for `vsetvli`-style strip-mining is upstream, and zero compiler infrastructure exists for emitting custom-instruction intrinsics (e.g., `vfexp`) from Triton.

**(d) Faculty alignment.** Weak. Triton is OpenAI/Meta/Intel-owned; the only top-target academic group meaningfully publishing on Triton is **Saman Amarasinghe at MIT** (peripherally, via FlashFormer w/ Ragan-Kelley, arXiv:2505.22758) and **Tushar Krishna at GT** ("Understanding Performance Implications of LLM Inference on CPUs," 2024). **Christopher Batten (Cornell)** has students working on RVV/HW-SW co-design but no Triton publications. **Jonathan Ragan-Kelley (MIT/CSAIL)** is on Halide/Exo — not Triton.

**(e) Arch co-design credibility.** Borderline. A Triton-CPU RVV backend is 80% compiler/MLIR work. Even adding `vfexp` on Saturn (~400-800 lines Chisel) reads as the appendix; reviewers will see "Triton port + intrinsic" and call it compiler-only. To upgrade credibility you would need (i) microarchitectural tuning of Saturn (e.g., a second FU port or chained-exp dispatch) and (ii) measurements showing the *compiler* exposes parallelism the hand-coded RVV doesn't. Without that, it's a CGO/LCTES paper, not an HPCA/ISCA signal.

**(f) Realistic 12-month scope.** Tight. Triton-CPU's MLIR machinery is well-built but you need: RVV vector-dialect lowering (currently nonexistent upstream), scalable-VL handling, custom-instr intrinsic plumbing, Saturn `vfexp` RTL + verif, llama.cpp integration. Plausibly: workshop paper. Stretch for full conference.

**(g) Risks.** (i) Triton community is GPU-first; an RVV PR may sit in review indefinitely (the existing CPU x86 backend was experimental for 2+ years). (ii) Terapines/SpacemiT have a head start with 12+ months of private optimization. (iii) ML-Triton (arXiv:2503.14985) is converging on multi-level Triton extensions — concurrent. (iv) Most importantly: if SpacemiT or Andes ships an open Triton-RVV upstream in 2026, your contribution loses novelty entirely.

**(h) vs baseline.** **Weaker for an arch+compiler co-design signal.** Stronger if your target advisor is purely a compiler PI (Amarasinghe-style). The HW component is too thin and Saturn becomes "the chip Triton runs on," not a co-designed artifact.

---

## 2. Exo-RVV scheduling DSL (+ minimal Saturn extension)

**(a) OSS state.** `exo-lang/exo` (728 stars, v1.0.0 released Nov 2024, ~1,253 commits). Exo 2 paper (arXiv:2411.07211, **ASPLOS 2025**, Ikarashi/Qian/Droubi/Reinking/Bernstein/Ragan-Kelley) explicitly demonstrates a *user-defined* `vectorize` scheduling operator parameterized over "vector width, precision, memory type, and vector instructions," instantiated across **AVX-512, NEON, Gemmini, and RVM (RISC-V Matrix)** — but the official tutorial's RVV-like target is **RVM (matrix extension), not RVV**. No upstream Exo backend exists for stock RVV LLM kernels. Custom-instruction support is *the strongest of all three frameworks*: the `@instr` decorator is literally designed for declaring custom instructions; this is the canonical use case. Recent commit activity (Jan 2025–May 2026) is steady but small-team — ~6 active contributors.

**(b) Most-recent papers.** Exo 2 (arXiv:2411.07211, ASPLOS 2025). MIT News press piece (Mar 2025). Related but compiler-side: "Evolution of Kernels" (arXiv:2509.14265, Sep 2025) — LLM-guided RVV kernel synthesis, *not* Exo-based but shows demand for the gap. FlashFormer (arXiv:2505.22758, Ragan-Kelley group, May 2025) — manual CUDA, but signals the lab's interest in monolithic-kernel codegen.

**(c) OSS gap (1 sentence).** Exo has no published RVV instruction library and no flash-attention/softmax LLM scheduling demonstration on any RVV silicon — the Exo 2 paper claims "many different vector machines" but ships no RVV intrinsics file in the public repo as of May 2026.

**(d) Faculty alignment.** **Strongest of the three.** Direct line to **Jonathan Ragan-Kelley (MIT, Halide/Exo PI)** — he's an Exo 2 author and arguably the canonical "compiler+arch co-design" name in your list. **Saman Amarasinghe (MIT)** is adjacent (Halide/Tiramisu/TACO lineage). **Alex Aiken / Fredrik Kjolstad (Stanford)** are in the Halide/TACO family. **Christopher Batten (Cornell)** would value this — his group works on virtual vector ISAs (Golden/Ilan/Huang/Zhang/Batten on CIM-SRAM virtual VRA). **Niansong Zhang/Zhiru Zhang (Cornell)** specifically work on MLIR/HLS-style HW-SW co-design. **Tushar Krishna (GT)** also runs a HW-SW co-design class explicitly using DNN co-design patterns.

**(e) Arch co-design credibility.** **Best of the three.** Exo's whole thesis is that scheduling and hardware abstraction co-evolve — the `@instr`/Exo-IR model maps naturally to "I added `vfexp` to Saturn and the Exo scheduler exploits it." This is the textbook arch+compiler co-design story and reviewers will see it as such, especially if Ragan-Kelley is the application reviewer.

**(f) Realistic 12-month scope.** Achievable. Concrete plan: (i) write the missing RVV instruction file in Exo for Saturn's RVV 1.0 ISA (~2 mo, mostly mechanical), (ii) add Saturn `vfexp` Chisel + decoder + cocotb tests + `@instr` declaration in Exo (~2 mo), (iii) port FA, softmax, GEMM kernels using Exo 2 scheduling operators (~4 mo), (iv) benchmark vs hand-coded RVV intrinsics and the Titopoulos arXiv:2510.06834 baseline on FireSim/VCS or SpacemiT K1 (~2 mo), (v) write up (~2 mo). Workshop paper at PLDI SRC, CGO SRC, MLArchSys, or even a regular ASPLOS-W/PLDI-W is realistic. ASPLOS short paper is a stretch.

**(g) Risks.** (i) Exo team is small (~4 students at MIT); your work may not gain upstream adoption traction the way an IREE PR would. (ii) The Ikarashi/Ragan-Kelley group may themselves be working on an RVV instruction lib for Exo 2 — concurrent work risk is highest here. (iii) Exo's runtime story for end-to-end inference (llama.cpp integration) is *worse* than Triton/IREE, so the "full-stack" end-to-end benchmark is harder. Mitigation: drop the Exo kernels into ggml/llama.cpp manually for end-to-end numbers — that's the established pattern in the Exo papers. (iv) Exo doesn't support dynamic shapes well; KV-cache attention with variable seq_len needs careful specialization.

**(h) vs baseline.** **Comparable, with stronger compiler signal but slightly weaker HW depth.** This is the best fit for an arch+compiler co-design advisor and the best Ragan-Kelley/Batten alignment. The headline becomes "a co-designed kernel library where the scheduler is custom-instruction-aware" — that reads as a real PLDI/CGO contribution, not a port. Saturn `vfexp` RTL still gives you the ASIC-flow appendix.

---

## 3. IREE-RVV LLM compiler path (+ minimal Saturn extension)

**(a) OSS state.** Strongest existing momentum of the three. **SiFive ships IREE in its production AI stack** (https://www.sifive.com/blog/llm-optimization-and-deployment-on-sifive-intellig); the IREE RISC-V64 path lowers `linalg.matmul` → `linalg.mmt4d` → SiFive Scalable Kernel Library (SKL) microkernels, with separate prefill/decode microkernels for Llama-3.2-1B, and reports 50× decode speedup vs upstream IREE. Tiling and matrix-extension support is presented at the RISC-V Summit NA 2025. Custom-instruction story: IREE supports microkernel calls, so any custom Saturn instruction is reachable via inline assembly inside a ukernel — but there is *no MLIR pass that pattern-matches and emits* custom-instruction lowerings automatically. MX/FP4 lowering for RVV: APFloat support for E2M1/E8M0 is upstream in LLVM (PR #95392, #107127), but operation-level MX→RVV lowering does not exist upstream.

**(b) Most-recent papers.** "Accelerating GenAI Workloads by Enabling RISC-V Microkernel Support in IREE" (arXiv:2508.14899, Aug 2025, Ahmad et al., affiliation unclear — likely Mentor/Siemens or 10xEngineers). RISC-V Summit Europe 2025 SiFive presentations (multiple). "VEXP: A Low-Cost RISC-V ISA Extension for Accelerated Softmax Computation in Transformers" (arXiv:2504.11227, ARITH 2025, PULP-Platform/ETH) — **this is essentially `vfexp` already implemented on Snitch with 162× softmax speedup and 8.2× FA-2 speedup**. "Vectorized FlashAttention with Low-cost Exponential Computation in RISC-V Vector Processors" (arXiv:2510.06834, Titopoulos/Alexandridis/Dimitrakopoulos, Oct 2025) — hand-coded RVV FA, no custom instrs, gem5 simulation, **explicitly says future work is compiler integration**.

**(c) OSS gap (1 sentence).** No upstream IREE pipeline lowers `linalg.attention` or `linalg.softmax` to a custom Saturn-extension intrinsic; SiFive's stack is closed-source SKL microkernels keyed to internal SiFive cores, so the academic compiler-side path is wide open.

**(d) Faculty alignment.** Weakest of the three for top compiler+arch PIs. IREE is Google/SiFive territory, no Berkeley/MIT faculty publish on it. Closest academic neighbors: **Tushar Krishna (GT)** HW-SW co-design for ML; **Christopher Batten (Cornell)** for RVV+microkernel; **Vijay Janapa Reddi (Harvard)** for MLPerf-Tiny/edge inference on RISC-V (Flex-RV in *Nature* 2024); **Luis Ceze (UW)** for TVM legacy (he's now NVIDIA VP — limited PhD advising bandwidth). Not a direct Ragan-Kelley/Amarasinghe fit.

**(e) Arch co-design credibility.** Better than Triton-CPU, worse than Exo for "compiler co-design" framing. The plus: real custom-instruction work via microkernel calls plus an MLIR pass that pattern-matches `linalg.softmax`→`call @vfexp_microkernel` is concrete compiler IR work. The minus: most reviewers will see this as "infrastructure plumbing for a production stack" rather than a new scheduling abstraction.

**(f) Realistic 12-month scope.** Tightest. IREE is the largest codebase of the three (~500k LoC), has the steepest learning curve, and the compiler pass for matching attention into a custom-instruction microkernel needs careful design. Saturn integration, FireSim/VCS test harness, llama.cpp-equivalent end-to-end via IREE compile target. Workshop paper is realistic; full conference is risky for one person.

**(g) Risks.** (i) **SiFive is publishing aggressively** in this exact space (Summit NA/EU 2025, 50× decode speedup) — your headline number gets compared to industry. (ii) VEXP/Snitch (arXiv:2504.11227) is the published custom-exp instruction already; your Saturn `vfexp` is a reimplementation of a known design point. (iii) IREE upstream may not accept an exotic Saturn-specific pass without broader vendor support. (iv) Most critical: **the academic novelty surface is small** — SiFive owns the production story, ETH/PULP owns the custom-exp ISA story, and what's left is the MLIR pattern-match pass, which is incremental.

**(h) vs baseline.** **Roughly equivalent, with stronger industry-relevance signal and weaker novelty signal.** IREE is what production teams use; that's good for industry-leaning PhD interviews (Harvard/GT/UW lean this way) but weaker for the Berkeley/MIT/Cornell architecture-DSL crowd. The baseline's MLIR FA pass is *already* most of the compiler-novelty of this path.

---

## Bottom Line — Honest Recommendation

**Exo-RVV (Direction 2) is the strongest compiler-first repositioning.** It dominates on faculty fit (Ragan-Kelley is literally an author and a top-target Berkeley/MIT name), on co-design framing (`@instr` is the textbook hook for `vfexp`), and on novelty (no public RVV instruction library for Exo exists yet). Triton-CPU is weakest on faculty fit and HW-depth credibility. IREE is strongest on industry-relevance but the novelty surface is too crowded (SiFive + VEXP + IREE-microkernel paper already exist).

**Compared to the baseline:** Exo-RVV is *stronger* signal for the specific advisor profile "LLM arch + compiler co-design" — particularly if that profile resembles Ragan-Kelley, Amarasinghe, or Batten. The cost is that the HW component (Saturn `vfexp`) sits behind the compiler contribution rather than driving it, so applications where the advisor is primarily an RTL/microarchitecture PI (Asanović, Krishna's NoC work, Cornell's Manchester/Adve) read the baseline as the stronger story. Net: if the applicant's advisor and target labs lean compiler-DSL, switch to Exo-RVV; if they lean RTL/microarchitecture, keep the baseline.

**Key concurrent-work hazards to monitor:** VEXP (arXiv:2504.11227) is your `vfexp` competitor — cite it, position Saturn-RVV as RVV-1.0-native vs Snitch's packed-SIMD. Titopoulos FA-RVV (arXiv:2510.06834) explicitly invites compiler integration as future work — claim that gap fast. ML-Triton (arXiv:2503.14985), EoK (arXiv:2509.14265), and SiFive's IREE stack are all moving fast in adjacent lanes.

## Verified URLs / IDs Cited

- triton-cpu: https://github.com/triton-lang/triton-cpu
- Terapines AI-Benchmark: https://github.com/Terapines/AI-Benchmark
- Triton RVV blog (Nov 2024): https://riscv.org/blog/triton-kernel-performance-on-risc-v-cpu/
- Exo: https://github.com/exo-lang/exo ; tutorial https://exo-lang.dev/tutorial.html
- Exo 2 (ASPLOS 2025): arXiv:2411.07211
- IREE+RISC-V microkernel paper: arXiv:2508.14899
- SiFive IREE blog: https://www.sifive.com/blog/llm-optimization-and-deployment-on-sifive-intellig
- VEXP softmax extension (ARITH 2025, PULP/ETH): arXiv:2504.11227
- Vectorized FA on RVV (Titopoulos et al.): arXiv:2510.06834
- ML-Triton: arXiv:2503.14985
- Evolution of Kernels (EoK, RISC-V LLM-guided codegen): arXiv:2509.14265
- FlashFormer (Ragan-Kelley group): arXiv:2505.22758
- CGO 2025 multi-level RISC-V compiler backend: doi:10.1145/3696443.3708952
- MLIR RVV dialect RFC (still open): https://discourse.llvm.org/t/rfc-add-risc-v-vector-extension-rvv-dialect/4146
- Saturn manual (Berkeley TR): EECS-2024-215; repo https://github.com/ucb-bar/saturn-vectors
