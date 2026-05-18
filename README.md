# Precision-Routing RVV Flash Attention

Open-source artifact for a co-designed mixed-precision fused
flash-attention stack on the Berkeley Saturn RVV vector unit. Headline
contributions:

- **4 RVV custom instructions on Saturn** with bit-exact-validated
  RTL (57 / 57 ChiselTests, exhaustive sweeps up to 65 536 cases per
  lane) at 0.317 mm² @ 16 nm standalone (6.34 % of Saturn) or
  0.055 mm² FMA-shared (1.10 %).
- **gem5 5.1 RISC-V decoder patch + FU-latency wiring** that issues
  the four customs natively (no `.4byte` runtime).
- **Exo 2 `@instr` library + precision-routing scheduling pass**
  declaring the four customs (plus 10 standard-RVV helpers) and
  lowering a high-level `fa_kernel_decode_naive` `@proc` to the
  fused-FA decode-step kernel through 14 sub-schedules. The
  Exo-generated kernel runs end-to-end on gem5 at **1.21× the
  hand-coded reference** (5.74 M cyc / 0.631 IPC at L2 K, vs
  4.76 M cyc / 1.366 IPC hand-coded).
- **Hand-coded fused-FA kernel** verified on gem5 RiscvO3CPU at
  `head_dim = 64` across `seq_len ∈ {2 K, 4 K, 8 K, 16 K}`:
  - 3.56× KV-cache DRAM-bandwidth reduction (verified).
  - 26-39 % per-row cycle speedup of the FU-integrated path over a
    software-dequant mixed-precision baseline at iso-FP8-quant policy.

Submission target: MLArchSys 2027 workshop (April 2027). The paper
draft lives at [`paper/paper_draft.md`](paper/paper_draft.md).

## Repositories

This project lives in three repositories that must be cloned together
(the `exo/` and `gem5/` directories of this repo are separate forks):

| Repo | Contents | URL |
|---|---|---|
| Main (this repo) | RTL + paper + microbenches + Exo schedule | `<TODO_MAIN_URL>` |
| Exo fork | `src/exo/platforms/saturn_rvv.py` with 4 Saturn customs + 10 standard-RVV helpers + scheduling pass | `<TODO_EXO_URL>` |
| gem5 fork | `src/arch/riscv/isa/decoder.isa` + `cpu/o3/FuncUnitConfig.py` patches for the 4 Saturn customs | `<TODO_GEM5_URL>` |

Clone layout (sibling directories of the main repo):

```bash
git clone <TODO_MAIN_URL> riscv && cd riscv
git clone <TODO_EXO_URL>  exo
git clone <TODO_GEM5_URL> gem5
```

## Layout

| Path                       | Contents                                                                 |
|----------------------------|--------------------------------------------------------------------------|
| [`paper/`](paper/)         | Paper draft, per-track measurement writeups, GCC bug report, design notes |
| [`saturn-fu/`](saturn-fu/) | Standalone Chisel project — 5 modules + 57 ChiselTests + Yosys synth     |
| `gem5/`                    | gem5 25.1.0.1 fork (separate repo) with decoder + FU-latency patches      |
| `exo/`                     | Exo 2 fork (separate repo) with `src/exo/platforms/saturn_rvv.py`         |
| [`microbench-mo2/`](microbench-mo2/) | NVFP4 vs BF16 K-cache DRAM-bandwidth microbench (Mo 2)        |
| [`microbench-j4/`](microbench-j4/)   | Single-instruction functional tests for the 4 customs on gem5  |
| [`microbench-fa/`](microbench-fa/)   | Full FA kernel variants — BF16 / mixed-SW / mixed-RVV / native |

The `gem5/` and `exo/` directories are separate git repos and are
listed in `.gitignore` of this repo. Clone them alongside; recipes
below assume `gem5/` and `exo/` are siblings of this directory inside
the same parent.

## Reproducing the headline numbers

Each measurement track has a self-contained writeup with a Recipes
section under `paper/`:

| What                                                            | Writeup                                          | Bench dir                                |
|-----------------------------------------------------------------|--------------------------------------------------|------------------------------------------|
| RTL bit-exact validation (57 / 57 ChiselTests)                  | `paper/track_f_results.md`, `f2_results.md`, `f3_results.md` | `saturn-fu/`                  |
| Yosys 0.9 synth + 16 nm area (Mo 4)                              | `paper/track_i_results.md`                       | `saturn-fu/build/`                       |
| NVFP4 K-cache BW reduction (Mo 2)                               | `paper/mo2_results.md`                           | `microbench-mo2/`                        |
| Mixed-prec FA cycle measurement vs SpacemiT BF16 (Mo 6)         | `paper/track_j2_results.md`                      | `microbench-fa/`                         |
| Full L-sweep + hd-sweep, GCC 14.2 toolchain (Mo 6 final)        | `paper/track_j4_results.md` (§ J-4.5b–J-4.5f)    | `microbench-fa/`                         |
| Native FU integration (gem5 decoder + FU patch)                 | `paper/track_j4_results.md`                      | `microbench-j4/`, `microbench-fa/`       |
| Exo `@instr` library smoke test                                 | `paper/exo_instr_decls.md`                       | `exo/`, `paper/exo_smoke_test.py`        |

## Toolchain

- **Bootlin GCC 14.2** (`riscv64-lp64d--glibc--bleeding-edge-2024.05-1`)
  is the default cross-compiler — the GCC 13.2 RVV `vsetvli` pass bug
  (see `paper/gcc_bug_report.md`) is fixed in 14.2+. Build with
  `-O2 -fno-tree-vectorize -march=rv64gcv -mabi=lp64d -static`.
- **gem5 25.1.0.1** SE mode, RVV `VLEN=256`, `ELEN=64`. Built with the
  `gem5-build` conda env. Requires `--num-cpus ≥ 2` for `clone()`.
- **sbt 1.10.6** for the Chisel modules under `saturn-fu/`. Install
  no-sudo by extracting the tarball under `~/.local/share/sbt/`.

## Status

This is the May-2026 working-draft artifact accompanying the paper.
The headline ≥ 1.5× speedup over BF16 RVV is *projected* on
HBM-class memory subsystems and is not met in the gem5 + DDR3-1600
configuration evaluated here (BF16 RVV stays compute-bound; the best
L2 K case lands at 1.59×, within 6 % of the literal target). The
3.56× DRAM-bandwidth reduction and the 26–39 % FU-integration cycle
delta are *verified*. See `paper/paper_draft.md` § 7.5–7.6 for the
full discussion.

The Exo-generated kernel reaches **1.21× the hand-coded reference**
on gem5 (down from 1.30× before targeted intrinsic-rewrite
optimisation). The ≤10 % compiler-parity PASS target is not yet
met; the remaining 11 pp gap is scoped in `paper/paper_draft.md` §5.6
to Saturn `.4byte` custom asm-volatile overhead — closing it
requires GCC intrinsic support for the Saturn customs (upstream
coordination), Exo register-subview support, or a peephole-merge
pass over consecutive asm-volatile macros.

Y2 work — FireSim integration for real-Saturn-µarch validation, a
HBM bandwidth model in gem5, GCC intrinsic support for the Saturn
customs to close the compiler-parity gap, and Llama-3.2-1B
end-to-end measurement — is outlined in § 9 of the paper.

## License

Released under the [3-Clause BSD License](LICENSE), matching the
Berkeley Saturn / Chipyard ecosystem this work integrates with. The
gem5 and Exo forks live in their own repos with their respective
upstream licenses; our patches there sit on top of upstream commits
and inherit those terms.
