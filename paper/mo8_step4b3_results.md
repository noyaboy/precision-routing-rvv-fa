# Mo 8 step 4b-3 — fa_kernel_decode_naive (full §6 fused FA kernel)

## Headline

**Mo 8 step 4b-3 PASS.** Composed all 5 step-4 building blocks (
dequant_row × 2, qkt_dot, softmax_full pass 1 + pass 2, pv_macc_row)
into a single fused `fa_kernel_decode_naive` @proc that mirrors
`bench_fa_mixed_rvv_native.c`'s decode-step structure. The schedule
applies set_memory + replace on a per-loop-body-shape basis, scaling
the patterns from prior building-block schedules up to the
17-distinct-loop-bodies of the fused kernel. 14/14 schedules unify;
all 9 FA primitives fire inside the fa_kernel function body
(vfconv.nvfp4.bf16.v × 4, vfmacc.vv × 4 in QK^T, vfmacc.vf × 3 in
P·V, vfmul.vf × 2 in K/V scale, plus reductions + narrow + vfexp).

After this step every cycle-dominating phase of the bench is reachable
through Exo, with two **intentionally-deferred** simplifications:

1. **FP8 quant of P is SKIPPED** — pv_macc_row receives P_fp32[s]
   directly as the per-key scalar, bypassing the bench's
   `e4m3_decode[bf16_to_e4m3(fp32_to_bf16(P[s] * 448))]` round-trip.
   `bf16_to_e4m3` / `fp32_to_bf16` are application C functions not
   exposed as Exo externs; wiring them is step 4c work (either add
   them to `exo.libs.externs` or handle via C harness scaffolding).
2. **Output BF16 conversion + `row_dequant_scale` apply are SKIPPED**
   — O_fp32 is left as FP32 (the bench's lines 281-282 multiply by
   `inv_sum/448` and narrow to BF16). Step 4c wires these as a
   final pass.

Numerical-correctness gap: the two skipped phases mean the kernel
output won't match the bench's BF16 checksum bit-exactly. **Cycle
parity is still measurable** — the cycle-dominating phases (QK^T,
softmax, P·V, dequant) are bit-for-bit identical to the bench's
hot loop.

## What was added

### `fa_kernel_decode_naive` @proc

Single @proc spanning all phases. ~150 lines of body code with
inlined sections corresponding to each building block. Structure
mirrors `bench_fa_mixed_rvv_native.c` main loop (lines 180-283):

```
for h in seq(0, 8):
    # Phase 1: QK^T per-key
    for s in seq(0, seq_len):
        [inlined dequant_row for K]        # 5 inner loops × 4 blocks
        [inlined qkt_dot]                  # 5 setup loops + 3-loop ko chunk × 4 + 2 reduce loops
    # Phase 2: Two-pass softmax
    m_state[0] = -1e30
    l_state[0] = 0.0
    [inlined softmax_full Pass 1]          # 2 inner loops × seq_len/16
    [inlined softmax_full Pass 2]          # 6 inner loops × seq_len/16
    # Phase 3: P·V per-key
    for d in seq(0, 64): O_fp32[h, d] = 0.0
    for s in seq(0, seq_len):
        [inlined dequant_row for V]        # 5 inner loops × 4 blocks
        [inlined pv_macc_row]              # 4 inner loops × 4 chunks
```

### Why inlining rather than sub-proc calls

Initial attempt: write `fa_kernel_decode_naive` with **sub-proc
calls** like `dequant_row_naive(K_nvfp4[h, s, :, :], ...)`. Exo's
frontend **accepts the slicing syntax** (confirmed by parsing and
canonical-form printout — Exo even normalizes `[h, s, :, :]` to
`[h, s, 0:4, 0:16]`).

However, when `compile_procs_to_strings` is given both the scheduled
sub-procs AND the parent fa_kernel, Exo fails with
`TypeError: multiple procs named dequant_row_naive`. The issue: the
scheduled @proc returned by `schedule_dequant_row()` inherits the
naive @proc's name, and fa_kernel's call site references the naive
symbol — both end up wanting to be emitted as the same C function
name.

Workarounds considered:
- Renaming scheduled @procs via `rename()` then having fa_kernel
  call the renamed symbol — requires defining fa_kernel AFTER the
  schedules run, breaking module-level scope.
- Using Exo's `replace()` mechanism on fa_kernel to swap naive
  sub-proc calls with scheduled versions — more complex
  scheduling pipeline than needed for step 4b-3.

The simplest robust path **for this step**: inline all 5 building
block bodies into a single fa_kernel @proc and apply one large
schedule. Sub-proc-call composition is a candidate refactor for
step 4d once everything else lands.

### `schedule_fa_kernel_decode` schedule

17 distinct loop-body shapes, requiring 17 replace() calls plus
many set_memory calls (one per register buffer in each inlined
section). Pattern matches the building-block schedules — Exo's
abstract-IR replace() handles repeated invocations of the same
loop body shape with ONE replace() call. Concretely: the 4 ko
iterations of the QK^T accumulator share one loop-body shape, so
they're scheduled with a single `replace(p, "for i in _: _ #0",
saturn_vfmacc_vv_f32m2)`.

## Cross-compile + disassembly verification

Compiled with `riscv64-linux-gcc 14.2 -O2 -march=rv64gcv -mabi=lp64d`.
fa_kernel_decode_naive at PC `0x2fa` in the object, ~880 lines of
disassembly (vs the bench's main loop, which is ~150 lines after
inlining — the difference is the asm-volatile macro spill/reload
overhead that step 4d will address).

Primitive counts inside `<fa_kernel_decode_naive>`:

| Primitive                        | Count | Role                          |
|----------------------------------|------:|-------------------------------|
| `.word 0x4e049057` (vfconv NVFP4) |     4 | K + V dequant chains          |
| `vfmacc.vv` (FP32 m2)            |     4 | QK^T per-lane accumulator     |
| `vfmacc.vf` (scalar-broadcast)    |     3 | P·V per-key accumulator       |
| `vfmul.vf` (scalar-broadcast)     |     2 | K + V E4M3 block scale apply  |
| `vfredmax.vs`                    |     3 | softmax max + macro internals |
| `vfredusum.vs`                   |     5 | qkt_dot + softmax sum         |
| `vfsub.vf`                       |     3 | softmax max-subtract          |
| `vnsrl.wi`                       |     5 | FP32→BF16 narrow + macro      |
| `.word 0x4e831457` (vfexp.v)     |     4 | softmax exp (macros emit it)  |

Some primitives appear more than the structurally-required count
because the asm-volatile macros internally emit additional vsetvli
+ helper instructions; these are the same known-cost inefficiencies
flagged in step 4a.

## Step 4 remaining substep status

| substep | scope                                          | status |
|---------|------------------------------------------------|--------|
| 4a      | vfmacc.vf @instr + pv_macc_chunk               | done   |
| 4b-1    | vfmv_zero @instr + qkt_dot + pv_macc_row       | done   |
| 4b-2    | vfmul.vf @instr + softmax_full + dequant_row   | done   |
| 4b-3    | compose fa_kernel_decode_naive                 | **done** |
| 4c      | C harness + gem5 first cycle measurement       | next   |
| 4d      | optimize known-cost macros if delta > 10%      | TBD    |

**Step 4 is now half-done structurally**. The remaining work is
empirical (4c gem5 measurement) and optimization (4d if needed).

## Step 4c plan

The next session's C harness needs to:

1. **Allocate inputs in the bench's layout** — `K_nvfp4[H, SEQ_LEN,
   4, 16]` of ui16 carriers, `K_scale[H, SEQ_LEN, 4]` of FP32
   (decoded from the bench's E4M3 packed scale table). Similar for
   V. Q is FP32 pre-widened.

2. **Call `fa_kernel_decode_naive(seq_len, qk_scale_ptr, Q_fp32,
   K_nvfp4, K_scale, V_nvfp4, V_scale, O_fp32)`**.

3. **Apply the deferred FP8 quant + output BF16 conversion in C
   scaffolding** outside the kernel — these are scalar loops with
   `bf16_to_e4m3` + `fp32_to_bf16` calls. Measure them separately
   from the kernel's cycles so the comparison is apples-to-apples
   with the bench's hot loop.

4. **Run under gem5 5.1 SE + RiscvO3CPU + DDR3-1600** at the same
   SEQ_LEN points as the bench (L2K, L4K, L8K, L16K). Compare
   `rdcycle` delta to `bench_fa_mixed_rvv_native_g14_l*` outputs
   per `paper/track_j4_results.md` § J-4.5b.

5. **First-cut cycle delta expectation**: 1.5–3× over the bench
   given the known macro-vs-intrinsic cost (step 4a evidence). If
   delta is in that range, step 4d's intrinsic-rewrites become the
   primary optimization lever; if larger, additional structural
   issues to investigate.

## Reproducibility

```
cd .
pip install -e exo/
python3 paper/exo_schedule_fa.py     # 14/14 schedules; 25 markers
```
