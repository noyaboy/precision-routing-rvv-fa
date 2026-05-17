# Mo 8 step 4a — pv_macc_chunk (proper FA P·V shape) + intrinsic-vs-macro cost evidence

## Headline

**Mo 8 step 4a PARTIAL** — step 4 (cycle parity vs
`bench_fa_mixed_rvv_native.c` within 10%) is a multi-substep
undertaking. This substep lands one new @instr (`vfmacc.vf`) and
one new @proc (`pv_macc_chunk_naive`) that **structurally match
the inner P·V loop of `bench_fa_mixed_rvv_native.c`**. A
side-finding from the disassembly: **intrinsic-based @instrs
produce dramatically cleaner code than asm-volatile-macro @instrs**
— this quantifies the "known-cost inefficiencies" flagged in steps
2b and 2c, and it shapes the optimization work that step 4b/4c
will need.

## Why step 4 is multi-substep

Reading `microbench-fa/bench_fa_mixed_rvv_native.c` carefully against
the step 2b/2c/3 @procs surfaces a structural gap:

- The **hand-coded FA decode-step kernel** does (a) **per-key dot
  products** over head_dim → scalar S[s] (vfmacc.vv across head_dim,
  then vfredusum reduction); (b) a single-pass softmax over the
  full S vector (max + exp + sum reductions across SEQ_LEN); (c)
  scalar FP8 quant of P[s]; (d) **per-key scalar-broadcast P·V**
  via `vfmacc.vf` (broadcast scalar P[s] × head_dim-wide V row).
- The **step 2b qkt_chunk_naive** does per-lane vfmacc.vv on a 16-
  lane accumulator — a valid building block of (a) but it produces
  16 partial sums per chunk and needs an outer head_dim/16 loop +
  final vfredusum to match the FA shape.
- The **step 3 pv_chunk_naive** does per-lane vfmacc.vv with both
  P and V as vectors — does NOT match (d)'s scalar-broadcast shape.
- The **step 2c online_softmax_chunk_naive** processes one 16-lane
  tile — useful but doesn't capture the full single-pass softmax
  over SEQ_LEN that the hand-coded kernel uses.

The step 2b/2c/3 chunks are valid **methodology demonstrations**
showing each precision-routing primitive is reachable from Exo's
scheduling API. To match the hand-coded kernel's cycle count, the
chunks need to be (re)composed with the correct loop nesting and
scalar-broadcast semantics. Step 4 in full needs:

| substep | scope                                                           | sessions |
|---------|-----------------------------------------------------------------|----------|
| 4a      | Add vfmacc.vf @instr + pv_macc_chunk_naive (THIS STEP)          | 1        |
| 4b      | Compose qkt_dot + softmax_full + pq + pv_macc into fa_kernel    | 1-2      |
| 4c      | C harness around fa_kernel + gem5 run + first cycle comparison  | 1-2      |
| 4d      | Optimize known-cost inefficiencies if cycle delta is large      | 1-N      |

Step 4a is the **smallest concrete piece** of step 4 — closes the
P·V structural gap and lands the missing primitive.

## What was added

### `saturn_vfmacc_vf_f32m2` @instr (exo fork saturn_rvv.py)

Standard RVV 1.0 vfmacc.vf on FP32 / LMUL=2 / 16 lanes. Scalar arg
follows the platform's `[f32][1] @ DRAM` convention (same as
`saturn_vfsub_vf_f32m2` from step 2c). Body matches
`bench_fa_mixed_rvv_native.c`'s inner P·V loop exactly:

```python
@instr("{dst_data} = __riscv_vfmacc_vf_f32m2("
       "{dst_data}, {scalar_data}, {src_data}, {vl});")
def saturn_vfmacc_vf_f32m2(
    dst:    [f32][16] @ SaturnRVV_M2,
    scalar: [f32][1]  @ DRAM,
    src:    [f32][16] @ SaturnRVV_M2,
    vl:     size,
):
    for i in seq(0, vl):
        dst[i] += scalar[0] * src[i]
```

### `pv_macc_chunk_naive` @proc (main repo exo_schedule_fa.py)

One (key, head_dim-chunk) pair of the §6 P·V accumulator. Takes
scalar `p_scalar` (FP32, dequant'd from FP8 by caller), pre-dequant'd
V chunk (FP32, 16 lanes), and O accumulator chunk (FP32, 16 lanes).
Emits one `vfmacc.vf`:

```python
@proc
def pv_macc_chunk_naive(
    p_scalar: f32[1]  @ DRAM,
    V_fp32:   f32[16] @ DRAM,
    O_acc:    f32[16] @ DRAM,
):
    V_reg: f32[16]
    O_reg: f32[16]
    for i in seq(0, 16):                  # vle32_m2 (V load)
        V_reg[i] = V_fp32[i]
    for i in seq(0, 16):                  # vle32_m2 (O load)
        O_reg[i] = O_acc[i]
    for i in seq(0, 16):                  # vfmacc.vf
        O_reg[i] += p_scalar[0] * V_reg[i]
    for i in seq(0, 16):                  # vse32_m2 (O store)
        O_acc[i] = O_reg[i]
```

Schedule is the same 4-replace() pattern as the prior chunks:

```python
def schedule_pv_macc_chunk(p=pv_macc_chunk_naive, ...):
    p = set_memory(p, "V_reg", SaturnRVV_M2)
    p = set_memory(p, "O_reg", SaturnRVV_M2)
    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vle32_m2)
    p = replace(p, "for i in _: _ #0", saturn_vfmacc_vf_f32m2)
    p = replace(p, "for i in _: _ #0", saturn_vse32_m2)
    return p
```

## Side-finding: intrinsic vs asm-volatile macro cost

Comparing disassembly between step 3's `pv_chunk_naive` (uses
asm-volatile macros for vfconv lanes + the widen) and step 4a's
`pv_macc_chunk_naive` (uses only intrinsics) is striking:

### `pv_macc_chunk_naive` (this step, all intrinsics)

```
00000000000004e6 <pv_macc_chunk_naive>:
  4e6: vsetivli  zero,16,e32,m2,ta,ma   # one vsetvli for the whole function
  4ea: vle32.v   v2,(a3)                # O_acc load
  4ee: vle32.v   v4,(a2)                # V_fp32 load
  4f2: flw       fa5,0(a1)              # p_scalar load (FP register)
  4f6: vfmacc.vf v2,fa5,v4              # accumulate
  4fa: vse32.v   v2,(a3)                # O_acc store
  4fe: ret
```

Total: 6 RVV/scalar instructions. No stack frame. No spills.

### `pv_chunk_naive` from step 3 (vfconv + widen via asm-volatile)

The same operation (modulo the dequant chain) compiles to ~70
disassembly lines, with:
- Stack frame setup + canary load/check.
- A `vs1r.v` / `vse16.v` / `vle16.v` spill-reload pair around each
  Saturn asm-volatile macro call (clobber lists force GCC to spill
  vector registers back to stack between calls).
- 6+ vsetvli flips (each macro flips SEW/LMUL internally).

Same 16-lane semantic step (load vector, multiply-accumulate, store)
but ~12× the instruction count.

### Implication for step 4 cycle parity

The "known-cost inefficiencies" flagged in steps 2b and 2c are
*quantitatively load-bearing*:

1. **vsetvli churn** — each asm-volatile Saturn macro flips
   SEW/LMUL internally. A peephole pass that merges adjacent
   macros with matching SEW/LMUL is a near-term optimization.
2. **Stack-spill through asm clobber zones** — every macro call
   forces GCC to spill+reload vector operands. The Saturn lanes
   that have no equivalent RVV intrinsic (the .4byte customs)
   *have* to go through asm-volatile, but the widen/narrow and
   reduction macros could be rewritten as intrinsic-based @instrs
   to escape this cost.
3. **DRAM-bouncing for scalars** — step 2c's vfredmax/vfredusum
   round-trip scalars through DRAM (via `flw`/`fsw`). A register-
   resident scalar variant would skip this.

The cycle-parity work in step 4c will likely need to address
(2) for the BF16<->FP32 widen/narrow (replaceable with intrinsic
vzext+vsll / vnsrl) and (3) for the reductions.

The Saturn .4byte customs (vfconv.nvfp4.bf16.v, vfconv.bf16.fp8.v,
vfconv.fp8.bf16.v, vfexp.v) are stuck with asm-volatile until /
unless GCC adds intrinsic support — which is an upstream-coordination
question that the paper §7.5 sidebar already touches on.

## Cross-compile + disassembly verification

Compiled with `riscv64-linux-gcc 14.2 -O2 -march=rv64gcv -mabi=lp64d`.
9/9 schedules unify in the demo file; 19 expected markers (1 new
for step 4a: `__riscv_vfmacc_vf_f32m2`).

Static count in the object:
- `vfmacc.vf`: 1 occurrence (only in `pv_macc_chunk_naive`).
- `vfmacc.vv`: still 1 occurrence (in `qkt_chunk_naive` from step
  2b — semantically distinct from `vfmacc.vf` per the analysis
  above).

## Remaining Mo 8 step 4 substeps

- **4b** Compose proper FA-shape kernel @procs:
  - `qkt_dot_naive`: per-key Q · K dot product over head_dim=64.
    Builds on step 2b's `qkt_chunk_naive` with a head_dim/16 outer
    loop + final `vfredusum` reduction → scalar S[s].
  - `full_softmax_naive`: single-pass softmax over SEQ_LEN-long S
    vector. max-reduce + exp + sum-reduce. Scales step 2c's
    `online_softmax_chunk_naive` up via divide_loop.
  - `fa_kernel_decode_naive`: composes qkt_dot + full_softmax +
    pq (scalar FP8 quant) + pv_macc_chunk into the full §6 fused
    decode-step kernel matching `bench_fa_mixed_rvv_native.c`'s
    structure.
- **4c** Build C harness around `fa_kernel_decode_naive` similar to
  the microbench-fa style, run under gem5 5.1 SE + RiscvO3CPU +
  DDR3-1600, compare cycles to `bench_fa_mixed_rvv_native`.
- **4d** If cycle delta > 10%, attack the known-cost inefficiencies
  identified above:
  - Rewrite SATURN_BF16_WIDEN_F32 / SATURN_F32_NARROW_BF16 macros
    as intrinsic-based @instrs (vzext.vf2 + vsll.vi / vnsrl.wi
    are all standard RVV 1.0 intrinsics — no asm-volatile needed).
  - Add register-resident-scalar reduction variants
    (vfredmax/vfredusum with SaturnRVV_M1 dst rather than DRAM).
  - Peephole-merge consecutive macros with shared SEW/LMUL.

## Reproducibility

```
cd /home/noah/project/riscv
pip install -e exo/
python3 paper/exo_schedule_fa.py     # 9/9 schedules; 19 markers
```

Disassembly probe (extend the script section in prior results docs
with `schedule_pv_macc_chunk()` in the procs list).
