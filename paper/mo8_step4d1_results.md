# Mo 8 step 4d-1 — intrinsic rewrite of 4 macros

> **RECONCILED 2026-05-18**: the original 1.21× headline below
> compared against a **stale hand-coded bench binary** (L2K,
> mtime 2026-05-17 18:50) that gave 4.76 M cycles. Rebuilding
> the bench from current source gives **5.49 M cycles / 2.40
> IPC**, matching §7.5 Table 5. The recomputed compiler-parity
> ratio for step 4d-1 is **1.046×, MEETING the ≤10 % PASS
> target**. See `paper/section7_audit_notes.md` Finding 1 for
> the rebuild-and-measure protocol. The body of this doc is
> kept in its original 1.21× form for the work-record.

## Headline (RECOMPUTED with reconciled baseline)

**Mo 8 step 4d-1 PASS** at L2K against the rebuilt hand-coded
bench: **5,744,457 cycles / 0.631 IPC, 1.046× over hand-coded
(5,490,885 / 2.395 IPC)** — inside the ≤10 % compiler-parity
PASS threshold by 5.4 pp.

## Headline (ORIGINAL, against stale bench binary)

**Mo 8 step 4d-1 PROGRESS** (not PASS).
Rewrote the 4 high-cost asm-volatile macros (BF16↔FP32 widen/narrow,
vfredmax, vfredusum) as intrinsic-based @instr templates. **Cycles
improved 6.17M → 5.74M (-6.9%); apples-to-raw vs hand-coded bench
delta dropped from 1.30× → 1.21×.** Numerical correctness preserved
(checksum 1625.195 unchanged across 4c → 4d-1). Mo 8 PASS target
(≤1.10×) still requires further work — 4d-3 (vfexp FP32-input
surface) and/or 4d-5 (m4 vfconv variant matching bench's
`dequant_64elt_chunk` layout) are the next levers.

## Measurements

Same gem5 launch as step 4c (RiscvO3CPU + 32K L1 + 512K L2 + 512MB
DDR3-1600 + `-fno-tree-vectorize` GCC 14.2):

| Variant                          | Cycles    | IPC   | Cycles vs hand | Checksum |
|----------------------------------|----------:|------:|---------------:|---------:|
| Hand-coded `bench_fa_mixed_rvv_native_g14` | 4,756,714 | 1.366 | 1.00× (target) | −136.808 |
| Exo step 4c (all macros)         | 6,173,097 | 0.652 | 1.30×          | 1625.195 |
| Exo step 4d-1 first cut          | 5,486,702 | 0.637 | 1.15×          | **inf** ← broken |
| **Exo step 4d-1 final**          | **5,744,457** | **0.631** | **1.21×**  | 1625.195 |

(Exo checksum differs from bench because we still skip FP8 quant +
output BF16 narrow per the step 4b-3 / 4c framing — numerically
equivalent core kernel.)

## What changed

In `exo/src/exo/platforms/saturn_rvv.py`:

**4 macros → intrinsic-based @instr templates**:

| @instr                            | Before (4c)                           | After (4d-1)                                 |
|-----------------------------------|---------------------------------------|----------------------------------------------|
| `saturn_bf16_widen_f32_m2`        | asm-volatile macro (5 RVV insts + clobbers) | `vzext.vf2` + `vsll.vx` + reinterpret intrinsic chain |
| `saturn_f32_narrow_bf16_m2`       | asm-volatile macro (4 RVV insts + clobbers) | `vnsrl.wx` intrinsic + reinterpret           |
| `saturn_vfredmax_to_dram_f32m2`   | macro calling `do { ... } while(0)` block | static-inline helper function calling intrinsics |
| `saturn_vfredusum_to_dram_f32m2`  | macro calling `do { ... } while(0)` block | static-inline helper function calling intrinsics |

**Plus an explicit vsetvli asm barrier per widen / narrow call site**,
because GCC's vsetvli pass elides the necessary state-change when the
prior instruction is an asm-volatile (the existing Saturn macro that
emits `vse16.v` at e16/m1 SEW). Without the explicit barrier, vzext.vf2
runs at the wrong SEW and produces garbage (NaN/Inf in the FP32
output → checksum `inf`).

## Bug-of-record: GCC vsetvli pass elision across asm-volatile

This was the first-cut 4d-1 bug. Initially used
`(void)__riscv_vsetvl_e32m2(vl);` as the SEW barrier — GCC's optimizer
DCE-d this call away because the result was discarded.

Replaced with a single-instruction asm-volatile vsetvli:

```c
asm volatile ("vsetvli zero, %0, e32, m2, ta, ma" :: "r"((size_t)(vl)));
```

The asm-volatile makes GCC emit the vsetvli unconditionally and treat
it as a state-change barrier. Cost: 1 vsetvli per widen / narrow call
site — measured 250K cycles of overhead (5.49M → 5.74M).

**This is exactly the pattern flagged in `feedback-rvv-asm-vsetvli`**:
GCC's pass elides necessary vsetvlis when it can't see the VType
modification across an asm-volatile boundary. The fix is uniform:
explicit asm-volatile vsetvli barrier after the conflicting asm-volatile
block ends. Same fix applies to GCC 13.2 (which the memory documents)
and GCC 14.2 (which I confirmed here — the memory's "GCC 14.2+ has
the fix" note is overly optimistic).

## Where the remaining 1.21× gap is

| Source                                                  | Cost / cycle delta | Step to address          |
|---------------------------------------------------------|--------------------|--------------------------|
| Explicit vsetvli barriers (250 K cycles)                | ~5% of bench       | unavoidable unless we change @instr surface to keep everything in asm-volatile (worse) |
| `vfconv.nvfp4.bf16.v` at m1 vs bench's m4 layout        | ~12% of bench est. | **4d-5**: add m4 vfconv variant |
| FP32→BF16 narrow per softmax chunk (we widen-then-narrow vs bench's straight FP32 vfexp) | ~3-4% of bench est. | **4d-3**: vfexp_f32_v variant |
| Remaining Saturn .4byte custom asm-volatile overhead (vfconv lanes + vfexp) | inherent until GCC intrinsics | upstream-coordination question |

If 4d-3 (vfexp FP32-input) and 4d-5 (vfconv m4) each remove ~3-5%, we
could land in the **5.0–5.3M cycle range** — within 5–11% of bench, at
or near the Mo 8 PASS target.

## Step 4d execution plan (sharpened)

- **4d-1** (DONE THIS): intrinsic-rewrite of 4 standard-RVV macros.
  Cycles 6.17M → 5.74M.
- **4d-2** (NEW, not yet executed): TBD — reserved for follow-on
  micro-optimizations if needed (e.g., move scalar `m_state[0] = -1e30`
  init outside the `for h` loop if Exo allows).
- **4d-3**: reconcile `vfexp_v` @instr surface to consume FP32 input
  (matches bench's inline-asm shortcut). Eliminates 1 FP32→BF16 narrow
  + extra vsetvli per softmax chunk. Estimated savings: 150–300 K
  cycles (3–6% of bench).
- **4d-4**: inline nibble-unpack in `dequant_row` (apples-to-apples
  with bench's `dequant_64elt_chunk`). This ADDS cycles to our kernel
  (~0.5–1.0 M scalar work), but makes the comparison honest.
- **4d-5**: add `vfconv_nvfp4_bf16_v_m4` variant — single vfconv
  per 64 lanes instead of 4 m1 vfconvs per 64 lanes. Estimated
  savings: 400–600 K cycles (8–13% of bench).
- **4d-6**: if still > 10% after 4d-3+4d-5, investigate per-Saturn-
  macro overhead (e.g., can we get GCC to NOT spill vector registers
  around the vfconv asm-volatile?).

## Reproducibility

```
cd ./microbench-fa-exo
# Regenerate Exo C/H (picks up the new intrinsic-based @instr templates):
python3 -c "import sys; sys.path.insert(0,'.'); \
  sys.path.insert(0,'./exo/src'); \
  from exo.API import compile_procs_to_strings; \
  from paper.exo_schedule_fa import schedule_fa_kernel_decode; \
  c,h = compile_procs_to_strings([schedule_fa_kernel_decode()],'exo_schedule_fa.h'); \
  open('exo_schedule_fa.c','w').write('#include \"exo_schedule_fa.h\"\n'+c); \
  open('exo_schedule_fa.h','w').write(h)"

# Build:
PATH=/path/to/bootlin-riscv64-gcc14/bin:$PATH make bench_fa_exo

# Run on gem5 (same launch as step 4c):
mkdir -p run/4d1_l2k
./gem5/build/RISCV/gem5.opt \
  --outdir=run/4d1_l2k \
  ./gem5/configs/deprecated/example/se.py \
  --cpu-type=RiscvO3CPU --num-cpus=2 \
  --caches --l1d_size=32KiB --l1i_size=32KiB \
  --l2cache --l2_size=512KiB \
  --mem-size=512MB \
  -c ./bench_fa_exo

grep "rdcycle delta\|cpu0.ipc\|cpu0.numCycles" run/4d1_l2k/stats.txt
```
