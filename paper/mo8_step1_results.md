# Mo 8 step 1 — head_dim=64 scaling via divide_loop (2026-05-17 LATE-LATE+1)

## Headline

**Mo 8 step 1 PASS.** `paper/exo_schedule_fa.py` extended with a
`dequant_64_naive` @proc (64-element NVFP4 → BF16 opaque copy) and a
`schedule_dequant_64` function that tiles it into four 16-lane chunks
via `divide_loop(p, "i", 16, ["io", "ii"], perfect=True)`, then applies
the per-tile stage_mem + set_memory + replace chain from the 16-lane
foundation. Lowered C is exactly the §6 hand-coded shape:

```c
void dequant_64_naive(void *ctxt, const uint16_t* src, uint16_t* dst) {
  for (int_fast32_t io = 0; io < 4; io++) {
    vuint16m1_t src_reg;
    src_reg = __riscv_vle16_v_u16m1(&src[16 * io], 16);
    vuint16m1_t dst_reg;
    SATURN_VFCONV_NVFP4_BF16(&dst_reg, &src_reg, 16);
    __riscv_vse16_v_u16m1(&dst[16 * io], dst_reg, 16);
  }
}
```

This is the smallest concrete demonstration that the foundation
methodology (`stage_mem` + `replace()` of @instrs) composes with Exo's
tiling primitives — scaling to head_dim=N is one `divide_loop` call
when N is a multiple of 16, and one `divide_loop(tail="cut")` call
otherwise.

## What was scheduled

| Step | Op                                                              | After                                                                                  |
|------|-----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| (a)  | `divide_loop(p, "i", 16, ["io", "ii"], perfect=True)`           | `for io in seq(0,4): for ii in seq(0,16): dst[16*io+ii] = src[16*io+ii]`               |
| (b)  | `stage_mem(p, "for ii in _: _", "src[16*io:16*io+16]", ...)` ×2 | per-io load/compute/store with `src_reg`, `dst_reg` allocs of shape `[16*io+16-16*io]` |
| (c)  | `simplify(p)`                                                   | shapes collapse to literal `[16]`; inner offsets collapse to literal `ii`              |
| (d)  | `set_memory(p, "src_reg" / "dst_reg", SaturnRVV_M1)`            | buffers tagged for Saturn register groups                                              |
| (e)  | `replace(p, "for ii in _: _ #0", vfconv_nvfp4_bf16_v)`          | compute loop → Saturn vfconv lane                                                      |
| (e)  | `replace(p, "for i0 in _: _ #0", saturn_vle16_m1 / vse16_m1)`   | staged load + store loops → RVV vle16 / vse16                                          |

The emitted C uses one dynamic Saturn call per io iteration (4 issues
total per invocation), not 4 unrolled copies — GCC keeps the loop intact.

## Cross-compile + disassembly verification

Compiled with `riscv64-linux-gcc 14.2 -O2 -march=rv64gcv -mabi=lp64d`
(Bootlin toolchain at `/tmp/bootlin-14/...`). No diagnostics.

Disassembly of `dequant_64_naive` (excerpt around `.L2` = the io loop body):

```
0000000000000000 <dequant_64_naive>:
   ... stack frame + base pointer setup ...
  48: li    a5,0                                  # io_byte_off = 0
  4a: li    a7,16                                 # vl = 16
  4c: li    a6,128                                # loop limit = 4 chunks × 32 bytes
  50: vsetivli zero,16,e16,m1,ta,ma

0000000000000054 <.L2>:                           # <-- io loop body
  54: add   a0,a1,a5                              # src + io_byte_off
  58: vle16.v v1,(a0)                             # load 16 × ui16 from src[16*io]
  5c: vs1r.v v1,(a3)                              # asm clobber spill
  60: vsetvli zero,a7,e16,m1,ta,ma                # SATURN_VFCONV_NVFP4_BF16 macro
  64: vle16.v v0,(a3)
  68: 4e049057   .word 0x4e049057                 # <-- vfconv.nvfp4.bf16.v v0,v0
  6c: vse16.v v0,(a4)
  70: vl1re16.v v1,(a4)
  74: add   a0,a2,a5                              # dst + io_byte_off
  78: addi  a5,a5,32                              # io_byte_off += 32
  7c: vse16.v v1,(a0)                             # store 16 × ui16 to dst[16*io]
  80: bne   a5,a6,54 <.L2>                        # loop until io_byte_off == 128
```

Statics:
- 1 occurrence of `.word 0x4e049057` in `dequant_64_naive` (in the io
  loop; dynamically executes 4 times per call).
- 2 occurrences total in the object (the other is the 16-lane
  `dequant_chunk_naive` foundation).
- io loop bound = 128 = 4 × 32 bytes (16 × ui16 = 32 bytes/chunk). ✓.

## Two new Exo scheduling idioms surfaced

Captured in [[feedback-exo-scheduling-idioms]] alongside the two
prior idioms; replicated here for the project record:

**(3) `stage_mem` auto-renames new iters from a global `i{N}` pool, not
from the existing iter name.** A `for ii in _: _` body, after
`stage_mem`, produces load/store loops with iter `i0` — NOT `ii0`. The
existing compute loop keeps its `ii`. Pattern strings in subsequent
`replace()` calls must use the actual `i0` (or `i1`, etc.) regardless
of the outer iter context.

**(4) Tiled `stage_mem` leaves uncollapsed affine clutter that breaks
shape-validating memory classes.** After
`stage_mem(p, "for ii in _: _", "src[16*io:16*io+16]", "src_reg")`,
the alloc comes out as `src_reg: ui16[16*io + 16 - 16*io]` and the
inner refs as `src_reg[16*io + ii - 16*io]`. `SaturnRVV.alloc()`'s
literal-decimal shape check (`shape[-1].isdecimal()` in
`exo/src/exo/platforms/saturn_rvv.py`) fails on the unsimplified form.
**Always call `simplify(p)` between `stage_mem` and `set_memory` in
tiled schedules.** `simplify()` collapses both the alloc shape and
the inner ref offsets to literal constants.

## Open Mo 8 substeps

This step (1 of 4 in the `y1_handoff_kickoff.md` plan) covers
head_dim tiling for one chunk type (NVFP4 → BF16 dequant). Remaining:

2. Compose §6 FA structure: 8 heads × seq_len × QK^T + softmax
   max/sum reductions + P-quant + P·V passes.
3. Wire the remaining 2 vfconv lanes: `bf16.fp8.v` for FP8 quant
   (post-softmax → attention weights), `fp8.bf16.v` for P·V dequant.
4. Build the Exo-generated kernel on gem5 + compare cycles to
   `bench_fa_mixed_rvv_native`. Target: within 10% (Mo 8 PASS).

## Reproducibility

```
cd .
pip install -e exo/                        # editable install of Saturn-extended Exo fork
python3 paper/exo_schedule_fa.py           # emits + verifies markers + prints C bodies
```

For the disassembly probe:

```
mkdir -p /tmp/mo8s1 && cd /tmp/mo8s1
python3 -c "
import sys; sys.path.insert(0, '.')
sys.path.insert(0, './exo/src')
from exo.API import compile_procs_to_strings
from paper.exo_schedule_fa import schedule_dequant_chunk, schedule_softmax_exp_chunk, schedule_dequant_64
c, h = compile_procs_to_strings([schedule_dequant_chunk(), schedule_softmax_exp_chunk(), schedule_dequant_64()], 'exo_schedule_fa.h')
open('exo_schedule_fa.c','w').write('#include \"exo_schedule_fa.h\"\n' + c)
open('exo_schedule_fa.h','w').write(h)
"
/path/to/bootlin-riscv64-gcc14/bin/riscv64-linux-gcc \
    -O2 -march=rv64gcv -mabi=lp64d -c exo_schedule_fa.c -o exo_schedule_fa.o
/path/to/bootlin-riscv64-gcc14/bin/riscv64-linux-objdump \
    -d exo_schedule_fa.o
```
