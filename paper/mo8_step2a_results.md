# Mo 8 step 2a — §6 outer-loop composition (heads × seq_len × head_dim tile)

## Headline

**Mo 8 step 2a PASS.** `paper/exo_schedule_fa.py` extended with
`fa_dequant_per_row_naive` (3-D K-cache dequant over `[H=8, seq_len,
head_dim=64]` with `seq_len: size`), scheduled via the same
`divide_loop` + per-tile `stage_mem` + `simplify` + `replace` chain
from step 1. The methodology composes cleanly with two extra outer
loops and a `size`-parameterized middle dim. Lowered C is the §6
fused-FA outer shape:

```c
void fa_dequant_per_row_naive(void *ctxt, int_fast32_t seq_len,
                              const uint16_t* K_nvfp4, uint16_t* K_bf16) {
  EXO_ASSUME(seq_len > 0);
  for (int_fast32_t h = 0; h < 8; h++) {
    for (int_fast32_t s = 0; s < seq_len; s++) {
      for (int_fast32_t io = 0; io < 4; io++) {
        vuint16m1_t src_reg;
        src_reg = __riscv_vle16_v_u16m1(
            &K_nvfp4[(h) * (seq_len * 64) + (s) * (64) + 16 * io], 16);
        vuint16m1_t dst_reg;
        SATURN_VFCONV_NVFP4_BF16(&dst_reg, &src_reg, 16);
        __riscv_vse16_v_u16m1(
            &K_bf16[(h) * (seq_len * 64) + (s) * (64) + 16 * io], dst_reg, 16);
      }
    }
  }
}
```

This is the §6 outer-loop structure exactly: 8 heads × seq_len query
rows × 4 head_dim chunks. The Saturn vfconv lane fires
8 × seq_len × 4 times per invocation. The K-dequant body here is a
plain opaque copy; subsequent step-2 substeps (2b QK^T matmul, 2c
online softmax) replace this body with the corresponding compute
chain while keeping the outer loop structure intact.

## What was scheduled

The schedule is structurally identical to step 1 — the only changes
are (i) the @proc has 3-D DRAM buffers, (ii) the window expression
inside `stage_mem` is 3-D with `h` and `s` bound from outer loops:

```python
p = divide_loop(p, "i", 16, ["io", "ii"], perfect=True)
p = stage_mem(p, "for ii in _: _",
              "K_nvfp4[h, s, 16*io:16*io+16]", "src_reg")
p = stage_mem(p, "for ii in _: _",
              "K_bf16[h, s, 16*io:16*io+16]",  "dst_reg")
p = simplify(p)
p = set_memory(p, "src_reg", SaturnRVV_M1)
p = set_memory(p, "dst_reg", SaturnRVV_M1)
p = replace(p, "for ii in _: _ #0", vfconv_nvfp4_bf16_v)
p = replace(p, "for i0 in _: _ #0", saturn_vle16_m1)
p = replace(p, "for i0 in _: _ #0", saturn_vse16_m1)
```

The four idioms from `feedback-exo-scheduling-idioms` all apply
unchanged: `stage_mem` still produces fresh `i0` iter names (not
`ii0`); `simplify(p)` is still required before `set_memory` to
collapse `16*io + 16 - 16*io` (no new clutter from the `h, s` outer
indices — those are bound symbolically and don't enter the alloc shape).

## Cross-compile + disassembly verification

Compiled with `riscv64-linux-gcc 14.2 -O2 -march=rv64gcv -mabi=lp64d`.
No diagnostics. Disassembly of `fa_dequant_per_row_naive` (excerpt
around the three nested-loop bodies):

```
0000000000000148 <fa_dequant_per_row_naive>:
   ... stack frame + base pointer setup ...
  19a: li    t4,8                            # h loop count = 8
  19c: li    t1,16                           # vl = 16

<.L16>:                                     # <-- HEADS loop (h = 0..8)
  1a2: addi  a7,t5,1                         # next row end offset
  1a6: slli  a7,a7,0x7                       # × 128 (= 64 elems × 2 bytes)
  1a8: li    t3,0

<.L20>:                                     # <-- SEQ_LEN loop (s = 0..seq_len)
  1aa: addi  a5,a7,-128                      # current row byte offset

<.L17>:                                     # <-- IO loop (io = 0..4)
  1ae: add   a4,a2,a5                        # K_nvfp4 + io_byte_off
  1b2: vle16.v v1,(a4)
  1b6: vs1r.v v1,(a6)
  1ba: vsetvli zero,t1,e16,m1,ta,ma
  1be: vle16.v v0,(a6)
  1c2: 4e049057   .word 0x4e049057           # <-- vfconv.nvfp4.bf16.v
  1c6: vse16.v v0,(a0)
  1ca: vl1re16.v v1,(a0)
  1ce: add   a4,a3,a5                        # K_bf16 + io_byte_off
  1d2: addi  a5,a5,32                        # io_byte_off += 32
  1d6: vse16.v v1,(a4)
  1da: bne   a5,a7,1ae <.L17>                # io loop back-edge
  1de: addi  t3,t3,1                         # s += 1
  1e0: addi  a7,a5,128                       # next row end
  1e4: bne   a1,t3,1aa <.L20>                # s loop back-edge (a1 = seq_len)
  1e8: addi  t4,t4,-1                        # h -= 1
  1ea: add   t5,t5,a1                        # h_byte_base += seq_len (rows)
  1ec: bnez  t4,1a2 <.L16>                   # h loop back-edge
```

Statics:
- `.4byte 0x4e049057` occurs 3 times in the object (16-lane
  `dequant_chunk_naive` + tiled 64-lane `dequant_64_naive` + 3-D
  `fa_dequant_per_row_naive` — once per dequant entry point, all
  with their own loop structure around the call).
- Inside `fa_dequant_per_row_naive` the Saturn issue is 1 static
  emission at PC 0x1c2, inside `.L17` (innermost). Dynamic count =
  8 × seq_len × 4 issues per call.
- All three back-edges (`bne a5,a7` at 0x1da, `bne a1,t3` at 0x1e4,
  `bnez t4` at 0x1ec) are present — confirms no GCC fusion/unroll
  collapsed the nested structure.

## Open Mo 8 substeps

This is step 2a of the carve in `paper/mo8_step1_results.md`.
Remaining:

- **2b. QK^T tiled matmul kernel.** Requires adding ~3 fp32 @instrs
  to `exo/src/exo/platforms/saturn_rvv.py` (e.g., `vfmacc.vv`,
  `vfmul.vf` for Q · K^T accumulation; gem5 doesn't support
  `vfwmaccbf16.vv` so the kernel will pre-widen to FP32 in the load
  path, matching `bench_fa_mixed_rvv_native.c`'s existing approach).
  Then a `qkt_chunk_naive` @proc with the dequant chain feeding a
  vfmacc accumulator, scheduled through the same stage_mem + replace
  pattern.
- **2c. Online softmax mini-kernel.** Requires reduction @instrs
  (`vfredmax.vs`, `vfredsum.vs`) and the cross-tile rescale
  arithmetic. Uses the existing `vfexp_v` lane unchanged.
- **3. Wire the remaining 2 vfconv lanes** (`bf16.fp8.v`,
  `fp8.bf16.v`) for the P-quant + P·V dequant passes.
- **4. Build the Exo-generated kernel on gem5** and compare cycles
  to `bench_fa_mixed_rvv_native`. Target: within 10% (Mo 8 PASS).

## Reproducibility

Same recipe as step 1; the new schedule is exercised by the existing
demo entry point:

```
cd /home/noah/project/riscv
python3 paper/exo_schedule_fa.py           # 4/4 schedules; 9 markers
```

Disassembly probe (re-uses /tmp/mo8s1/ from step 1):

```
python3 -c "
import sys; sys.path.insert(0, '/home/noah/project/riscv')
sys.path.insert(0, '/home/noah/project/riscv/exo/src')
from exo.API import compile_procs_to_strings
from paper.exo_schedule_fa import (schedule_dequant_chunk,
    schedule_softmax_exp_chunk, schedule_dequant_64,
    schedule_fa_dequant_per_row)
c, h = compile_procs_to_strings(
    [schedule_dequant_chunk(), schedule_softmax_exp_chunk(),
     schedule_dequant_64(), schedule_fa_dequant_per_row()],
    'exo_schedule_fa.h')
open('/tmp/mo8s1/exo_schedule_fa.c','w').write('#include \"exo_schedule_fa.h\"\n'+c)
open('/tmp/mo8s1/exo_schedule_fa.h','w').write(h)
"
/tmp/bootlin-14/riscv64-lp64d--glibc--bleeding-edge-2024.05-1/bin/riscv64-linux-gcc \
    -O2 -march=rv64gcv -mabi=lp64d -c /tmp/mo8s1/exo_schedule_fa.c \
    -o /tmp/mo8s1/exo_schedule_fa.o
/tmp/bootlin-14/riscv64-lp64d--glibc--bleeding-edge-2024.05-1/bin/riscv64-linux-objdump \
    -d /tmp/mo8s1/exo_schedule_fa.o | sed -n '/fa_dequant_per_row_naive/,/<softmax_exp_chunk_naive>/p'
```
