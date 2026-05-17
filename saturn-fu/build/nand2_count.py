# NAND2-equivalent gate counts using standard industry weights.
# Source: typical ASIC standard-cell library scaling factors.
W = {
    "$_NOT_":    0.5,
    "$_NAND_":   1.0,
    "$_NOR_":    1.0,
    "$_AND_":    1.25,
    "$_OR_":     1.25,
    "$_ANDNOT_": 1.5,
    "$_ORNOT_":  1.5,
    "$_AOI3_":   2.0,
    "$_OAI3_":   2.0,
    "$_AOI4_":   2.5,
    "$_OAI4_":   2.5,
    "$_MUX_":    2.5,
    "$_XOR_":    2.5,
    "$_XNOR_":   2.5,
    "$_DFF_P_":  5.0,
}

modules = {
    "PolyExpQ2_30 (standalone)": {
        "$_ANDNOT_": 8166, "$_AND_": 4226, "$_AOI3_": 2231, "$_AOI4_": 2,
        "$_DFF_P_": 293, "$_NAND_": 362, "$_NOR_": 2537, "$_NOT_": 2942,
        "$_OAI3_": 395, "$_ORNOT_": 273, "$_OR_": 2218, "$_XNOR_": 4120,
        "$_XOR_": 8344,
    },
    "VFExpLane (incl. 1 PolyExp)": {
        "$_ANDNOT_": 9538, "$_AND_": 4358, "$_AOI3_": 2635, "$_AOI4_": 3,
        "$_DFF_P_": 527, "$_MUX_": 156, "$_NAND_": 423, "$_NOR_": 3065,
        "$_NOT_": 3405, "$_OAI3_": 543, "$_ORNOT_": 455, "$_OR_": 2644,
        "$_XNOR_": 4818, "$_XOR_": 9800,
    },
    "VFConvNvfp4Bf16Lane": {
        "$_ANDNOT_": 86, "$_AND_": 12, "$_AOI3_": 32, "$_AOI4_": 1,
        "$_DFF_P_": 54, "$_MUX_": 11, "$_NAND_": 20, "$_NOR_": 20,
        "$_NOT_": 25, "$_OAI3_": 25, "$_OAI4_": 1, "$_ORNOT_": 9,
        "$_OR_": 36, "$_XNOR_": 13,
    },
    "VFConvBf16Fp8Lane": {
        "$_ANDNOT_": 52, "$_AND_": 5, "$_AOI3_": 23, "$_DFF_P_": 30,
        "$_MUX_": 15, "$_NAND_": 11, "$_NOR_": 13, "$_NOT_": 20,
        "$_OAI3_": 8, "$_OAI4_": 2, "$_ORNOT_": 11, "$_OR_": 59,
        "$_XNOR_": 3, "$_XOR_": 13,
    },
    "VFConvFp8Bf16Lane": {
        "$_ANDNOT_": 18, "$_AOI3_": 7, "$_DFF_P_": 11, "$_MUX_": 6,
        "$_NAND_": 4, "$_NOR_": 2, "$_NOT_": 5, "$_OAI3_": 1,
        "$_ORNOT_": 4, "$_OR_": 10,
    },
}

# 16 nm standard cell NAND2 area (TSMC 16FF+ HD library, ~0.26 um^2)
NAND2_AREA_UM2 = 0.26
SATURN_BASELINE_MM2 = 5.0  # per Zhao et al. arXiv:2412.00997

print(f"{'Module':<32} {'Cells':>8} {'NAND2-eq':>10} {'Area@16nm':>13}")
print("-" * 70)
totals = {}
for name, cells in modules.items():
    n2 = sum(c * W[g] for g, c in cells.items())
    cellct = sum(cells.values())
    a_um2 = n2 * NAND2_AREA_UM2
    print(f"{name:<32} {cellct:>8d} {n2:>10.0f} {a_um2/1000:>10.3f} K um^2")
    totals[name] = n2

# 16-lane projection per fu_sketch.md
LANES = 16
vfexp_total = totals["VFExpLane (incl. 1 PolyExp)"] * LANES
vfconv_total = (
    totals["VFConvNvfp4Bf16Lane"]
    + totals["VFConvBf16Fp8Lane"]
    + totals["VFConvFp8Bf16Lane"]
) * LANES
fu_total = vfexp_total + vfconv_total
fu_area_um2 = fu_total * NAND2_AREA_UM2
fu_area_mm2 = fu_area_um2 / 1e6

print()
print(f"=== 16-lane FU rollup (standalone synth, no FMA share) ===")
print(f"  vfexp @ {LANES} lanes: {vfexp_total:>12.0f} NAND2 = {vfexp_total * NAND2_AREA_UM2 / 1e6:>6.3f} mm^2")
print(f"  vfconv (3 lanes) @ {LANES}: {vfconv_total:>11.0f} NAND2 = {vfconv_total * NAND2_AREA_UM2 / 1e6:>6.3f} mm^2")
print(f"  FU total:           {fu_total:>12.0f} NAND2 = {fu_area_mm2:>6.3f} mm^2")
print(f"  Saturn baseline:                            {SATURN_BASELINE_MM2:>6.3f} mm^2")
print(f"  FU fraction of Saturn:                      {fu_area_mm2 / SATURN_BASELINE_MM2 * 100:>6.2f}%")

# FMA-shared projection: subtract PolyExp from each vfexp lane.
poly_n2 = totals["PolyExpQ2_30 (standalone)"]
vfexp_wrapper_per_lane = totals["VFExpLane (incl. 1 PolyExp)"] - poly_n2
fu_shared = vfexp_wrapper_per_lane * LANES + vfconv_total
fu_shared_mm2 = fu_shared * NAND2_AREA_UM2 / 1e6
print(f"\n=== FMA-shared projection (subtracting PolyExp mults from vfexp) ===")
print(f"  vfexp wrapper (no muls) @ {LANES}: {vfexp_wrapper_per_lane * LANES:>10.0f} NAND2 = {vfexp_wrapper_per_lane * LANES * NAND2_AREA_UM2 / 1e6:>6.3f} mm^2")
print(f"  vfconv (unchanged):                  {vfconv_total:>10.0f} NAND2 = {vfconv_total * NAND2_AREA_UM2 / 1e6:>6.3f} mm^2")
print(f"  FU total (FMA-shared):               {fu_shared:>10.0f} NAND2 = {fu_shared_mm2:>6.3f} mm^2")
print(f"  FU fraction of Saturn:                                          {fu_shared_mm2 / SATURN_BASELINE_MM2 * 100:>6.2f}%")
