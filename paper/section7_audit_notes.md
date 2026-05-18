# §7 Evaluation Audit Notes (2026-05-18)

Defensive sweep of §7 (Evaluation) for stale post-Mo-8 references after
the paper-wide update + figure conversion + public-push work. Audit
performed against the live paper draft + the public repos
(`github.com/noyaboy/*`).

## Scope

§7 covers the hand-coded reference kernel measurements. §5.5–§5.6 cover
the Exo-generated kernel measurements (rewritten in commit 206665e
post-Mo-8). The audit specifically looked for:

1. Numerical inconsistencies between §5 (Exo) and §7 (hand-coded)
   tables.
2. Stale claims about GCC version status, FU coverage, or substep
   status now obsoleted by Mo 8.
3. References to artifacts (gem5 patches, Exo @instrs, scripts,
   figures) that don't match the current commit / push state.

## Findings

### 1. Numerical inconsistency: hand-coded native at L2K (RESOLVED 2026-05-18)

**Resolution**: the `bench_fa_mixed_rvv_native_g14` binary on disk
at the time of the step 4c gem5 run (2026-05-18) was **stale** —
built 2026-05-17 18:50:30 from an earlier source state; the L4K
/ L8K / L16K binaries (rebuilt 2026-05-17 20:00) capture the
post-edit state. Rebuilding the L2K binary from the current
source (2026-05-18 19:05) and re-running gives **5,490,885 cyc /
2.395 IPC**, exact match with §7.5 Table 5's 5.49 M / 2.40 IPC.
The 4.76 M figure in the pre-resolution §5.6 draft was from the
stale binary.

**Mo 8 PASS implication**: the compiler-parity ratio recomputes
as 5,743,133 / 5,490,885 = **1.046×, MEETING the ≤10 % PASS
target**. §5.6 + §1 + §9 + Abstract + README + step-4d result
docs all updated in the same commit as this audit-notes update.

**Original audit context (kept for the record)**:

**Location**: §5.6 row "Hand-coded `bench_fa_mixed_rvv_native_g14`"
vs §7.5 Table 5 row "L2 K Native (FU)".

**Discrepancy**:

| Source             | L2K native cycles | L2K native IPC |
|--------------------|------------------:|---------------:|
| §5.6 (this paper)  | 4 756 714         | 1.366          |
| §7.5 Table 5       | 5.49 M            | 2.40           |
| Project memory     | 7.00 M            | 2.19           |

Three different numbers for "hand-coded native at L2K". Likely
sources:

- **GCC 13.2 vs 14.2 toolchain transition**: project memory's 7.00 M
  is from the GCC 13.2 era; Table 5 documents the 14.2-rebuild
  improvement ("13–22 % faster than the corresponding GCC 13.2 +
  workaround baselines reported in earlier drafts of this paper" per
  §7.7).
- **Bench-binary rebuild between Track J-4.5b (paper-final
  measurements) and Mo 8 step 4c (current §5.6 measurement)**: the
  `bench_fa_mixed_rvv_native_g14` binary on disk (used in the step
  4c run that produced 4.76 M) was built 2026-05-17 18:50. Track
  J-4.5b runs that fed Table 5 happened earlier the same day; the
  binary may have been rebuilt between the two passes.
- **gem5 launch-flag differences**: §7.1 says "4 cores"; step 4c
  used `--num-cpus=2`. Single-threaded cycle counts should be
  unaffected, but it's a non-zero possibility.
- **gem5 binary rebuild**: gem5 fork commit `841d376` is unchanged
  since 2026-05-17 PM, so no gem5-side rebuild between the two runs.

**Recommended action** (not in this audit pass):
- Re-run `bench_fa_mixed_rvv_native_g14` at L2K under §7.1's exact
  launch args (`--num-cpus=4` per §7.1) and compare to both 4.76 M
  (step 4c) and 5.49 M (Table 5).
- If the canonical hand-coded number is 4.76 M, update §7.5 Table 5
  + §7.7's "13–22 % faster than GCC 13.2 baselines" claim.
- If the canonical number is 5.49 M (i.e., my step 4c got a bad
  outlier reading), re-run the Exo bench under the same conditions
  and update §5.6's cycle ratio (currently 5.74 / 4.76 = 1.21 ×;
  would become 5.74 / 5.49 = 1.05 × under Table 5's reference —
  putting the Exo result **inside** the ≤10 % Mo 8 PASS target).

**Stakes**: the Mo 8 PASS verdict in §5.6 depends on which baseline
is canonical.

### 2. §7.7 GCC 14.2 claim outdated (FIXED in this commit)

**Location**: §7.7 second-to-last paragraph.

**Old claim**: "Cross-version testing on three Bootlin toolchains
confirms the bug is **fixed in GCC 14.2.0 and 15.1.0**, present only
in GCC 13.x".

**Reality** (per Mo 8 step 4d-1, 2026-05-18): GCC 14.2's fix is
*partial*. The pure-intrinsic widening chain is fixed; but when an
asm-volatile block precedes the widening chain and internally
changes vtype, GCC 14.2 still fails to emit a vsetvli before the
next intrinsic — same wrong-SEW miscompile pattern.

**Update applied** (this audit pass):
- §7.7 paragraph rewritten to qualify "fixed" as "pure-intrinsic
  form fixed" and add a sentence on the asm-volatile boundary
  limitation, with the working `asm volatile ("vsetvli ...")`
  workaround inline.
- `paper/gcc_bug_report.md` already extended with Part 2 in
  commit cac2b6d.

### 3. Other §7 subsections — clean

§7.1 (Setup), §7.2 (RTL Validation), §7.3 (Area), §7.4 (KV-cache
DRAM Bandwidth), §7.5 Table 5 (modulo finding #1), §7.6 (literal
≥1.5× threshold discussion): no stale Mo 8 references found. These
sections describe the hand-coded kernel measurements as-of paper
finalization and are independent of the Mo 8 Exo work.

## Summary

| Finding                                                | Status         |
|--------------------------------------------------------|----------------|
| §5.6 vs §7.5 L2K cycle inconsistency (4.76 M vs 5.49 M) | **Resolved** — bench rebuilt 2026-05-18; canonical = 5.49 M; Mo 8 PASS verdict flipped to MET |
| §7.7 GCC 14.2 "fixed" claim                            | **Fixed** in the audit pass    |
| §7.1, §7.2, §7.3, §7.4, §7.6 other sections            | Clean          |

The reconciliation flipped the Mo 8 compiler-parity verdict from
"not met (1.21 ×)" to **MET (1.046 ×)**. The Exo result of 5.74 M
cyc against the rebuilt hand-coded 5.49 M baseline lands inside
the ≤10 % PASS threshold by a margin of 5.4 pp.
