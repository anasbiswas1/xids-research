# Phase B RED-Edge Sensitivity Findings (Notebook 08d)

**Date**: 2026-06-09
**Verdict**: **DISCRETE-STEP** -- bimodal, v2 is the conservative mode

## Why this analysis

08c showed the NSL R2L RED catch rate is structurally invariant to the GREEN-edge thresholds we swept (all 150 combos -> 78.8%). Mechanism: GREEN edges modulate AMBER/GREEN boundary, not RED/non-RED. To complete the sensitivity audit, 08d sweeps the RED-edge thresholds.

## Grid

- CLIFF_RED_LO in {0.10, 0.15, 0.20, 0.25, 0.30, 0.40} (v2 default: 0.20)
- N_RED_HI in {10, 20, 30, 50, 100} (v2 default: 30)
- T_RED edges held at degeneracy literature defaults (T_RED_LO=0.001, T_RED_HI=0.999)
- GREEN edges held at v2 defaults

30 combinations total.

## Result -- the pivot table

    CLIFF_RED_LO  0.10  0.15  0.20  0.25  0.30  0.40
    N_RED_HI
    10            78.8  78.8  78.8  78.8  78.8  78.8
    20            78.8  78.8  78.8  78.8  78.8  78.8
    30            78.8  78.8  78.8  78.8  78.8  78.8
    50            84.8  84.8  84.8  84.8  84.8  84.8
    100           84.8  84.8  84.8  84.8  84.8  84.8

The NSL R2L RED-flag catch rate takes exactly two values across the 30 combos:
- **78.8% in 18 combos** (60% of grid, including v2)
- **84.8% in 12 combos** (40% of grid)

The transition is determined entirely by N_RED_HI:
- N_RED_HI in {10, 20, 30} -> 78.8%
- N_RED_HI in {50, 100} -> 84.8%

CLIFF_RED_LO has **zero effect** on NSL R2L catch rate within the swept range.

## Mechanism

The bimodal split reflects a discrete cell-flip mechanism. Several NSL cells have per-class calibration support between 30 and 50 samples. When N_RED_HI=30, those cells satisfy signal 3 GREEN/AMBER (sufficient support). When N_RED_HI=50, the same cells flip to signal 3 RED (insufficient support). R2L samples landing in those cells then get caught at the higher rate.

CLIFF_RED_LO does not matter because NSL cells' cliff fractions are strongly bimodal (either much below 0.10 or much above 0.40) -- they do not straddle the 0.10-0.40 range we swept.

## Position of v2

v2's N_RED_HI=30 sits in the **conservative mode** -- reporting the lower catch rate (78.8%). A reviewer arguing for stricter calibration support (N_RED_HI=50 or 100) would push the headline catch rate higher, not lower. v2 is therefore not optimistic relative to reasonable alternatives within the swept grid.

## What this means for the paper

The Phase B sensitivity audit (08a, 08b, 08c, 08d combined) reveals:

1. **Six of the seven flag thresholds have zero effect** on the NSL R2L headline catch rate within reasonable variation ranges
2. **One threshold (N_RED_HI)** produces a single-step bimodal effect: either 78.8% or 84.8%, depending on the support cutoff
3. **v2's choice** lands in the conservative (lower-claim) mode

This is a stronger position than 'robust to threshold tuning' -- it is 'the headline depends on exactly one threshold choice, and v2 is the conservative selection.'

## Per-dataset comparison

- NSL-KDD: bimodal {78.8%, 84.8%}, range 6.0 pp
- UNSW-NB15: small variation 0.3-3.1%, range 2.8 pp (R2L predictions rare)
- CIC-IDS2017: bimodal {~0%, ~84%}, range 84.4 pp -- but absolute R2L count on CIC is small (NSL R2L is the headline)

## Files

- notebooks/08d_phase_b_red_edge_sensitivity.ipynb
- results/tables/phase_b_red_edge_sensitivity.csv (30 rows)
- results/figures/phase_b_red_edge_sensitivity.png
