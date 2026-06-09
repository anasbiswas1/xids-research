# Phase B RED-Edge Sensitivity Findings (Notebook 08d)

**Date**: 2026-06-09
**Verdict**: **MODERATE**

## Why this analysis

08c showed the NSL R2L RED catch rate is structurally invariant to the GREEN-
edge thresholds we swept (all 150 combos → 78.8%). Mechanism: GREEN edges
modulate AMBER/GREEN boundary, not RED/non-RED. To complete the sensitivity
audit, 08d sweeps the RED-edge thresholds — the ones that actually determine
the headline.

## Grid

- `CLIFF_RED_LO ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.40}` (v2 default: 0.20)
- `N_RED_HI ∈ {10, 20, 30, 50, 100}` (v2 default: 30)
- T_RED edges held at degeneracy literature defaults (T_RED_LO=0.001, T_RED_HI=0.999)
- GREEN edges held at v2 defaults (since 08c proved they don't affect RED)

30 combinations total.

## Result

NSL R2L RED-flag catch rate across 30 combos:
- Min: 78.8%
- P25: 78.8%
- **Median: 78.8%**
- P75: 84.8%
- Max: 84.8%
- Range: **6.0 percentage points**

v2 default catch rate: 78.8% (0th percentile)

## Interpretation

RED-edge variation produces a 6.0 pp range. v2 sits at the 0th percentile.

## What this means for the paper

The headline 78.8% catch rate is **a property of the RED-edge selections**.
We retain v2's RED edges because they are calibration-pathology detectors
from the literature, not tuned hyperparameters:

- `T_RED_HI=0.999`: degenerate-loose conformal (threshold near 1)
- `T_RED_LO=0.001`: degenerate-strict conformal (threshold collapsed)
- `CLIFF_RED_LO=0.20`: saturation cliff (≥20% of calibration at p≥0.95)
- `N_RED_HI=30`: minimum sample support for stable isotonic fit

The figure (`phase_b_red_edge_sensitivity.png`) shows the distribution of
catch rates produced by reasonable variation in CLIFF_RED_LO and N_RED_HI.
The paper §6.7 will include this figure and state v2 defaults explicitly
as literature-derived disease markers.

## Files

- `notebooks/08d_phase_b_red_edge_sensitivity.ipynb`
- `results/tables/phase_b_red_edge_sensitivity.csv` (30 rows)
- `results/figures/phase_b_red_edge_sensitivity.png`
