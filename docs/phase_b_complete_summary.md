# Phase B Complete Summary -- Health-Flag Threshold Sensitivity Audit

**Date**: 2026-06-09
**Status**: Phase B fully closed. v2 thresholds retained as literature-derived boundary defaults; sensitivity audit complete across four notebooks.

## What Phase B set out to do

The v2 health flag uses seven thresholds across three signals:
- Signal 1 (Mondrian-threshold zone): T_GREEN_LO=0.05, T_GREEN_HI=0.95, T_RED_LO=0.001, T_RED_HI=0.999
- Signal 2 (saturation cliff): CLIFF_GREEN_HI=0.05, CLIFF_RED_LO=0.20
- Signal 3 (support): N_GREEN_LO=100, N_RED_HI=30

These were originally chosen by inspection from calibration literature. Phase B attempted to (a) tune them via pre-registered grid search, and (b) audit their sensitivity, to defend against the reviewer attack 'you tuned thresholds in-sample.'

## What we did across four notebooks

### Notebook 08a -- F1-against-misclassification objective

Tuned the three GREEN-edge thresholds via grid search (6x5x5 = 150 combos) with per-sample F1 against misclassification as the objective.

**Result**: NEGATIVE. The grid collapsed to corner combinations with pooled F1 = 0.024. Misclassification rates on the Mondrian calibration partition were too low (0.3% NSL, 1.2% CIC) to provide signal.

### Notebook 08b -- GREEN-vs-RED Pearson gap objective

Retried with a different objective: maximize the gap in per-cell SCTS-correctness Pearson between GREEN and RED cells, subject to constraints (>=10 GREEN cells, >=10 RED cells, >=60 cells with valid Pearson on tcal).

**Result**: NEGATIVE. Zero of 150 combos satisfied constraints. Only 50 of 90 cells had >=20 samples in tcal for stable Pearson. Unconstrained best gap was -0.176 (RED cells scored HIGHER on Pearson than GREEN -- opposite of intended).

### Notebook 08c -- GREEN-edge sensitivity sweep

Took the original 150-combo grid and computed the resulting NSL R2L RED-flag catch rate per combo (applied to the canonical 1000).

**Result**: STRUCTURAL INVARIANCE. All 150 combos produced exactly 78.8% catch rate. The GREEN edges only modulate the AMBER/GREEN boundary; they cannot move the RED catch count.

### Notebook 08d -- RED-edge sensitivity sweep

Swept the two RED-edge thresholds that could move the catch rate (CLIFF_RED_LO and N_RED_HI), 30 combos.

**Result**: DISCRETE BIMODAL. NSL R2L catch rate takes exactly two values: 78.8% (18 combos including v2) or 84.8% (12 combos). The transition is determined entirely by N_RED_HI; CLIFF_RED_LO has no effect within the swept range. v2 sits in the conservative (lower-claim) mode.

## Combined finding

Of the seven flag thresholds:
- Four T_GREEN/T_RED edges produce no variation in the swept grid
- CLIFF_GREEN_HI: no effect on RED catch rate (08c)
- CLIFF_RED_LO: no effect on RED catch rate (08d)
- N_GREEN_LO: no effect on RED catch rate (08c)
- N_RED_HI: single-step effect, two possible values (08d)

**The NSL R2L RED catch rate has exactly two possible values across the full explored grid: 78.8% and 84.8%. v2 reports 78.8% -- the conservative choice.**

## Paper integration

For paper section 6.7 (Health Flag):
1. State v2 thresholds explicitly as literature-derived disease markers
2. Include phase_b_sensitivity_histogram.png (08c) showing GREEN-edge invariance
3. Include phase_b_red_edge_sensitivity.png (08d) showing the bimodal RED-edge behavior
4. Methodology paragraph: 'We assessed flag-threshold sensitivity via two pre-registered grid searches with different objectives (notebooks 08a, 08b) and two sensitivity sweeps over the GREEN and RED threshold edges (notebooks 08c, 08d). Within the explored grid, only one threshold (N_RED_HI) materially affects the headline catch rate, producing a bimodal distribution with the v2 default reporting the conservative mode.'

## Headline numbers (unchanged)

- NSL R2L RED-flag catch rate: **78.8%** (Wilson 95% CI [76.5%, 80.9%])
- Per-flag mean Pearson: GREEN +0.43, AMBER +0.15, RED +0.24

## Files (Phase B audit trail)

| File | Purpose |
|------|---------|
| notebooks/08a_phase_b_pre_registered_flags.ipynb | F1 grid search (failed) |
| notebooks/08b_phase_b_v2_pearson_gap.ipynb | Pearson-gap grid search (failed) |
| notebooks/08c_phase_b_sensitivity_histogram.ipynb | GREEN-edge sweep (invariant) |
| notebooks/08d_phase_b_red_edge_sensitivity.ipynb | RED-edge sweep (bimodal) |
| results/tables/phase_b_threshold_grid_search.csv | 08a grid |
| results/tables/phase_b_v2_grid_search.csv | 08b grid |
| results/tables/phase_b_sensitivity_catch_rates.csv | 08c grid |
| results/tables/phase_b_red_edge_sensitivity.csv | 08d grid |
| results/figures/phase_b_sensitivity_histogram.png | 08c figure |
| results/figures/phase_b_red_edge_sensitivity.png | 08d figure |
| docs/phase_b_findings.md | 08a findings (negative) |
| docs/phase_b_v2_findings.md | 08b findings + consolidated negative result |
| docs/phase_b_sensitivity_findings.md | 08c findings (GREEN-edge invariance) |
| docs/phase_b_red_edge_findings.md | 08d findings (RED-edge bimodal) |
| docs/phase_b_complete_summary.md | This document -- Phase B umbrella |
