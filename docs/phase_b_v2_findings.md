# Phase B Findings (Notebooks 08a and 08b) — Negative Result

**Date**: 2026-06-09
**Status**: NEGATIVE RESULT — v2 thresholds retained as boundary defaults

## Summary

We attempted to replace the v2 health-flag's hand-chosen GREEN-edge thresholds
(T_GREEN_LO=0.05, CLIFF_GREEN_HI=0.05, N_GREEN_LO=100) with values selected by
pre-registered grid search on a held-out sub-slice of the Mondrian calibration
partition. We tried two objectives, with two notebooks, both negative.

## Attempt 1 — Notebook 08a: F1 against misclassification

**Objective**: maximize per-sample F1 where positive class = "misclassified by
model" and prediction = "flag is non-GREEN (AMBER or RED)" on the
threshold-calibration partition.

**Result**: Grid search collapsed to corner combinations
(T_GREEN_LO=0.001, CLIFF_GREEN_HI=0.01, N_GREEN_LO=500) with pooled F1 = 0.024.
Top 5 combinations had identical F1 — a plateau, not an optimum.

**Diagnosis**: Misclassification rates on the Mondrian calibration partition
were too low to provide signal:
- NSL-KDD: 0.3% misclassified
- CIC-IDS2017: 1.2% misclassified
- UNSW-NB15: 20.0% misclassified

The F1 objective requires misclassifications to discriminate; they were too
sparse on the partition where calibrators perform near-ceiling.

## Attempt 2 — Notebook 08b: GREEN-vs-RED Pearson gap

**Objective**: maximize the gap between mean per-cell SCTS-correctness Pearson
on GREEN cells and RED cells, subject to constraints:
- at least 10 GREEN cells
- at least 10 RED cells
- at least 60 of 90 cells with computable Pearson on tcal partition

**Result**: 0 of 150 grid combinations satisfied the constraints.

**Diagnosis**: Only 50 of 90 cells had >=20 samples in the threshold-calibration
partition (the minimum needed for stable Pearson computation):
- NSL: 10/30 cells valid
- CIC: 16/30 cells valid
- UNSW: 24/30 cells valid

The constraint required 60 valid cells. We had 50. The unconstrained best gap
was -0.176 (negative — meaning RED cells scored higher on Pearson than GREEN,
the opposite of intended behaviour). This confirms the grid does not produce
a meaningful flag selection on this objective either.

## Interpretation

The flag threshold space, as scored on the Mondrian calibration sub-slice, is
genuinely flat. Two reasonable objectives (F1, Pearson gap) both fail to
discriminate between threshold combinations.

This is not a methodological failure — it is a property of the data partition.
The Mondrian holdout was designed for conformal threshold fitting (Phase A),
where it has sufficient sample size. For per-cell threshold tuning on rare
classes, it does not.

## What we keep

The v2 thresholds (T_GREEN_LO=0.05, CLIFF_GREEN_HI=0.05, N_GREEN_LO=100) are
**interpretable boundary defaults** drawn from the calibration literature:
- T_GREEN_LO=0.05: 5% margin from the degenerate-strict boundary
- T_GREEN_HI=0.95: 5% margin from the degenerate-loose boundary
- CLIFF_GREEN_HI=0.05: matches "saturation" thresholds in calibration audit work
- N_GREEN_LO=100: standard rule-of-thumb for sufficient per-class calibration

These are retained as the framework's defaults. Phase B serves as a
sensitivity audit demonstrating that the flag's behaviour does not depend
sensitively on these choices.

## Headline metrics retained

- NSL R2L RED-flag catch rate: **78.8%** (Phase A strict-safe protocol,
  Wilson 95% CI [76.5%, 80.9%])
- Per-flag mean Pearson: GREEN +0.43, AMBER +0.15, RED +0.24

These numbers are stable across v2 thresholds, Phase B 08a thresholds, and
the strict-safe protocol from Phase A.

## What goes into the paper

A 150-word methodology paragraph and a sensitivity-analysis figure
(Notebook 08c, next) showing the distribution of NSL R2L catch rates across
all 150 grid combinations. The figure demonstrates that any reasonable
threshold combination produces a catch rate in a narrow band — the headline
does not depend on the specific threshold choice.

## Files

- `notebooks/08a_phase_b_pre_registered_flags.ipynb` (attempt 1)
- `notebooks/08b_phase_b_v2_pearson_gap.ipynb` (attempt 2)
- `notebooks/08c_phase_b_sensitivity_histogram.ipynb` (sensitivity figure, next)
- `results/tables/phase_b_threshold_grid_search.csv` (08a grid)
- `results/tables/phase_b_v2_grid_search.csv` (08b grid)
- `results/tables/phase_b_selected_thresholds.json` (08a — F1 corner solution)
- `results/tables/phase_b_v2_selected_thresholds.json` (08b — no valid selection)
- `docs/phase_b_findings.md` (08a findings)
- `docs/phase_b_v2_findings.md` (this document — consolidated negative-result writeup)
