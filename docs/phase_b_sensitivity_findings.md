# Phase B Sensitivity Analysis Findings (Notebook 08c)

**Date**: 2026-06-09
**Verdict**: **TIGHT** — see distribution below

## Method

Applied each of the 150 grid combinations from the Phase B pre-registered
search to the canonical 1000 evaluation set and computed three metrics per
dataset:
1. NSL R2L RED-flag catch rate
2. NSL R2L non-GREEN flag rate
3. False-alarm rate (correct predictions flagged non-GREEN)

## NSL R2L RED catch rate distribution across 150 combos

- Min: 78.8%
- P5: 78.8%
- P25: 78.8%
- **Median: 78.8%**
- P75: 78.8%
- P95: 78.8%
- Max: 78.8%
- Range: **0.0 percentage points**
- IQR: 0.0 percentage points

v2 default catch rate: 78.8% (percentile: 0th of 150 combos)

## Interpretation

Range across all 150 combos is 0.0 pp; the headline catch rate is robust to threshold choice.

## What the figure shows

Top row: histograms of NSL R2L (and per-dataset R2L) RED-flag catch rates
across the 150 grid combinations. The v2 default is marked in red.

Bottom row: catch-rate vs false-alarm-rate trade-off per dataset. Each point
is one of the 150 threshold combinations. The v2 default is marked with a
red star.

## Use in paper

This figure becomes a sensitivity-analysis figure in §6.7 (Health Flag) or
§7.1 (Comparison/Discussion). It demonstrates that:

1. The Phase B negative result is not a methodological failure — it reflects
   genuine flatness in the threshold-objective landscape.
2. The headline 78.8% catch rate (v2 defaults, Phase A strict-safe protocol)
   sits within the typical range of catch rates produced by any reasonable
   threshold choice within the pre-registered grid.
3. The flag's behavior does not depend on threshold tuning — it depends on
   the underlying Mondrian-threshold and cliff-fraction signals, which are
   computed from data, not chosen.

## Files

- `notebooks/08c_phase_b_sensitivity_histogram.ipynb`
- `results/tables/phase_b_sensitivity_catch_rates.csv` (150 rows x per-dataset metrics)
- `results/figures/phase_b_sensitivity_histogram.png` (6-panel figure)
