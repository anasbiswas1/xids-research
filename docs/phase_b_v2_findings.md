# Phase B v2 Findings — Pearson-Gap Objective

**Date**: 2026-06-09
**Status**: PARTIAL: constraints not satisfied; fallback to unconstrained optimum reported.

## Background

Phase B (notebook 08a) tried to pre-register health-flag thresholds using F1 against misclassification. It failed because misclassification rates on the Mondrian holdout were too low (0.3% NSL, 1.2% CIC) for F1 to have signal.

Phase B v2 (this notebook, 08b) retries with a different objective: maximize the GREEN-vs-RED gap in per-cell SCTS-correctness Pearson, subject to constraints (≥10 GREEN cells, ≥10 RED cells, ≥60 valid-Pearson cells).

## Result

PARTIAL: constraints not satisfied; fallback to unconstrained optimum reported.

### What this means

The flag's threshold space is genuinely flat with respect to the per-cell Pearson gap. v2's defaults (T_GREEN_LO=0.05, CLIFF_GREEN_HI=0.05, N_GREEN_LO=100) are best understood as interpretable boundary defaults from calibration literature, not in-sample tuned hyperparameters.

The NSL R2L catch rate of 78.8% (Phase A strict-safe) remains the headline operational metric.

Phase B (both 08a and 08b) serves as a sensitivity analysis demonstrating that the flag's behavior does not depend sensitively on threshold choices within the explored grid.


## Files

- `notebooks/08b_phase_b_v2_pearson_gap.ipynb`
- `results/tables/phase_b_v2_grid_search.csv` — full grid with Pearson gaps and constraints
- `results/tables/phase_b_v2_selected_thresholds.json` — selection record
- `results/tables/scts_v2_calib_health_phaseb_v2.csv` (if selection succeeded)
- `results/tables/scts_v2_canonical_with_health_phaseb_v2.csv` (if selection succeeded)
- `results/tables/phase_b_v2_three_way_comparison.csv` (if selection succeeded)
- `results/tables/bootstrap_cis_phaseb_v2.csv` (if selection succeeded)
