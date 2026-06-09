# Phase B Findings — Pre-Registered Health-Flag Thresholds

**Date**: 2026-06-09
**Status**: For internal record; supervisor review deferred until all experimental phases complete

## What we tested

The v2 health flag's three GREEN-edge thresholds (T_GREEN_LO=0.05, CLIFF_GREEN_HI=0.05, N_GREEN_LO=100) were chosen by inspection on the same data the flag was evaluated on. Phase B replaces this with pre-registered grid search on a 50% sub-slice of the Phase A Mondrian holdout.

## Selected thresholds

- T_GREEN_LO = 0.001
- CLIFF_GREEN_HI = 0.01
- N_GREEN_LO = 500

Grid size: 150 combinations. Selection criterion: pooled per-sample F1 against misclassification labels on the threshold-calibration partition. Pooled F1 at selected thresholds: 0.0243.

## Headline results

| Metric | v2 | Strict-safe (Phase A) | Phase B (pre-registered) |
|---|---|---|---|
| Class-level flags | G37/A16/R37 | G24/A29/R37 | G22/A31/R37 |
| Sample-level flags | G7618/A3423/R6959 | G6663/A4972/R6365 | G7346/A4289/R6365 |
| NSL R2L RED rate | 94.4% | 78.8% | 78.8% |
| NSL R2L non-GREEN rate | 94.4% | 99.9% | 85.1% |

## Files produced

- `notebooks/08a_phase_b_pre_registered_flags.ipynb`
- `results/tables/phase_b_threshold_grid_search.csv`
- `results/tables/phase_b_selected_thresholds.json`
- `results/tables/scts_v2_calib_health_phaseb.csv`
- `results/tables/scts_v2_canonical_with_health_phaseb.csv`
- `results/tables/phase_b_v2_vs_strict_vs_phaseb.csv`
- `results/tables/bootstrap_cis_phaseb.csv`


## Update (Phase B v2 attempt)

A second attempt with a different objective (notebook 08b) also produced no valid selection. See `docs/phase_b_v2_findings.md` for the consolidated negative-result writeup. Both 08a and 08b serve as a sensitivity audit confirming v2 thresholds are retained as boundary defaults.
