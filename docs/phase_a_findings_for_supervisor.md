# Phase A Findings — X-IDS Strict Conformal Protocol

**Date**: 2026-06-09
**Author**: Md Anas Biswas
**Status**: For supervisor review

## What we tested

The v2 dissertation's Mondrian conformal thresholds were fitted on `test_set \ canonical_1000`. The TA flagged this as test-bound (thresholds fitted on data that is structurally part of the evaluation distribution). Phase A implements the strict alternative: thresholds fitted on a held-out 20% slice of `X_calib`, disjoint from both training and evaluation.

The models were not retrained. Macro-F1 numbers are unchanged.

## Headline numbers

| Metric | v2 protocol | Strict protocol |
|---|---|---|
| Mean SCTS overall | 67.77 | 47.48 |
| Mean c3 (conformal component) | 0.823 | 0.571 |
| Mean SCTS-correctness Pearson | +0.280 | +0.314 |
| NSL R2L RED-flag catch rate | 94.4% | 78.8% |
| Class-level flag distribution | G37/A16/R37 | G24/A29/R37 |

## What changed and why

The c3 component dropped substantially. The cause is dataset-specific:

- **NSL-KDD**: under the strict protocol, Mondrian thresholds collapse near zero for most (model, class) cells. Diagnostic in §6.7 below shows this is driven by genuine train/test distribution drift in NSL-KDD itself — a known dataset characteristic, not a Phase A bug.
- **UNSW-NB15**: largely robust to the protocol change. Mondrian thresholds and per-flag Pearson values stay close to v2.
- **CIC-IDS2017**: middle ground, modest shift.

## Calibration drift diagnostic

The diagnostic in `phase_a_calibration_drift_diagnostic.csv` measures calibrator quality on three partitions:
- `X_calib_strict` (where calibrators were fitted)
- `X_mondrian_holdout` (where strict Mondrian thresholds were fitted)
- `X_test full` (where SCTS is evaluated)

The pattern: NSL shows large ECE/accuracy gap between `X_mondrian_holdout` and `X_test full`. UNSW and CIC do not. This confirms NSL-KDD's known train/test drift is the mechanism.

## What this means for the paper

The strict protocol is the methodologically correct version. The v2 numbers were optimistic because they used test data in the conformal calibration step.

Two viable framings for journal submission:

**Framing A — Strict as the headline.** Report strict numbers as the canonical X-IDS results. Discuss v2 as a methodological alternative in an appendix. This is the cleanest scientific path.

**Framing B — Report both protocols.** Headline strict, but note v2 numbers as well, with the calibration-drift diagnostic explaining the gap. Use the NSL-KDD drift finding as a separate contribution showing the framework's diagnostic value.

Both framings are honest. Framing B has more story — it turns a correction into a finding about dataset properties.

## Pending decisions

- Which framing to adopt
- Whether to recompute downstream artifacts (SHAP, stability, Krishna) under the strict protocol — these do not depend on conformal thresholds, so they are unaffected. Only SCTS c3 and the health flag s1 component shift.
- Whether to seek a stricter holdout (Option α: retrain models with a carved-out holdout) — adds ~8 hours of compute, marginal improvement over Option β.

## Files produced

- `notebooks/07e_phase_a_strict_protocol.ipynb` — strict protocol pipeline (committed `7f6c11b`)
- `notebooks/07f_phase_a_diagnostic.ipynb` — drift diagnostic and safe-c3 fix (this notebook)
- `results/tables/scts_v2_canonical_strict.csv` — strict per-sample SCTS (07e, with NaN in 1 cell)
- `results/tables/scts_v2_canonical_strict_safe.csv` — strict per-sample SCTS (07f, safe c3, no NaN)
- `results/tables/scts_v2_canonical_with_health_strict_safe.csv` — augmented with health flag
- `results/tables/scts_v2_calib_health_strict_safe.csv` — safe-protocol health table
- `results/tables/phase_a_calibration_drift_diagnostic.csv` — Brier/ECE/accuracy per partition
- `results/tables/phase_a_nsl_per_class_drift.csv` — NSL drift drill-down
- `results/tables/phase_a_v2_vs_strict_diff.csv` — cell-by-cell v2-vs-strict comparison
