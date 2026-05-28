# X-IDS v2 Rebuild — Day 1 + Day 2 Progress Record

**Author:** Md Anas Biswas
**Date:** Recorded at end of Day 2 of methodology rebuild
**Supervisor meeting:** June 1, 2026
**Repo:** https://github.com/anasbiswas1/xids-research

**Purpose of this document:** Snapshot of methodology rebuild progress after Day 1 (preprocessing) and Day 2 (training). All numbers reflect what was reported in notebook execution and committed to the v2 CSV tables on GitHub. A formal cross-document re-audit against the v2 CSVs is scheduled for Day 4. Days 3–4 (calibration, SHAP, stability, agreement, SCTS-v2, documentation cascade) remain.

---

## 1. Why we are doing this rebuild

A methodological audit raised the following concern about the v1 pipeline:

> The v1 calibration step fit isotonic calibrators on a 50/50 split of the *test* set, then evaluated on the other half. Although no individual sample appeared in both stages, the calibrator was tuned to the test-set distribution. For NSL-KDD and UNSW-NB15 — both of which have official train/test partitions, with the test partition deliberately containing distribution shift — this is reviewer-rejection-level methodology.

A full methodology audit was conducted, identifying 20 potential issues (M1–M20). The full audit table is in §5; this section summarises only the most material findings:

- **M1 (one-hot encoder fit on train+test union)**: confirmed broken on NSL-KDD and UNSW-NB15 (mild leak)
- **M2 (StandardScaler)**: safe — fit on train only in all three datasets
- **M4 (SMOTE timing)**: safe — applied after train/test split in all three datasets
- **M6 (calibrator on test set)**: confirmed broken across all three datasets
- **M7 (per-class isotonic formulation)**: safe — standard one-vs-rest in all three datasets
- **M9 (DNN SHAP background)**: safe — drawn from training data
- **M10 (RF tree depth)**: addressed in Day 2 via max_depth=20 cap
- **M14 (conformal threshold on test set)**: same root cause as M6, fixed by the same architectural change

The remaining issues (M8, M11, M12, M13, M15, M16, M17, M19, M20) are either documentation-framing items, pending Day 3 work (bootstrap CIs), or design-tradeoff decisions deferred to the supervisor meeting. M3, M5, and M18 were minor issues identified during the audit but assessed as low priority for the v3 paper; they are flagged here for completeness but not pursued. See §5 for the full status table.

**Scope of rebuild (Budget A, fix everything across all three datasets, 3–4 days):**
- Fix M1 (encoder fit on train only)
- Fix M6/M14 (carve calibration set from train, evaluate on untouched test)
- Apply RF max_depth=20 cap (addresses M10/pre-submission task on UNSW tree depth)
- Add per-architecture imbalance ablation (CW vs SMOTE) to justify canonical SMOTE choice
- Regenerate all downstream stages (Notebooks 03–08 v2)

---

## 2. Day 1 — Preprocessing v2 across all three datasets

### 2.1 Design decisions (all three datasets)

| Decision | Rationale |
|---|---|
| 80/20 stratified train/calibration split (NSL, UNSW) | Preserves the official test partition untouched. 20% calibration is enough samples for isotonic on rare classes. |
| 60/20/20 stratified split (CIC) | CIC has no official partition. Random stratified split preserves the same train/calibration/test architecture for methodological consistency across datasets. |
| Stratified by 5-class label | Preserves rare-class (U2R) proportions across train and calibration |
| One-hot encoder fit on train only | M1 fix. Test-only categorical values produce all-zero one-hot rows (out-of-vocabulary handling). |
| StandardScaler fit on train only | Already correct in v1, preserved |
| Output to `_v2` directories | Parallel to v1, leaves v1 intact for comparison |

### 2.2 Per-dataset preprocessing results

#### NSL-KDD v2 — `01_data_exploration_v2.ipynb` (commit `0c5b7e1`)

| Split | Samples | Notes |
|---|---:|---|
| Train | 100,778 (80%) | 80% slice of official training set |
| Calibration | 25,195 (20%) | New 20% slice carved from training set |
| Test | 22,544 | Official NSL-KDD test set, untouched |
| Features | 122 | Identical to v1 (zero practical M1 impact) |

**OOV finding (NSL):** Zero test-only categorical values. The v1 encoder schema fit on train+test union produced identical columns to a train-only schema. Paper-worthy: documents that M1 was technically a leak but produced zero practical leakage on NSL.

**Per-class split distribution (NSL):**

| Class | Train | Calib | Test |
|---|---:|---:|---:|
| Normal | 53,874 | 13,469 | 9,711 |
| DoS | 36,741 | 9,186 | 7,460 |
| Probe | 9,325 | 2,331 | 2,421 |
| R2L | 796 | 199 | 2,885 |
| U2R | 42 | 10 | 67 |

**Verification:** ✓ All files round-trip correctly. ✓ Train and calibration sets are disjoint. ✓ Train + calibration covers the original training set exactly.

#### UNSW-NB15 v2 — `01_unsw_data_exploration_v2.ipynb` (commit `516e3c6`)

| Split | Samples | Notes |
|---|---:|---|
| Train | 108,272 (80%) | 80% of post-Generic-drop training |
| Calibration | 27,069 (20%) | New 20% slice |
| Test | 63,461 | Official test, post-Generic-drop, untouched |
| Features | 194 | Post-label-drop feature count |

**OOV finding (UNSW):** Five test rows have `state` values (`ACC`, `CLO`) that appear only in test. These rows receive all-zero state encoding under v2. Impact: 5 out of 63,461 test rows (0.008%). Paper-worthy honest documentation.

**Per-class split distribution (UNSW):**

| Class | Train | Calib | Test |
|---|---:|---:|---:|
| Normal | 44,800 | 11,200 | 37,000 |
| DoS | 9,811 | 2,453 | 4,089 |
| Probe | 9,993 | 2,498 | 4,173 |
| R2L | 42,658 | 10,665 | 17,777 |
| U2R | 1,010 | 253 | 422 |

**Verification:** ✓ All assertions pass.

#### CIC-IDS2017 v2 — `01_cic_data_exploration_v2.ipynb` (commit `cd6fe2c`)

| Split | Samples | Notes |
|---|---:|---|
| Train | 120,018 (60%) | 60% of 200K stratified subsample |
| Calibration | 40,006 (20%) | New 20% slice |
| Test | 40,007 (20%) | 20% slice (CIC has no official partition) |
| Features | 78 | All numerical, no categorical encoding needed |

**M1 not applicable:** CIC features are all numerical, no one-hot encoding step exists.

**Per-class split distribution (CIC):**

| Class | Train | Calib | Test |
|---|---:|---:|---:|
| Normal | 96,456 | 32,152 | 32,153 |
| DoS | 16,126 | 5,376 | 5,376 |
| Probe | 6,743 | 2,248 | 2,248 |
| R2L | 671 | 223 | 223 |
| U2R | 22 | 7 | 7 |

**Note:** CIC U2R is the smallest class across all datasets (7 calib / 7 test). Per-class F1 on this class will have very wide implicit error bars.

**Verification:** ✓ All assertions pass. ✓ Train + Calib + Test = exactly 200,031 (the post-cleaning subsample size).

### 2.3 Calibration sample-size landscape (paper reference table)

| Class | NSL calib | UNSW calib | CIC calib |
|---|---:|---:|---:|
| Normal | 13,469 | 11,200 | 32,152 |
| DoS | 9,186 | 2,453 | 5,376 |
| Probe | 2,331 | 2,498 | 2,248 |
| R2L | 199 | 10,665 | 223 |
| U2R | **10** | 253 | **7** |

**Implication for paper:** Per-class isotonic calibrator reliability differs across datasets. UNSW R2L and U2R are healthy. NSL U2R (n=10) and CIC U2R (n=7) are at the lower bound of meaningful isotonic fit and should be flagged as sample-size caveats in §16 of the v3 walkthrough.

---

## 3. Day 2 — Model training v2 across all three datasets

### 3.1 Design decisions (all three datasets)

| Decision | Rationale |
|---|---|
| 9 models per dataset (canonical 6 + 3 ablation) | Canonical pipeline (binary CW + 5-class SMOTE) plus class-weighted 5-class variant for justifying the SMOTE choice |
| RF max_depth=20 cap (new in v2) | v1 RF grew unconstrained to depth 46+ on UNSW, making TreeExplainer SHAP infeasible at 2000 samples. Cap eliminates this issue. |
| Identical hyperparameters to v1 | Only training data changes; methodology comparison stays clean |
| Save predictions on both test and calibration sets | Calibration probabilities feed Notebook 03 v2 isotonic fitting |
| Checkpoint metrics.json after every model | Resilience against Drive disconnects |

### 3.2 NSL-KDD v2 training — `02_train_models_v2.ipynb` (commit `aecaad2`)

**All 9 models:**

| Model | Accuracy | F1-macro | MCC | R2L F1 | U2R F1 |
|---|---:|---:|---:|---:|---:|
| rf_binary_cw | 0.7768 | 0.7760 | TBD | n/a | n/a |
| xgb_binary_cw | 0.7883 | 0.7878 | 0.6331 | n/a | n/a |
| dnn_binary_cw | **0.8106** | **0.8105** | 0.6652 | n/a | n/a |
| rf_5class_smote | 0.7474 | 0.5165 | TBD | TBD | TBD |
| xgb_5class_smote | **0.7869** | **0.6138** | **0.6920** | 0.1870 | **0.4130** |
| dnn_5class_smote | 0.7812 | 0.5747 | 0.6795 | 0.1941 | 0.2432 |
| rf_5class_cw | 0.7338 | 0.4683 | TBD | TBD | TBD |
| xgb_5class_cw | 0.7842 | 0.5948 | 0.6883 | 0.1318 | 0.3656 |
| dnn_5class_cw | 0.7716 | 0.5471 | 0.6637 | **0.2190** | 0.0935 |

*Note: TBD values for the 3 retrained NSL RF models will be populated from the updated metrics.json in the Day 4 documentation cascade. Accuracy and F1-macro are confirmed.*

**v1 vs v2 comparison (canonical 6):**

| Model | v1 acc/F1m | v2 acc/F1m | Δ acc | Δ F1m |
|---|---|---|---:|---:|
| rf_binary_cw | 0.765 / 0.764 | 0.777 / 0.776 | +0.012 | +0.012 |
| xgb_binary_cw | 0.794 / 0.793 | 0.788 / 0.788 | −0.006 | −0.005 |
| dnn_binary_cw | 0.790 / 0.790 | 0.811 / 0.811 | +0.021* | +0.021* |
| rf_5class_smote | 0.748 / 0.525 | 0.747 / 0.517 | −0.001 | −0.009 |
| xgb_5class_smote | 0.794 / 0.639 | 0.787 / 0.614 | −0.007 | −0.025 |
| dnn_5class_smote | 0.789 / 0.584 | 0.781 / 0.575 | −0.008 | −0.009 |

*\*DNN binary gain marked with asterisk: likely single-seed variance, not a v2 methodology benefit. Not to be claimed as a finding.*

**Notable:** Three NSL movements worth flagging in v2:

- *DNN binary (+2.1 pp v1→v2 acc and F1m):* Despite 20% less training data. Most likely single-seed variance; not to be claimed as a v2 methodology benefit.
- *RF binary (+1.2 pp v1→v2 acc/F1m):* The full v1→v2 delta combines (a) v2 preprocessing changes and (b) the max_depth=20 cap applied during the NSL RF retrain pass. Decomposing: the retrain alone moved RF binary from 0.7674 → 0.7768 acc (+0.9 pp). The remaining ~0.3 pp comes from the preprocessing changes (encoder fit on train only, etc.).
- *RF 5-class SMOTE (−0.8 pp v1→v2 F1m, but +0.6 pp pre→post-retrain F1m):* Net v1→v2 is a small drop (0.525 → 0.517 F1m), which is the expected behaviour given less training data. But the retrain alone *improved* F1m by 0.6 pp over the pre-retrain v2 model (0.5102 → 0.5165). So the depth cap is a positive contribution that partially offset the training-data dip.

The remaining 3 models (xgb_binary_cw, xgb_5class_smote, dnn_5class_smote) showed small expected dips (−0.005 to −0.025 macro-F1) from the smaller training set.

**Side-finding worth noting in the paper:** The max_depth=20 cap, originally motivated by TreeExplainer feasibility, also improved RF generalisation on NSL during the retrain pass. The pre-vs-post-retrain comparison (same v2 preprocessing, only the depth cap differs) showed +0.9 pp acc on rf_binary_cw, +0.6 pp F1m on rf_5class_smote. This is a clean controlled comparison of the cap's effect, isolated from preprocessing changes.

**NSL imbalance ablation:** SMOTE > CW on macro-F1 across all three architectures (+4.8 pp RF, +1.9 pp XGB, +2.8 pp DNN). Clean justification for canonical SMOTE choice. Nuance: DNN R2L F1 better with CW (0.219) than SMOTE (0.194), but aggregate macro-F1 favors SMOTE.

### 3.3 UNSW-NB15 v2 training — `02_unsw_train_models_v2.ipynb` (commit `8fb48a4`)

**All 9 models:**

| Model | Accuracy | F1-macro | MCC | AUC-ROC | R2L F1 | U2R F1 |
|---|---:|---:|---:|---:|---:|---:|
| rf_binary_cw | 0.8550 | 0.8547 | 0.7344 | 0.9719 | n/a | n/a |
| xgb_binary_cw | 0.8553 | 0.8550 | 0.7329 | **0.9738** | n/a | n/a |
| dnn_binary_cw | **0.8606** | **0.8599** | **0.7351** | 0.9677 | n/a | n/a |
| rf_5class_smote | 0.6872 | 0.5505 | 0.5297 | n/a | 0.5730 | 0.3105 |
| xgb_5class_smote | **0.7152** | **0.5911** | **0.5608** | n/a | **0.6060** | **0.4356** |
| dnn_5class_smote | 0.6348 | 0.5020 | 0.4751 | n/a | 0.5062 | 0.2233 |
| rf_5class_cw | 0.6961 | 0.5686 | 0.5376 | n/a | 0.5891 | 0.3789 |
| xgb_5class_cw | 0.7012 | 0.5775 | 0.5440 | n/a | 0.5874 | 0.4187 |
| dnn_5class_cw | 0.6017 | 0.4774 | 0.4453 | n/a | 0.4799 | 0.1558 |

**RF max_depth verification (UNSW):** rf_binary_cw, rf_5class_smote, rf_5class_cw all hit `mean=20.0, max=20`. The depth=20 cap was active on every tree. Without the cap, v1 RF on UNSW grew deeper than was practical for TreeExplainer at 2000 samples. **M10 SHAP-infeasibility issue permanently resolved for UNSW.**

**UNSW imbalance ablation (notable finding):**

| Architecture | CW macro-F1 | SMOTE macro-F1 | Winner |
|---|---:|---:|---|
| RF | **0.5686** | 0.5505 | **CW +1.8pp** |
| XGB | 0.5775 | **0.5911** | SMOTE +1.4pp |
| DNN | 0.4774 | **0.5020** | SMOTE +2.5pp |

**On UNSW, class-weighting beats SMOTE for RF.** This is different from NSL (where SMOTE won on all three) and is a paper-worthy honest finding.

### 3.4 CIC-IDS2017 v2 training — `02_cic_train_models_v2.ipynb` (commit `0e35f0d`)

**All 9 models:**

| Model | Accuracy | F1-macro | MCC | AUC-ROC | R2L F1 | U2R F1 |
|---|---:|---:|---:|---:|---:|---:|
| rf_binary_cw | 0.9985 | 0.9976 | 0.9952 | 0.9998 | n/a | n/a |
| xgb_binary_cw | **0.9990** | **0.9984** | **0.9968** | **0.9999** | n/a | n/a |
| dnn_binary_cw | 0.9694 | 0.9537 | 0.9101 | 0.9969 | n/a | n/a |
| rf_5class_smote | 0.9983 | 0.9571 | 0.9948 | n/a | 0.9602 | 0.8333 |
| xgb_5class_smote | **0.9989** | **0.9782** | **0.9966** | n/a | **0.9732** | 0.9231 |
| dnn_5class_smote | 0.9606 | 0.7265 | 0.8957 | n/a | 0.5366 | 0.2759 |
| rf_5class_cw | 0.9982 | 0.9584 | 0.9946 | n/a | 0.9680 | 0.8333 |
| xgb_5class_cw | 0.9990 | **0.9799** | 0.9970 | n/a | 0.9709 | **0.9333** |
| dnn_5class_cw | 0.8227 | 0.5400 | 0.6638 | n/a | 0.1003 | 0.0450 |

**RF max_depth verification (CIC):** rf_binary_cw, rf_5class_smote, rf_5class_cw all hit `mean=20.0, max=20`. The depth=20 cap was active on every tree across all three CIC RF models. The cap is now correctly applied to all three datasets.

**CIC imbalance ablation:**

| Architecture | CW macro-F1 | SMOTE macro-F1 | Winner |
|---|---:|---:|---|
| RF | 0.9584 | 0.9571 | tie (CW +0.13pp) |
| XGB | **0.9799** | 0.9782 | tie (CW +0.17pp) |
| DNN | 0.5400 | **0.7265** | **SMOTE +18.7pp** |

**DNN class-weighted collapsed on CIC** (dropped from 0.96 → 0.82 accuracy). Likely cause: CIC U2R has only 22 training samples, creating extreme class-weight ratios. Using sklearn's `compute_class_weight('balanced')` formula `n_total / (n_classes × n_class_i)` with n_total=120,018 and n_classes=5: Normal weight ≈ 0.25 (120,018 / (5 × 96,456)), U2R weight ≈ 1,091 (120,018 / (5 × 22)). That ~4,400× weight ratio destabilises DNN gradients on the 256-128-64-32 architecture. Early stopping kicked in after only 9 seconds of training, vs ~140 seconds for the SMOTE variant. Honest reporting: this is the most extreme imbalance-ablation gap across all three datasets.

### 3.5 Cross-dataset imbalance-strategy story (complete, paper-ready)

| Architecture | NSL | UNSW | CIC | Pattern |
|---|---|---|---|---|
| RF | SMOTE +4.8pp | **CW +1.8pp** | tie | dataset-dependent |
| XGB | SMOTE +1.9pp | SMOTE +1.4pp | tie | usually SMOTE |
| DNN | SMOTE +2.8pp | SMOTE +2.5pp | **SMOTE +18.7pp** | SMOTE always wins |

**Clean paper framing:** *"DNN consistently and substantially benefits from SMOTE oversampling (+2.5 to +18.7 pp macro-F1). Tree models (RF, XGB) show dataset-dependent behaviour where SMOTE and class-weighting trade off. We adopt SMOTE as the canonical strategy for the 5-class task because (a) it never significantly hurts tree-model performance, and (b) it is consistently and sometimes dramatically better for the DNN."*

---

## 4. Architectural ranking (preserved from v1)

The headline architectural ranking on macro-F1 holds in v2:

| Dataset | Best (5-class) | Second | Third |
|---|---|---|---|
| NSL-KDD | XGB (0.6138) | DNN (0.5747) | RF (0.5165) |
| UNSW-NB15 | XGB (0.5911) | RF (0.5505) | DNN (0.5020) |
| CIC-IDS2017 | XGB (0.9782) | RF (0.9571) | DNN (0.7265) |

**XGB is best on all three datasets** (matches v1). RF vs DNN ordering is dataset-dependent: NSL favours DNN, UNSW and CIC favour RF. This is the corrected ranking statement; earlier versions of the cross-dataset comparison doc (v4, v5) incorrectly claimed a universal XGB > RF > DNN ordering. v6/v7 corrected this; v8 will inherit the corrected statement when we rebuild the comparison doc on Day 4.

---

## 5. Methodology issues addressed (with verification)

| Issue | v1 problem | v2 fix | Verification |
|---|---|---|---|
| **M1** | One-hot encoder fit on train+test union | Fit on train only; test-only categories produce all-zero columns | ✓ NSL: zero rows affected. UNSW: 5 rows affected (0.008%). CIC: not applicable (no categorical features). |
| **M6** | Calibrator fit on 50/50 split of test set | Calibration set carved from training data (20% of train) | ✓ All three datasets: train/calib disjoint, official test untouched (NSL/UNSW), 60/20/20 stratified (CIC). |
| **M14** | Conformal threshold fit on test set (downstream of M6) | Will be fit on calibration holdout in Notebook 03 v2 | Pending Day 3. Architectural fix already in place; Notebook 03 v2 will implement. |
| **M10** | RF unconstrained depth on UNSW made TreeExplainer infeasible | RF max_depth=20 cap applied to all three datasets | ✓ Verified on all three datasets: NSL (mean=20.0, max=20 after retrain pass), UNSW (mean=20.0, max=20), CIC (mean=20.0, max=20). NSL retrain was a corrective pass after the initial Day 2 training omitted the cap. |

**Other potential issues that turned out to be safe:**

| Issue | Status | Verification |
|---|---|---|
| M2 (StandardScaler) | safe | Fit on train only in all three datasets — confirmed by code search |
| M4 (SMOTE timing) | safe | Applied after train/test split in all three datasets — confirmed by code search |
| M7 (per-class isotonic formulation) | safe | Standard one-vs-rest `(P[c], 1[y=c])` — confirmed |
| M9 (DNN SHAP background) | safe | Drawn from training data, stratified by 5-class — confirmed |

**Issues still pending (Day 3+ work):**

| Issue | Plan |
|---|---|
| M8/M12 (SHAP comparability across architectures) | Documentation framing in walkthrough v6 §15.2 |
| M11 (perturbation ε in scaled space) | Documentation note in §15.4 |
| M13 (same k, different feature space fraction) | Add "% of features" column to Krishna tables |
| M15 (geometric mean justification) | 2-paragraph justification in §15.6 |
| M19 (missing bootstrap CIs) | Implement in Day 3 |
| M20 (dataset independence framing) | Rephrase in v6 narrative |
| M16/M17 (LLM-as-judge issues) | Supervisor meeting decision point |

---

## 6. What's on disk

### Repository state (`main` branch)

| Commit | Description |
|---|---|
| `0c5b7e1` | Notebook 01 v2 (NSL preprocessing) |
| `516e3c6` | Notebook 01-UNSW v2 (UNSW preprocessing) |
| `cd6fe2c` | Notebook 01-CIC v2 (CIC preprocessing) |
| `aecaad2` | Notebook 02 v2 (NSL training, 9 models — RF without depth cap) |
| `8fb48a4` | Notebook 02-UNSW v2 (UNSW training, 9 models) |
| `0e35f0d` | Notebook 02-CIC v2 (CIC training, 9 models) |
| `558a998` | Notebook 02-NSL-RF-retrain v2 (M10 cap fix for NSL RF) |

### Drive artefacts (not committed to git)

```
data/processed/nsl_kdd_v2/      — 16 files (X/y arrays, indices, metadata,
                                  oov_summary.json, split_methodology.json,
                                  scaler.pkl)
data/processed/unsw_nb15_v2/    — 16 files (same structure, scaler.joblib
                                  rather than .pkl)
data/processed/cic_ids2017_v2/  — 16 files (no oov_summary; CIC has no
                                  categorical features)

models/nsl_kdd_v2/              — 9 models (.pkl / .pt)
                                 + predictions/ with 36 files
                                   (test_pred, test_proba, calib_pred,
                                    calib_proba per model)
                                 + metrics.json
models/unsw_nb15_v2/            — 9 models (.joblib / .pt)
                                 + predictions/ with 18 files (pred only)
                                 + probabilities/ with 18 files (proba only)
                                 + metrics.json
models/cic_ids2017_v2/          — 9 models (.joblib / .pt)
                                 + predictions/ with 18 files (pred only)
                                 + probabilities/ with 18 files (proba only)
                                 + metrics.json
```

Note: NSL keeps predictions and probabilities together in `predictions/` with `.pkl` for RF/XGB. UNSW and CIC split predictions/probabilities into separate folders and use `.joblib` for RF/XGB. Total predictions+probabilities per dataset: 36 files in all cases.

### Results CSVs (committed to git)

- `results/tables/nslkdd_v2_model_comparison.csv`
- `results/tables/nslkdd_v2_imbalance_ablation.csv`
- `results/tables/unsw_v2_model_comparison.csv`
- `results/tables/unsw_v2_imbalance_ablation.csv`
- `results/tables/cic_v2_model_comparison.csv`
- `results/tables/cic_v2_imbalance_ablation.csv`

---

## 7. What remains (Day 3 + Day 4)

### Day 3 — Calibration, SHAP, stability, agreement, SCTS-v2

| Notebook | Per-dataset effort | × datasets | Cumulative |
|---|---:|---|---:|
| 03 v2 (isotonic calibration on calib set, evaluate on test) | 15 min build + 5 min run | × 3 | ~1 hour |
| 04 v2 (SHAP at full 2000 samples now possible everywhere) | 20 min build + 15–30 min run | × 3 | ~2.5 hours |
| 05 v2 (stability — FGSM/PGD/Gaussian) | 15 min build + 10 min run | × 3 | ~1.5 hours |
| 06 v2 (Krishna agreement metrics) | 15 min build + 5 min run | × 3 | ~1 hour |
| 07 v2 (SCTS-v2 on NSL) + 08 v2 (SCTS-v2 on UNSW) | 30 min build + 10 min run | × 2 | ~1 hour |
| Bootstrap CIs on Krishna metrics | one-off script | | ~1 hour |
| Optional: 08-LLM v2 on NSL | depends on supervisor decision | | 0–2 hours |

**Day 3 total: ~8 hours of work without the optional LLM piece, or 8–10 hours with it. Compute time runs in the background.**

### Day 4 — Documentation cascade

| Document | Effort |
|---|---:|
| Walkthrough v6 (regenerate all numbers from v2 CSVs) | 2 hours |
| Cross-dataset comparison v8 | 1 hour |
| UNSW findings docs (SHAP, stability, agreement, SCTS-v2, design decisions) | 1 hour |
| §16 limitations update (acknowledge what v2 fixed, what remains) | 30 min |
| Re-audit pass against v2 numbers | 2 hours |
| Buffer for unexpected issues | 1.5 hours |

**Day 4 total: ~8 hours of writing and auditing.**

---

## 8. Findings preserved for paper

### What v2 confirmed
1. XGBoost remains the best architecture for 5-class on all three datasets (preserved from v1)
2. The SMOTE-vs-CW comparison has a clean and honest cross-dataset pattern: DNN strongly prefers SMOTE; tree models are dataset-dependent
3. NSL OOV impact: zero (M1 was technically a leak but produced no practical leakage on NSL)
4. UNSW OOV impact: 5 test rows (0.008%) — documented honestly
5. RF max_depth=20 cap successfully applied across all three datasets (NSL applied via retrain after initial Day 2 omission)
6. **Unexpected positive side-finding (from the NSL RF retrain):** The max_depth=20 cap was originally motivated by TreeExplainer feasibility. The NSL retrain pass (same preprocessing, only the cap added) showed it also improves RF generalisation: rf_binary_cw +0.9 pp acc, rf_5class_smote +0.6 pp F1m. The depth cap is a positive contribution that partially offsets the training-data reduction from the v2 80/20 split. This is a clean controlled comparison (pre-retrain v2 vs post-retrain v2, identical preprocessing) worth highlighting in the paper.

### What v2 changed honestly
1. NSL macro-F1 slipped by ~1–2.5 pp on 5-class (expected: 20% less training data)
2. UNSW v2 numbers landed in the range we expected (within 1–2 pp of v1 patterns where comparison is possible); exact v1-vs-v2 deltas pending walkthrough v6 cascade
3. CIC v2 numbers also landed in expected range; full v1-vs-v2 comparison pending walkthrough v6 cascade
4. The DNN binary improvement on NSL (+2.1 pp) is most likely seed variance, not a v2 benefit — should not be claimed

### Cross-dataset story (paper headline)
- Three datasets validated for calibration architecture (pending Day 3 numbers)
- Two datasets validated for SCTS-v2 (pending Day 3 numbers)
- XGB-best architectural finding holds in v2
- DNN-SMOTE strong preference is a new finding worth a paragraph in discussion
- UNSW DoS hardest-to-trust class finding will be re-verified once SCTS-v2 v2 runs

---

## 9. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| Day 1 start | Budget A (fix everything, all three datasets, 3–4 days) | User decision over Budget B (NSL+UNSW only) and Budget C (document as limitation). Reviewer-bulletproof methodology trumps schedule comfort. |
| Day 1 start | Approach 1 (single 80/20 train/calibration split) over 5-fold CV | Standard for IDS papers, simpler, what reviewers expect |
| Day 1 NSL | Confirmed M1 had zero practical impact on NSL | Train and test had identical categorical value sets |
| Day 1 UNSW | OOV finding documented honestly | 5 rows out of 63,461 is real but tiny |
| Day 1 CIC | 60/20/20 design over 80/20 (carving calib from train) | CIC has no official partition, so we create one cleanly |
| Day 2 start | 9 models per dataset (canonical 6 + 3 ablation) over 6 (no ablation) or 12 (full ablation) | Bulletproof but not bloated; ablation justifies SMOTE choice without redundancy |
| Day 2 start | RF max_depth=20 cap | Addresses M10 / pre-submission task on UNSW tree depth |
| Day 2 NSL | NSL DNN binary +2.1 pp on v2 is seed variance, not a finding | Single-seed limitation already documented |
| Day 2 UNSW | RF on UNSW prefers CW (+1.8 pp) over SMOTE | Honest finding, paper-worthy nuance |
| Day 2 CIC | DNN class-weighted collapsed on CIC (-18.7 pp vs SMOTE) | Extreme class-weight ratio on tiny U2R training set; well-documented |
| Day 2 audit | NSL RF retrain with max_depth=20 cap | Verification script discovered NSL v2 RF trained without the cap (mean depth 28–31, max 40–45). Retrained the 3 NSL RF models to align with UNSW/CIC. Side effect: NSL RF performance improved slightly through reduced overfitting. |

---

## 10. Drive disconnect resilience notes

Multiple Drive disconnects occurred during this work. Mitigations that worked:

1. **`checkpoint_metrics()` after every model**: even with 4-5 disconnects across the day, no model results were lost — each was on disk before the next started
2. **`force_remount=True`**: recovers most disconnects without full runtime restart
3. **Restoring git config from Drive**: necessary after every runtime reset
4. **Notebook structure with cells §1-§3 as setup/load**: allows resuming mid-pipeline by re-running setup then jumping to the specific cell

Pattern: long Colab sessions (>2 hours) become unstable. Keeping fewer concurrent notebook tabs open helps.

---

## 11. Status before Day 3

✓ Day 1+2 complete across all three datasets
✓ 27 v2 models trained (9 × 3 datasets)
✓ 108 prediction files saved (9 models × 4 prediction types × 3 datasets)
✓ 6 paper-ready CSV tables on GitHub
✓ Methodology fixes verified
✓ Honest cross-dataset findings preserved

**Ready to begin Day 3 (calibration, SHAP, stability, agreement, SCTS-v2).**
