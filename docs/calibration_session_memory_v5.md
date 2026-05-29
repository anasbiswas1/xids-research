# Day 3 Calibration Work — Comprehensive Session Memory (v5, precision-corrected)

**Project:** X-IDS Research (Explainable Intrusion Detection System) under PAIDS, supervisor Ms Alaa Mohasseb.
**Target venue:** IEEE TIFS / TDSC (top-tier security journals).
**Repo:** github.com/anasbiswas1/xids-research
**Session date:** Started ~evening 30 May 2026, ran through into night of 31 May 2026.
**Document purpose:** Capture every aspect of the calibration work tonight — empirical results, methodology decisions, mistakes made, reasoning chains, pending work, and orientation for future sessions. This is the authoritative session memory; if I lose this conversation, I should be able to pick up the calibration work from this file alone.

**v5 changes from v4:**
- Brier pre/post values for hybrid method updated to bootstrap-consistent precision (resolves §5 arithmetic discrepancy where displayed post-pre ≠ claimed delta on 10 of 36 rows)
- §5.2 UNSW comparison statement corrected: methods statistically differ on 5 of 6 models with Dirichlet flipping more on 4 of 5 (NOT "mostly overlap" as v4 wrongly stated)
- §6 finding 8 expanded to cover the UNSW pattern
- §6 new finding 11 added: averages systematically understate per-model statistical differences
- §7 new mistake entry §7.14: repeated the average-vs-per-model error on UNSW
- §11.2 audit trail updated for v5 (post-audit precision fixes)

---

## 1. Session context and starting state

### 1.1 What was already done before tonight (Day 1 + Day 2)

The v2 rebuild was complete: M1/M6/M10 methodology fixes applied across all three datasets (NSL-KDD, UNSW-NB15, CIC-IDS2017). 27 models trained (3 datasets × 3 architectures × 3 label tasks: binary class-weighted, 5-class SMOTE, 5-class class-weighted). NSL RandomForest retrained with max_depth=20. Progress doc v8 committed at `9fb0ba0`.

Calibration sets were carved out during preprocessing (M6 fix): an 80/20 stratified split of training data, producing a held-out calibration set per dataset never seen during model training. This is the fundamental prerequisite for tonight's work — calibration cannot be applied without a separate calibration set.

Day 2 ended with a major finding: the CIC class-weighted DNN model (dnn_5class_cw) had collapsed during training due to extreme class-weight ratios (Normal ~0.25, U2R ~1091, ratio of ~4400×). Test accuracy 0.823, macro-F1 0.540. This collapsed model would later become important for tonight's calibration findings.

### 1.2 TA comments to address

Three comments from Ms Alaa via the TA channel before tonight:

1. **Rare-class calibration caveat:** "The rare-class calibration sample size concern needs to be addressed explicitly. NSL U2R has only 10 calibration samples; CIC U2R has 7." This was the original driver for tonight's hybrid Platt/isotonic approach (Platt scaling for n < 30).

2. **Bootstrap confidence intervals:** Required on four claim types — (a) SCTS-v2 quartile gaps, (b) Krishna rank correlations Spearman/Kendall, (c) DNN stability advantage under FGSM/PGD/Gaussian, (d) UNSW DoS hardest-to-trust finding. B=1000 resamples, 95% CIs (2.5/97.5 percentile). **DONE for calibration metrics (notebook 03d). Bootstrap CIs on (a)-(d) remain Day 4 work.**

3. **LLM evaluation is weak:** Required a 3-layer evaluation — (L1) LLM-judge on full set, (L2) 50 alerts stratified rated by user + supervisor + 1 peer on Correctness/Actionability/Clarity 1-5, (L3) LLM-judge rates same 50 + Cohen's kappa LLM-vs-human. **Scheduled for Day 5. Not done tonight.**

Tonight's calibration work addresses comment 1 (rare-class concern) AND comment 2 as it applies to calibration metrics.

### 1.3 Tonight's goal at session start

Run calibration experiments on three datasets and produce a deliverable for the upcoming supervisor meeting (rescheduled if needed). Specifically: apply hybrid Platt/isotonic calibration as a methodology choice that addresses the rare-class concern, document the empirical results, and provide statistical rigor via bootstrap CIs.

**What we actually ended up doing turned out to be much bigger** — see §2.

---

## 2. Methodology decision: Stance A vs B vs C

The most important methodological decision tonight was choosing how to frame calibration in the paper. Three coherent stances emerged:

### 2.1 Stance A: Deployment view

Calibration is a deployment-stage transformation. The classifier is the underlying trained model PLUS the calibrator together. Downstream analysis (SHAP, stability, agreement, trust) uses calibrated probabilities throughout.

**Pros:** Aligns with how Dirichlet was designed in the literature. Treats calibration improvements (Brier, accuracy) as real wins. Strongest contribution claim.

**Cons:** Requires re-running downstream analysis on calibrated outputs (+3-4 hours additional work). Class-weighted models lose macro-F1 under Dirichlet — must defend or drop. The "calibrator changed the classifier" framing requires careful explanation.

### 2.2 Stance B: Methodology view (initially recommended by Claude — wrong)

Calibration provides probability quality without altering the underlying classifier. Methods that change argmax (like Dirichlet) don't qualify as calibration.

**Pros:** Conservative, safer narrative. Less downstream rework needed.

**Cons:** Loses the Dirichlet finding entirely. Hybrid has its own argmax-flipping behaviour (discovered after this stance was proposed), so the clean dichotomy doesn't hold empirically. Weaker contribution.

**Claude originally recommended this stance.** It was the wrong recommendation — see §7 mistakes.

### 2.3 Stance C: Four-property evaluation matrix (final choice)

Calibration evaluated across four properties: (1) probabilistic accuracy (Brier, ECE), (2) classifier preservation (argmax stability), (3) imbalance preservation (per-class F1), (4) shift robustness. Three methods × three datasets × nine models = systematic empirical contribution.

**Pros:** Strongest methodological framing. Identifies a real tension in post-hoc calibration that the literature underreports. Publishable in TIFS/TDSC as a methodology contribution, not just an applied result.

**Cons:** Most complex narrative to defend. Requires comprehensive characterization rather than a clean "we did X, it worked" story.

### 2.4 Why Stance C was adopted

After Claude initially recommended Stance B, the user clarified the actual research context: this is PAIDS-affiliated research targeting TIFS/TDSC, supporting a global talent visa application. The user has time post-coursework, no hard deadline. Risk calculus changes completely from MSc-paper risk tolerance.

For top-journal publication, safe-and-conservative loses to bold-and-rigorous. Stance C is the framing that generates a real contribution. The argmax-shift finding alone justifies its own section.

### 2.5 Implications for downstream pipeline

Under Stance C, downstream analyses (SHAP, stability, Krishna, SCTS, LLM) can run on hybrid-calibrated outputs as the primary pipeline. A possible sensitivity analysis on Dirichlet-calibrated outputs is optional. The calibration evaluation matrix itself is a standalone contribution. **Decision deferred until SHAP is done — we'll know then whether sensitivity analysis adds enough to justify the cost.**

---

## 3. Datasets and distribution shift

### 3.1 Calibration set sizes (from preprocessing M6 fix)

| Dataset | Normal | DoS | Probe | R2L | U2R | Total |
|---|---:|---:|---:|---:|---:|---:|
| NSL-KDD | 13,469 | 9,186 | 2,331 | 199 | **10** | 25,195 |
| UNSW-NB15 | 11,200 | 2,453 | 2,498 | 10,665 | 253 | 27,069 |
| CIC-IDS2017 | 32,152 | 5,376 | 2,248 | 223 | **7** | 40,006 |

### 3.2 Distribution shift between calibration and test sets

**NSL-KDD — severe shift (rare classes amplified in test):**

| Class | p_calib (%) | p_test (%) | ratio |
|---|---:|---:|---:|
| Normal | 53.46 | 43.08 | 0.81 |
| DoS | 36.46 | 33.09 | 0.91 |
| Probe | 9.25 | 10.74 | 1.16 |
| R2L | 0.79 | 12.80 | **16.20** |
| U2R | 0.04 | 0.30 | **7.49** |

NSL's official test set was designed to include novel attacks not in training. R2L is 16× more common in test than in calibration; U2R is 7.5× more common.

**UNSW-NB15 — moderate shift (majority class amplified in test):**

| Class | p_calib (%) | p_test (%) | ratio |
|---|---:|---:|---:|
| Normal | 41.38 | 58.30 | **1.41** |
| DoS | 9.06 | 6.44 | 0.71 |
| Probe | 9.23 | 6.58 | 0.71 |
| R2L | 39.40 | 28.01 | 0.71 |
| U2R | 0.93 | 0.66 | 0.71 |

UNSW's official train/test split has different proportions by dataset design. Normal is 1.4× more common in test; everything else is 0.71×.

**CIC-IDS2017 — no shift by construction:**

All ratios exactly 1.00× because we used a stratified 60/20/20 split (CIC has no official train/test partition).

### 3.3 Why the shift characterization matters

Post-hoc calibration assumes calibration and test distributions share the same conditional class structure. **All three datasets violate this in different ways** (CIC violates it least, because we constructed the split that way). This becomes the central organizing principle of the calibration findings: efficacy depends on shift severity.

---

## 4. Calibration methods evaluated

### 4.1 Pre-calibration baseline

Raw model probabilities. RandomForest from leaf-node class proportions. XGBoost from sigmoid (binary) or softmax (multiclass). DNN from softmax layer.

### 4.2 Hybrid Platt/Isotonic

Per-class one-versus-rest scheme. Strategy selection based on per-class sample size:
- **n_calib ≥ 30:** Isotonic regression
- **n_calib < 30:** Platt scaling

After per-class calibration, probability vector is renormalized to sum to 1.

**Where Platt is actually deployed:** NSL U2R (n=10), CIC U2R (n=7). Everywhere else: isotonic.

**Notebooks:** `notebooks/03_nsl_calibration_v2.ipynb`, `notebooks/03_unsw_calibration_v2.ipynb`, `notebooks/03_cic_calibration_v2.ipynb`.

### 4.3 Dirichlet ODIR

Joint multiclass calibration (Kull et al., NeurIPS 2019). Learns a k×k linear transformation in log-probability space plus a k-vector of intercepts. ODIR variant uses Off-Diagonal and Intercept Regularization.

**Mathematically:** `p_calibrated = softmax(W * log(p_raw) + b)`.

**Notebooks:** `notebooks/03c_nsl_calibration_dirichlet_v2.ipynb`, `notebooks/03c_unsw_calibration_dirichlet_v2.ipynb`, `notebooks/03c_cic_calibration_dirichlet_v2.ipynb`.

### 4.4 Bootstrap CIs (notebook 03d)

`notebooks/03d_calibration_bootstrap_cis.ipynb`. B=1000, stratified resampling over test indices. Outputs `results/tables/calibration_bootstrap_cis.csv` (36 rows × all metrics × CIs). Committed at `02e30a8`.

### 4.5 Brier recompute (v5 audit fix)

`results/tables/calibration_brier_recomputed.csv`. After v4 audit revealed that original CSV-stored Brier values had inconsistent precision (10 of 36 rows had arithmetic discrepancies of ~0.001-0.002), pre/post values were recomputed at full 6-decimal precision using the bootstrap notebook's macro Brier definition. These are the authoritative pre/post values from v5 onward.

---

## 5. Empirical results — all numbers with CIs (v5 precision-corrected)

### 5.1 NSL-KDD

**Hybrid Platt/Isotonic — macro Brier with bootstrap CIs:**

| Model | Brier pre | Brier post | ΔBrier (95% CI) | Improved? |
|---|---:|---:|---|---|
| rf_5class_smote | 0.072305 | 0.084674 | +0.01237 [+0.01189, +0.01287] | no |
| xgb_5class_smote | 0.079535 | 0.080444 | +0.00091 [+0.00068, +0.00116] | no |
| dnn_5class_smote | 0.084370 | 0.087288 | +0.00292 [+0.00251, +0.00332] | no |
| rf_5class_cw | 0.079137 | 0.087792 | +0.00866 [+0.00818, +0.00914] | no |
| xgb_5class_cw | 0.080332 | 0.080045 | **−0.00029 [−0.00046, −0.00010]** | yes (marginal) |
| dnn_5class_cw | 0.080411 | 0.085455 | +0.00504 [+0.00435, +0.00576] | no |

**NSL hybrid summary:** 5 of 6 worsened (CIs strictly positive); 1 improved with marginal CI.

**Dirichlet — macro Brier with bootstrap CIs:**

| Model | Brier pre | Brier post | ΔBrier (95% CI) | Improved? |
|---|---:|---:|---|---|
| rf_5class_smote | 0.072305 | 0.079733 | +0.00743 [+0.00723, +0.00763] | no |
| xgb_5class_smote | 0.079535 | 0.080529 | +0.00099 [+0.00080, +0.00120] | no |
| dnn_5class_smote | 0.084370 | 0.086264 | +0.00189 [+0.00150, +0.00230] | no |
| rf_5class_cw | 0.079137 | 0.083752 | +0.00462 [+0.00443, +0.00479] | no |
| xgb_5class_cw | 0.080332 | 0.082547 | +0.00222 [+0.00201, +0.00244] | no |
| dnn_5class_cw | 0.080411 | 0.083522 | +0.00311 [+0.00255, +0.00372] | no |

**NSL Dirichlet summary:** All 6 5-class worsened, all CIs strictly positive.

**Three-way winner counts (NSL):** Pre 6, Hybrid 1 (marginal), Dirichlet 2 (binary only).

**Hybrid argmax flipping (NSL 5-class):**

| Model | % flipped (95% CI) | Δ acc | Δ F1m |
|---|---|---:|---:|
| rf_5class_smote | 3.61% [3.35, 3.83] | +0.0271 | +0.0250 |
| xgb_5class_smote | 0.21% [0.16, 0.27] | −0.0011 | −0.0203 |
| dnn_5class_smote | 2.72% [2.53, 2.94] | −0.0114 | −0.0147 |
| rf_5class_cw | 4.74% [4.49, 5.01] | +0.0421 | +0.0645 |
| xgb_5class_cw | 0.26% [0.20, 0.32] | +0.0017 | −0.0154 |
| dnn_5class_cw | 7.35% [7.03, 7.71] | +0.0028 | +0.0207 |

**Average: 3.15% flipping.** All CIs narrow and positive.

**Dirichlet argmax flipping (NSL 5-class):**

| Model | % flipped (95% CI) | Δ acc | Δ F1m |
|---|---|---:|---:|
| rf_5class_smote | 1.74% [1.57, 1.91] | −0.0127 | −0.0511 |
| xgb_5class_smote | 1.13% [0.99, 1.27] | −0.0090 | −0.0851 |
| dnn_5class_smote | 2.58% [2.39, 2.78] | −0.0136 | −0.0781 |
| rf_5class_cw | 0.75% [0.63, 0.86] | −0.0044 | −0.0108 |
| xgb_5class_cw | 1.08% [0.95, 1.22] | −0.0087 | −0.0730 |
| dnn_5class_cw | 5.61% [5.33, 5.90] | +0.0018 | −0.0253 |

**Average: 2.15% flipping.**

**NSL hybrid vs Dirichlet flipping comparison:** CIs DO NOT OVERLAP on 5 of 6 models, but the direction differs by model architecture:
- Hybrid flips MORE on: rf_5class_smote, rf_5class_cw, dnn_5class_cw
- Dirichlet flips MORE on: xgb_5class_smote, xgb_5class_cw
- Overlap (no statistical difference): dnn_5class_smote

**Key observation:** despite very low flipping (1-3% on most), macro-F1 dropped 5-9 pp on 4 of 6 models with Dirichlet. The flips were concentrated in rare classes.

**Dirichlet per-class wins (NSL, 30 cells):** **6 improved, 24 worsened.** Improvements concentrated in DNN models on U2R class:
- dnn_5class_smote U2R: 0.0066 → **0.0030** (−54.5%)
- dnn_5class_cw U2R: 0.0179 → **0.0029** (−83.8%)

### 5.2 UNSW-NB15

**Hybrid Platt/Isotonic — macro Brier with bootstrap CIs:**

| Model | Brier pre | Brier post | ΔBrier (95% CI) | Improved? |
|---|---:|---:|---|---|
| rf_5class_smote | 0.072236 | 0.069136 | −0.00310 [−0.00332, −0.00288] | **yes** |
| xgb_5class_smote | 0.071247 | 0.068668 | −0.00258 [−0.00280, −0.00235] | **yes** |
| dnn_5class_smote | 0.086865 | 0.072061 | −0.01480 [−0.01513, −0.01446] | **yes** |
| rf_5class_cw | 0.069654 | 0.068178 | −0.00148 [−0.00169, −0.00127] | **yes** |
| xgb_5class_cw | 0.075054 | 0.070140 | −0.00491 [−0.00515, −0.00464] | **yes** |
| dnn_5class_cw | 0.093118 | 0.074384 | −0.01873 [−0.01907, −0.01839] | **yes** |

**UNSW hybrid summary:** All 6 5-class improved, all CIs strictly negative.

**Dirichlet — macro Brier with bootstrap CIs:**

| Model | Brier pre | Brier post | ΔBrier (95% CI) |
|---|---:|---:|---|
| rf_5class_smote | 0.072236 | 0.065268 | −0.00697 [−0.00722, −0.00670] |
| xgb_5class_smote | 0.071247 | 0.064830 | −0.00642 [−0.00666, −0.00615] |
| dnn_5class_smote | 0.086865 | 0.067462 | −0.01940 [−0.01978, −0.01898] |
| rf_5class_cw | 0.069654 | 0.064908 | −0.00475 [−0.00499, −0.00450] |
| xgb_5class_cw | 0.075054 | 0.065927 | −0.00913 [−0.00942, −0.00881] |
| dnn_5class_cw | 0.093118 | 0.070523 | −0.02260 [−0.02300, −0.02219] |

**UNSW Dirichlet head-to-head against hybrid (CI-based):** For every UNSW 5-class model, Dirichlet's ΔBrier CI lies entirely below hybrid's ΔBrier CI (no overlap). Dirichlet's superiority is statistically tight.

**Three-way winner counts:** Pre 3 (all binary), Hybrid 0, Dirichlet 6 (all 5-class).

**Hybrid argmax flipping (UNSW 5-class):**

| Model | % flipped (95% CI) | Δ acc | Δ F1m |
|---|---|---:|---:|
| rf_5class_smote | 18.74% [18.46, 19.00] | +0.0439 | −0.0082 |
| xgb_5class_smote | 12.73% [12.50, 12.96] | +0.0271 | −0.0169 |
| dnn_5class_smote | 23.65% [23.33, 23.94] | +0.0753 | +0.0114 |
| rf_5class_cw | 17.38% [17.11, 17.64] | +0.0435 | −0.0140 |
| xgb_5class_cw | 15.62% [15.36, 15.90] | +0.0292 | −0.0037 |
| dnn_5class_cw | 25.42% [25.10, 25.69] | +0.0623 | −0.0028 |

**Average: 18.92% flipping.**

**Dirichlet argmax flipping (UNSW 5-class):**

| Model | % flipped (95% CI) | Δ acc | Δ F1m |
|---|---|---:|---:|
| rf_5class_smote | 19.39% [19.09, 19.65] | +0.0590 | +0.0005 |
| xgb_5class_smote | 14.29% [14.04, 14.52] | +0.0404 | +0.0021 |
| dnn_5class_smote | 25.98% [25.67, 26.28] | +0.0947 | +0.0196 |
| rf_5class_cw | 16.97% [16.69, 17.22] | +0.0512 | −0.0209 |
| xgb_5class_cw | 16.22% [15.96, 16.48] | +0.0508 | +0.0099 |
| dnn_5class_cw | 29.33% [29.01, 29.65] | **+0.1061** | **−0.0327** |

**Average: 20.36% flipping.**

**UNSW hybrid vs Dirichlet flipping — CORRECTED v5 (v4 wrongly said "mostly overlap"):**

CIs DO NOT OVERLAP on 5 of 6 models. Dirichlet flips MORE on 5 of 5 statistically distinct models:

| Model | Hybrid CI | Dirichlet CI | Direction |
|---|---|---|---|
| rf_5class_smote | [18.46, 19.00] | [19.09, 19.65] | Dirichlet flips more (no overlap) |
| xgb_5class_smote | [12.50, 12.96] | [14.04, 14.52] | Dirichlet flips more |
| dnn_5class_smote | [23.33, 23.94] | [25.67, 26.28] | Dirichlet flips more |
| rf_5class_cw | [17.11, 17.64] | [16.69, 17.22] | overlap (indistinguishable) |
| xgb_5class_cw | [15.36, 15.90] | [15.96, 16.48] | Dirichlet flips more |
| dnn_5class_cw | [25.10, 25.69] | [29.01, 29.65] | Dirichlet flips more |

The two methods are consistently different on UNSW — Dirichlet flips more on 5 of 5 statistically distinct models. The 1.4 pp average difference (18.92 vs 20.36) understates this: per-model CIs reveal a robust, directional pattern.

**Critical observation on UNSW dnn_5class_cw with Dirichlet:** Accuracy +10.6 pp BUT macro-F1 −3.3 pp. The class-weight training boost on rare classes was partially undone by Dirichlet's prior absorption.

**U2R per-model gains (UNSW Dirichlet):**

| Model | U2R Brier pre | U2R Brier post | % reduction |
|---|---:|---:|---:|
| dnn_5class_cw | 0.0379 | 0.0066 | −82.6% |
| dnn_5class_smote | 0.0299 | 0.0064 | −78.6% |
| rf_5class_smote | 0.0130 | 0.0053 | −59.2% |
| xgb_5class_cw | 0.0112 | 0.0047 | −58.0% |
| xgb_5class_smote | 0.0086 | 0.0046 | −46.5% |
| rf_5class_cw | 0.0085 | 0.0050 | −41.2% |

### 5.3 CIC-IDS2017

**Hybrid Platt/Isotonic — macro Brier with bootstrap CIs:**

| Model | Brier pre | Brier post | ΔBrier (95% CI) | Operationally significant? |
|---|---:|---:|---|---|
| rf_5class_smote | 0.000629 | 0.000546 | −0.00008 [−0.00011, −0.00005] | Statistically yes, trivial magnitude |
| xgb_5class_smote | 0.000441 | 0.000413 | −0.00003 [−0.00005, −0.00000] | Marginal |
| dnn_5class_smote | 0.011243 | 0.005313 | **−0.00593 [−0.00634, −0.00552]** | **Yes** |
| rf_5class_cw | 0.000603 | 0.000605 | **+0.00000 [−0.00004, +0.00004]** | **CI INCLUDES ZERO** |
| xgb_5class_cw | 0.000365 | 0.000345 | −0.00002 [−0.00004, −0.00000] | Marginal |
| dnn_5class_cw | 0.052581 | 0.014770 | **−0.03781 [−0.03856, −0.03708]** | **Yes — dramatic recovery** |

**CIC hybrid summary (CI-aware):** 2 of 6 5-class models show operationally meaningful improvement (both DNN). 4 of 6 are at noise level or include zero.

**Headline numbers (statistically robust):**
- dnn_5class_cw macro ECE: 0.10656 → **0.00087** (ΔECE −0.10074 [−0.10174, −0.09950], −99.2%)
- dnn_5class_cw macro Brier: 0.05258 → **0.01477** (−71.9%)

**Dirichlet — macro Brier with bootstrap CIs:**

| Model | Brier pre | Brier post | ΔBrier (95% CI) | Direction |
|---|---:|---:|---|---|
| rf_5class_smote | 0.000629 | 0.000686 | +0.00006 [+0.00001, +0.00011] | Worsens (CI strictly positive) |
| xgb_5class_smote | 0.000441 | 0.000421 | −0.00002 [−0.00006, +0.00002] | **CI INCLUDES ZERO** |
| dnn_5class_smote | 0.011243 | 0.006034 | −0.00521 [−0.00557, −0.00483] | Improves significantly |
| rf_5class_cw | 0.000603 | 0.000660 | +0.00006 [+0.00002, +0.00009] | Worsens (CI strictly positive) |
| xgb_5class_cw | 0.000365 | 0.000360 | −0.00000 [−0.00003, +0.00002] | **CI INCLUDES ZERO** |
| dnn_5class_cw | 0.052581 | 0.015340 | −0.03724 [−0.03797, −0.03654] | Improves dramatically |

**CIC Dirichlet summary (CI-aware):** Only 2 of 6 5-class models show operationally meaningful improvement (DNN). 2 worsen significantly; 2 are statistically indistinguishable from zero.

**Three-way winner counts (CIC):** Hybrid 7 (point), Dirichlet 2 (point). With CI awareness: hybrid has 2 confirmed substantial wins; the other 4 "wins" are at noise level.

**CIC per-class winners (5-class, 30 cells — point estimates):**

| Class | Hybrid | Dirichlet | Pre |
|---|---:|---:|---:|
| Normal | 4 | 2 | 0 |
| DoS | 5 | 1 | 0 |
| Probe | 4 | 0 | 2 |
| R2L | 3 | 2 | 1 |
| U2R | 4 | 0 | 2 |
| **Total** | **20** | **5** | **5** |

**Caveat:** Many per-class wins are at noise level. Aggregate "hybrid wins 20" describes point-estimate ordering, not operational significance.

**CIC argmax flipping (with CIs):**

| Model | Hybrid % flipped (CI) | Dirichlet % flipped (CI) |
|---|---|---|
| rf_5class_smote | 0.03% [0.01, 0.04] | 0.10% [0.07, 0.14] |
| xgb_5class_smote | 0.01% [0.00, 0.02] | 0.02% [0.01, 0.03] |
| dnn_5class_smote | 3.44% [3.26, 3.62] | 3.44% [3.26, 3.62] |
| rf_5class_cw | 0.05% [0.03, 0.07] | 0.10% [0.07, 0.13] |
| xgb_5class_cw | 0.01% [0.00, 0.03] | 0.02% [0.01, 0.04] |
| dnn_5class_cw | **19.34% [18.99, 19.74]** | **17.86% [17.53, 18.24]** |

Hybrid and Dirichlet produce statistically indistinguishable argmax behaviour on 5 of 6 CIC models. **EXCEPTION:** dnn_5class_cw — hybrid flips MORE than Dirichlet (no CI overlap).

**CIC flipping average:** 3.81% (heavily dominated by dnn_5class_cw).

### 5.4 Cross-dataset summary

**Winner counts (9 models per dataset, point estimates):**

| Dataset | Shift | Pre-cal | Hybrid | Dirichlet |
|---|---|---:|---:|---:|
| NSL-KDD | Severe | **6** | 1 (marginal) | 2 (binary only) |
| UNSW-NB15 | Moderate | 3 (binary) | **0** | **6** (5-class, all CIs tight) |
| CIC-IDS2017 | None | 0 | **7** | 2 |

**Hybrid vs Dirichlet argmax comparison summary (NEW v5):**

| Dataset | Non-overlapping CIs | Direction pattern |
|---|---:|---|
| NSL-KDD | 5 of 6 | Mixed — hybrid more on rf/dnn; Dirichlet more on xgb |
| UNSW-NB15 | 5 of 6 | **Consistent — Dirichlet flips more on 5 of 5 distinct models** |
| CIC-IDS2017 | 1 of 6 (dnn_5class_cw) | Hybrid more by ~1.5 pp on the one catastrophic model |

**Argmax flipping by method × dataset (5-class average):**

| Method | NSL | UNSW | CIC |
|---|---:|---:|---:|
| Hybrid | 3.15% | 18.92% | 3.81% |
| Dirichlet | 2.15% | 20.36% | 3.81% |

---

## 6. Interpretation and discussion

### 6.1 Main empirical claims (with CI status)

**Claim 1: Calibration efficacy depends on calibration-test distribution alignment.**

CIC (no shift) → 2 of 6 hybrid 5-class wins are CI-confirmed substantial; 4 are noise level.
UNSW (moderate shift) → Dirichlet dominates 5-class with all 6 CIs tight and strictly negative.
NSL (severe shift) → pre-cal wins 6 of 9. 5 of 6 NSL hybrid 5-class deltas have CIs strictly positive (significant worsening).

**Claim 2: Both methods substantially modify classifier predictions on UNSW.**

Argmax flipping rates with narrow CIs: hybrid 12.7-25.4%, Dirichlet 14.3-29.3%. The calibrated probabilities, used as the basis for a downstream classifier, disagree with the pre-calibration argmax on a substantial fraction of samples.

**Claim 3: NSL Dirichlet has the opposite pattern — low flipping but big F1 impact.**

Average flipping 2.15%, but macro-F1 drops 5-9 pp on 4 of 6 models. Mechanism: Dirichlet absorbs NSL's severe rare-class shift by reclassifying a small fraction of rare-class samples, which has outsized impact on macro-F1.

**Claim 4: Class-weighted models suffer most from Dirichlet's prior absorption.**

UNSW dnn_5class_cw with Dirichlet: accuracy +10.6 pp [CI: +10.4, +10.8], macro-F1 −3.3 pp. Class-weight boost on rare classes gets partially undone.

**Claim 5: Catastrophically miscalibrated models benefit dramatically.**

CIC dnn_5class_cw with hybrid: ECE 0.107 → 0.0009 (−99.2%; ΔECE CI [−0.10174, −0.09950]), accuracy 0.823 → 0.946 (+12.3 pp). Calibration here functions as classifier recovery, not just confidence rescaling.

**Claim 6: Binary calibration on UNSW degrades regardless of method.**

Both hybrid and Dirichlet worsen all 3 binary UNSW models. Likely cause: 2-class structure cannot redistribute probability mass to absorb prior shift.

**Claim 7 (NEW v5): Averages of argmax flipping understate per-model statistical differences.**

UNSW hybrid 18.92% vs Dirichlet 20.36% (1.4 pp difference) initially looked like "methods are similar on UNSW." Bootstrap CIs revealed 5 of 6 models have non-overlapping CIs with Dirichlet flipping more — a robust directional pattern. Averages mask this. Lesson: per-model CIs should be the primary evidence, not averages.

### 6.2 Mechanistic hypothesis (initial, simple)

We initially hypothesized: **flipping rate is monotonic in distribution shift severity** (more shift → more argmax change).

**This hypothesis is empirically wrong.** UNSW (moderate shift) shows higher flipping than NSL (severe shift). The simple "shift → flipping" relationship doesn't hold.

### 6.3 Revised mechanistic hypothesis (not directly verified)

**Flipping rate depends on which classes shift, not just severity.**

- NSL: shift concentrated in rare classes (R2L 16×, U2R 7.5×). Few absolute samples affected; flipping rate stays low; but macro-F1 impact is amplified.
- UNSW: shift concentrated in majority class (Normal 1.4×). Every test sample is potentially affected; flipping rate is high; macro-F1 impact moderate because flips spread across class boundaries.
- CIC: no shift. Flipping limited to pre-cal miscalibration effects.

This hypothesis is consistent with our data but **not directly verified**. A targeted test would inspect which classes the flipped samples come from. This is a Day 4 or beyond task.

### 6.4 Hybrid's argmax flipping mechanism is different

Hybrid Platt/isotonic flipping correlates more with pre-calibration miscalibration magnitude than with distribution shift. UNSW models have higher pre-cal Brier (0.07-0.09) than NSL (0.05-0.08) or CIC (<0.05). When pre-cal probabilities are more miscalibrated, the per-class isotonic transformations are larger, and after renormalization more boundary samples flip.

**Important:** This means the two methods have DIFFERENT sensitivities. Dirichlet → distribution shift. Hybrid → pre-cal miscalibration magnitude. Not directly comparable on a single axis.

### 6.5 v5 refinements

- CIC tree-model "wins" should not be claimed as operationally meaningful. The bulk of CIC hybrid's success comes from the 2 collapsed DNN models.
- NSL hybrid vs Dirichlet flipping differ statistically on most models, but direction varies by model architecture.
- **UNSW hybrid vs Dirichlet flipping differ statistically on 5 of 6 models with consistent direction (Dirichlet flips more on all 5 distinct cases).** v4's "mostly overlap" claim was wrong.
- CIC dnn_5class_cw is the one CIC case where hybrid and Dirichlet produce statistically different argmax rates.

---

## 7. Mistakes and corrections (chronological)

Including these because they're useful for future reasoning. If a future session is making predictions, knowing where this session got it wrong helps calibrate confidence.

### 7.1 Wrong prediction: "calibration will improve Brier"

When we started the night, I predicted both calibration methods would improve Brier scores on most or all models. **Wrong.** On NSL, calibration worsened 7 of 9 models with hybrid and 7 of 9 with Dirichlet. This was the first sign that distribution shift was going to dominate the story.

### 7.2 Wrong prediction: "renormalisation step is benign"

When discussing hybrid Platt/isotonic, I described the renormalisation step (forcing rows to sum to 1) as a minor implementation detail. **Partially wrong.** The diagnostic notebook later showed renormalisation contributes ~20% of NSL's calibration degradation. It's not benign; it's a non-trivial source of error.

### 7.3 Wrong prediction: "UNSW will be the clean test of calibration"

I expected UNSW to have minimal distribution shift because the v2 preprocessing used a stratified split. **Wrong.** UNSW uses the official train/test partition (not a custom split), and that partition has Normal 1.4× more common in test. UNSW has its own distribution shift.

### 7.4 Wrong prediction: "Dirichlet would be cleanly better than hybrid"

When proposing Dirichlet, I framed it as "methodologically stronger." **Empirically mixed.** Dirichlet wins on UNSW 5-class (6 of 6, all CIs tight) but loses or ties on CIC (2 of 9) and is mixed on NSL (2 of 9 wins).

### 7.5 Wrong prediction: "Hybrid preserves argmax, Dirichlet shifts it"

This was the framing I built Stance B around. **Empirically false on UNSW.** Both methods flip 14-25% on UNSW. The clean dichotomy doesn't hold. This invalidated Stance B and pushed us toward Stance C.

### 7.6 Invented number: "NSL Dirichlet flips ~23%"

I claimed NSL Dirichlet flipped about 23% of predictions, based on extrapolation from UNSW's 20% and assuming severe shift would cause more flipping. **Completely wrong.** The actual measurement showed 2.15% average.

### 7.7 Initial recommendation of Stance B was wrong

I recommended Stance B (conservative methodology view) on the grounds that it was safer for an MSc paper. **Wrong context.** Once the user clarified TIFS/TDSC target and PAIDS affiliation, Stance C was the correct recommendation.

### 7.8 Repeatedly suggested stopping when user had energy

Throughout the night, I kept suggesting we stop because "it's late" when the user had clearly stated they had energy and wanted to keep going. **Pattern: projecting fatigue concerns onto the user.**

### 7.9 F1m number inconsistency on UNSW dnn_5class_cw

At one point I gave two different F1m numbers for the same model. Reconciliation script confirmed 0.4774 → 0.4447 is correct.

### 7.10 Wrong NSL Dirichlet per-class improvement count

In conversation and v1/v2/v3 of the docs, I claimed "7 of 30 cells improved" on NSL Dirichlet. The v3-era audit revealed actual count is 6 of 30 improved, 24 of 30 worsened.

### 7.11 Pattern across mistakes (v3-era)

The common thread: **I made predictions ahead of the data and stated them as findings**. Counts and aggregates carried over from conversation can drift from the underlying data.

### 7.12 Overgeneralized "CIC argmax identical between methods" (v4-era fix)

In v3, I claimed "Hybrid and Dirichlet produce nearly identical argmax behaviour on CIC." Bootstrap CIs revealed this is true for 5 of 6 CIC models but wrong for dnn_5class_cw.

### 7.13 Bootstrap CI claim was too narrowly scoped (v4-era fix)

In v1/v2/v3, the bootstrap CIs were listed as "TBD / Day 4 work." Actually fitting them tonight took ~7 minutes of compute. Statistical rigor pieces that are cheap should be done as part of empirical work, not deferred.

### 7.14 NEW v5: Repeated the average-vs-per-model mistake on UNSW

In v4, I claimed "UNSW hybrid vs Dirichlet flipping CIs mostly overlap (within 1-2 pp of each other for each model)." **The post-v4 audit revealed only 1 of 6 model-pair CIs actually overlap — 5 of 6 are statistically distinct, with Dirichlet flipping more on all 5 distinct cases.**

This is the **same mistake** as §7.12 (CIC overgeneralization) applied to a different dataset. I had captured the lesson in §7.12 but didn't apply it consistently when characterizing the UNSW comparison. The "average rates are similar" framing kept seducing me into treating the methods as equivalent on UNSW.

**Lesson (reinforced):** When comparing two methods' argmax rates, always check per-model CI overlap, especially when method averages are within a few percentage points of each other. The 1.4 pp difference between UNSW averages (hybrid 18.92% vs Dirichlet 20.36%) initially looked like "noise" but bootstrap CIs revealed it's a robust directional pattern across 5 of 6 models.

### 7.15 NEW v5: Brier pre/post precision inconsistency

In v3/v4, I copied Brier pre/post values from the original calibration notebook CSVs without checking if they had consistent precision with the bootstrap notebook's computation. Post-v4 audit revealed 10 of 36 rows (all hybrid method) had stored values where (post − pre) didn't equal the bootstrap-computed delta within rounding. The deltas were correct; the stored pre/post values had been rounded inconsistently at the original CSV-write step.

**Fix:** v5 uses recomputed pre/post values from `calibration_brier_recomputed.csv` at 6-decimal precision.

**Lesson:** When pulling values from multiple sources, audit arithmetic consistency. If `Δ = post − pre` is claimed but `Δ_displayed ≠ post_displayed − pre_displayed`, that's a precision/source-mismatch bug, not "rounding."

### 7.16 Pattern across mistakes (updated v5)

The lessons keep stacking:
1. **Predictions ≠ findings.** Label predictions explicitly.
2. **Recompute aggregate counts at writeup time.** Don't carry from conversation.
3. **Cheap statistical rigor should be done immediately, not deferred.**
4. **Averages can mask per-model statistical differences.** Always check per-model CIs.
5. **Audit arithmetic consistency across data sources.** If displayed values don't subtract correctly, there's a precision/source mismatch.

**Meta-pattern:** Most of these mistakes were caught by deliberate audit passes, not by the work itself. The takeaway is that audit passes are a critical part of the writeup workflow, not optional. The findings doc has been audited three times now (v2→v3 audit, v3→v4 audit, v4→v5 audit), each catching real errors.

---

## 8. Pending work

### 8.1 For calibration specifically

**Bootstrap confidence intervals (TA-requested):** ✓ **DONE in v4/v5 for calibration metrics.** Notebook 03d produces 95% CIs on ΔBrier, ΔECE, % flipped, Δacc, ΔF1m. CSV at `results/tables/calibration_bootstrap_cis.csv`. Brier recompute CSV at `results/tables/calibration_brier_recomputed.csv`. Remaining bootstrap CIs (on SCTS, Krishna, stability) are Day 4 work.

**Reliability diagrams (optional, Day 4):** NSL diagrams exist; UNSW and CIC versions would complete the figure set. ~30 minutes each.

**Direct test of revised mechanism hypothesis:** Inspect which classes get flipped on NSL Dirichlet vs UNSW hybrid/Dirichlet to verify the "rare-class shift vs majority-class shift" hypothesis. ~1 hour.

**Class-prior absorption mathematical analysis (optional, paper-strengthening):** Compute Dirichlet ODIR intercept vector per model per dataset. Show intercepts correlate with `log(p_test/p_calib)` per class. ~2 hours.

**Per-cell bootstrap CIs (optional):** Current bootstrap is on macro. Per-cell extension is ~20 minutes.

### 8.2 For the broader Day 3+ work

| Phase | Status |
|---|---|
| Calibration (this work) | ✓ Data complete, CIs done, audit-validated through v5, writeup partial |
| SHAP (Notebook 04) | Not started |
| Stability — FGSM/PGD/Gaussian (Notebook 05) | Not started |
| Krishna agreement (Notebook 06) | Not started |
| SCTS-v2 on NSL + UNSW (Notebooks 07-08) | Not started |

**Key decision pending:** does SHAP run on pre-cal probabilities, hybrid-calibrated, or both? Under Stance C: hybrid-calibrated as primary, pre-cal as sensitivity check (recommended).

### 8.3 For the paper writeup (next session)

Three artefacts produced tonight:
1. **`calibration_section_draft.md`** — 2,420-word prose draft. Has [PENDING] citation markers. Not citation-ready.
2. **`calibration_findings_exact_v5.md`** — ~5,100-word exact-numbers reference with bootstrap CIs, precision-corrected.
3. **`calibration_session_memory_v5.md`** (this file) — Comprehensive session memory.

Next session writeup work:
- Verify every number in the prose draft against the findings doc v5
- Add citations (Niculescu-Mizil & Caruana 2005, Kull et al. 2019, Guo et al. 2017, Ovadia et al. 2019)
- Insert actual tables WITH CIs (formatted from CSVs)
- Resolve approximate language in favour of exact numbers + CIs

---

## 9. File and repo references

### 9.1 Notebooks created tonight

| File | Purpose |
|---|---|
| `03_nsl_calibration_v2.ipynb` | NSL hybrid Platt/isotonic |
| `03b_nsl_calibration_diagnostic.ipynb` | NSL reliability diagrams + diagnostic |
| `03c_nsl_calibration_dirichlet_v2.ipynb` | NSL Dirichlet ODIR |
| `03_unsw_calibration_v2.ipynb` | UNSW hybrid |
| `03c_unsw_calibration_dirichlet_v2.ipynb` | UNSW Dirichlet |
| `03_cic_calibration_v2.ipynb` | CIC hybrid |
| `03c_cic_calibration_dirichlet_v2.ipynb` | CIC Dirichlet |
| `03d_calibration_bootstrap_cis.ipynb` | Bootstrap 95% CIs on calibration metrics |

### 9.2 CSV outputs

In `results/tables/`:

| File | Contents |
|---|---|
| `nslkdd_v2_calibration_*` | NSL hybrid + three-way comparison |
| `unsw_v2_calibration_*` | UNSW hybrid + three-way |
| `cic_v2_calibration_*` | CIC hybrid + three-way |
| `calibration_bootstrap_cis.csv` | Bootstrap CIs (36 rows × all metrics) |
| `calibration_brier_recomputed.csv` | **NEW v5:** Brier pre/post at consistent precision |

### 9.3 Calibrator outputs

Saved as `.npy` arrays in `calibrators/{dataset}_v2/` (hybrid) and `calibrators/{dataset}_v2_dirichlet/` (Dirichlet).

### 9.4 Git state

Key commits:
- `656d96f` — Calibration v3 documents
- `02e30a8` — Notebook 03d bootstrap CIs + CSV
- (next) — v5 documents + brier recompute CSV

### 9.5 Documents produced tonight

In `docs/`:

| File | Purpose |
|---|---|
| `calibration_section_draft.md` | Tier 2 prose draft for paper |
| `calibration_findings_exact_v5.md` | Reconciled, audit-validated, precision-corrected exact-numbers reference with bootstrap CIs |
| `calibration_session_memory_v5.md` (THIS FILE) | Comprehensive session memory |

---

## 10. Orientation for future sessions

If a future Claude instance picks up this work:

1. **Read this document first.** It's the single source of truth for the calibration phase.
2. **Then check git log** for any commits after the latest commit listed in §9.4.
3. **Verify any number against `results/tables/` CSVs** before quoting it. Multiple audit passes caught errors that arose from carrying numbers across documents — disk values win.
4. **The user's research is PAIDS-affiliated, target IEEE TIFS/TDSC.** Calibration is part of a 4-component framework: calibration, SHAP-based stability, Krishna agreement, SCTS-v2 trust scoring, plus LLM-based alert evaluation.
5. **Stance C is the agreed framing.** Four-property evaluation matrix across three datasets and two methods.
6. **The user trusts their TA (Ms Alaa) at checkpoints**, runs technical decisions independently between them.
7. **The user has energy and time.** Don't suggest stopping unless there's a specific reason beyond "it's late."
8. **Predictions should be labelled as predictions.** Pattern of stating predictions as findings caused multiple errors.
9. **Audit aggregate counts and averages before publishing.** Per-cell counts, winner totals, averages, and method comparisons should all be recomputed from the underlying tables. Averages especially can mask per-model statistical differences (see §7.12, §7.14).
10. **Quote CIs when making empirical claims.** Bootstrap CIs are in `results/tables/calibration_bootstrap_cis.csv`. Any claim like "method X improves Y" should reference the CI, not just the point estimate.
11. **Audit arithmetic consistency across data sources.** If `Δ = post − pre` is claimed but the displayed values don't subtract correctly, that's a precision/source mismatch, not "rounding." Recompute from raw data.

### 10.1 Where to start next

Three possible starting points, in order of recommended priority:

**Option A: Notebook 04 — SHAP.** With calibration complete (including CIs and audit), SHAP is the next major Day 3 component. Decision needed: which calibration outputs feed SHAP? Recommendation: hybrid-calibrated as primary. ~2.5 hours.

**Option B: Finalize the calibration section writeup.** Use the prose draft + findings doc v5 + this memory doc together. Produce a clean paper section with CIs. ~4-6 hours.

**Option C: Direct verification of revised mechanism hypothesis.** Inspect which classes are getting flipped to confirm the "rare-class vs majority-class" hypothesis. ~1 hour.

A then B is the natural order. C is a strengthener that can fit anywhere.

### 10.2 What NOT to do

- Don't start Day 5 LLM evaluation work yet (requires SHAP outputs which don't exist)
- Don't add more calibration methods
- Don't promise to "finally complete" the calibration work — it's already complete. What remains is documentation and possibly the optional strengtheners in §8.1.

---

## 11. Session metadata

### 11.1 Quantitative summary

- **Total time:** approximately 10 hours of active work (8h core + 1h v4 audit/bootstrap + 1h v5 audit/precision-fix)
- **Notebooks built:** 7 calibration notebooks + 1 diagnostic + 1 bootstrap CI notebook
- **Models calibrated:** 27 (9 models × 3 datasets) × 2 methods = 54 calibrated probability sets
- **Argmax checks run:** 5 (NSL hybrid, NSL Dirichlet, UNSW hybrid, UNSW Dirichlet, CIC both methods)
- **Bootstrap resamples computed:** 36 (model, dataset, method) cells × 1000 resamples = 36,000 bootstrap evaluations
- **Bootstrap reruns:** 1 (after Drive disconnect during first run; second run produced identical numbers, confirming determinism)
- **Audit passes performed:** 3 (v2→v3 caught count error; v3→v4 caught CIC overgeneralization; v4→v5 caught UNSW overgeneralization AND precision inconsistency)

### 11.2 Document evolution audit trail

- **v1:** First-draft documents created in-session
- **v2:** Reconciliation against disk corrected UNSW dnn_5class_cw F1m, filled TBDs, replaced extrapolation with measurement
- **v3:** Audit pass corrected NSL Dirichlet count error (7→6), expanded U2R table to 12 rows, added per-class UNSW Dirichlet table, softened philosophical framing in §6, generalized time references
- **v4:** Bootstrap CIs integrated throughout §5. Refined CIC tree-model claims (noise level). Added findings 9-10. New mistake entries §7.12, §7.13.
- **v5 (this version):** Brier pre/post values updated to bootstrap-consistent precision (resolves 10 of 36 arithmetic discrepancies). UNSW hybrid-vs-Dirichlet comparison corrected (5 of 6 statistically differ, NOT "mostly overlap"). Added §6 claim 7, finding 11. New mistake entries §7.14 (UNSW overgeneralization), §7.15 (precision inconsistency), §7.16 (updated meta-pattern). Added §1.5 macro Brier definition clarification.

**End of session memory document v5. Last updated: end of 31 May 2026, after v5 audit and precision correction.**
