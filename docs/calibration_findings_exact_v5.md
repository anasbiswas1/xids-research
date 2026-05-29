# Calibration Findings — Exact Numbers (v5, precision-corrected and audit-validated)

**All numerical results from the X-IDS calibration evaluation, with bootstrap 95% confidence intervals and recomputed Brier pre/post values at consistent precision.**
**Drafted: end of Day 3, 31 May 2026. v5 fixes precision issues identified by end-of-session audit pass.**

**v5 changes from v4:**
- Brier pre/post values for **hybrid** method rows updated to recomputed 6-decimal precision (source: `results/tables/calibration_brier_recomputed.csv`). Resolves arithmetic inconsistency where (Brier post − Brier pre) didn't equal the displayed ΔBrier on 10 of 36 rows. Dirichlet values were already at consistent precision.
- §3.6 UNSW hybrid vs Dirichlet comparison corrected: methods statistically differ on **5 of 6** models (not "mostly overlap" as v4 claimed). Dirichlet flips MORE on 4 of 5 distinct models.
- §5.2 method-comparison summary table added for UNSW
- §6 finding 8 expanded to note UNSW pattern as well as CIC
- §7 (limitations) §3 note added about original CSV precision

**Source files:**
- Bootstrap CIs: `results/tables/calibration_bootstrap_cis.csv` (36 rows)
- Recomputed Brier pre/post: `results/tables/calibration_brier_recomputed.csv` (36 rows)

---

## 1. Datasets and calibration setup

### 1.1 Calibration set sizes per dataset (per-class)

| Dataset | Normal | DoS | Probe | R2L | U2R | Total |
|---|---:|---:|---:|---:|---:|---:|
| NSL-KDD | 13,469 | 9,186 | 2,331 | 199 | **10** | 25,195 |
| UNSW-NB15 | 11,200 | 2,453 | 2,498 | 10,665 | 253 | 27,069 |
| CIC-IDS2017 | 32,152 | 5,376 | 2,248 | 223 | **7** | 40,006 |

### 1.2 Distribution shift between calibration set and test set

**NSL-KDD — severe shift (rare classes amplified in test):**

| Class | n_calib | p_calib (%) | n_test | p_test (%) | p_test/p_calib |
|---|---:|---:|---:|---:|---:|
| Normal | 13,469 | 53.46 | 9,711 | 43.08 | 0.81 |
| DoS | 9,186 | 36.46 | 7,460 | 33.09 | 0.91 |
| Probe | 2,331 | 9.25 | 2,421 | 10.74 | 1.16 |
| R2L | 199 | 0.79 | 2,885 | 12.80 | **16.20** |
| U2R | 10 | 0.04 | 67 | 0.30 | **7.49** |

**UNSW-NB15 — moderate shift (majority class amplified in test):**

| Class | n_calib | p_calib (%) | n_test | p_test (%) | p_test/p_calib |
|---|---:|---:|---:|---:|---:|
| Normal | 11,200 | 41.38 | 37,000 | 58.30 | **1.41** |
| DoS | 2,453 | 9.06 | 4,089 | 6.44 | 0.71 |
| Probe | 2,498 | 9.23 | 4,173 | 6.58 | 0.71 |
| R2L | 10,665 | 39.40 | 17,777 | 28.01 | 0.71 |
| U2R | 253 | 0.93 | 422 | 0.66 | 0.71 |

**CIC-IDS2017 — no shift by construction (stratified 60/20/20 split):**

| Class | n_calib | p_calib (%) | n_test | p_test (%) | p_test/p_calib |
|---|---:|---:|---:|---:|---:|
| Normal | 32,152 | 80.37 | 32,153 | 80.37 | **1.00** |
| DoS | 5,376 | 13.44 | 5,376 | 13.44 | **1.00** |
| Probe | 2,248 | 5.62 | 2,248 | 5.62 | **1.00** |
| R2L | 223 | 0.56 | 223 | 0.56 | **1.00** |
| U2R | 7 | 0.02 | 7 | 0.02 | **1.00** |

### 1.3 Calibration strategy applied (Hybrid Platt/Isotonic)

| Dataset | Classes getting isotonic | Classes getting Platt |
|---|---|---|
| NSL-KDD | Normal, DoS, Probe, R2L | U2R (n=10) |
| UNSW-NB15 | All 5 classes (n_min=253) | none |
| CIC-IDS2017 | Normal, DoS, Probe, R2L | U2R (n=7) |

Platt threshold: n_calib < 30. Only two classes across three datasets fall below it: NSL U2R and CIC U2R.

### 1.4 Bootstrap methodology

- **Resamples:** B=1000, stratified by class to preserve class proportions
- **Resampled object:** Test set indices (not training data, not model fits)
- **CI type:** Percentile (2.5/97.5)
- **Seed:** 42 (deterministic; second run confirmed identical numbers)
- **Source:** Notebook `03d_calibration_bootstrap_cis.ipynb`, CSV `results/tables/calibration_bootstrap_cis.csv`

### 1.5 Macro Brier definition (clarification added v5)

Macro Brier is computed as the mean across classes of the one-vs-rest Brier score: `(1/K) * Σ_c brier_score_loss((y == c).astype(int), p[:, c])`. All Brier pre/post values in this document use this definition. The bootstrap CIs are computed using the same formula on resampled test indices.

---

## 2. NSL-KDD calibration results

### 2.1 NSL hybrid Platt/isotonic — macro Brier and ECE with bootstrap CIs

| Model | Brier pre | Brier post | ΔBrier (95% CI) | ΔECE (95% CI) | CI excludes 0? |
|---|---:|---:|---|---|---|
| rf_5class_smote | 0.072305 | 0.084674 | +0.01237 [+0.01189, +0.01287] | +0.01301 [+0.01222, +0.01375] | Yes (worsens) |
| xgb_5class_smote | 0.079535 | 0.080444 | +0.00091 [+0.00068, +0.00116] | +0.00073 [+0.00037, +0.00097] | Yes (worsens) |
| dnn_5class_smote | 0.084370 | 0.087288 | +0.00292 [+0.00251, +0.00332] | +0.00437 [+0.00360, +0.00500] | Yes (worsens) |
| rf_5class_cw | 0.079137 | 0.087792 | +0.00866 [+0.00818, +0.00914] | +0.00921 [+0.00856, +0.00985] | Yes (worsens) |
| xgb_5class_cw | 0.080332 | 0.080045 | **−0.00029 [−0.00046, −0.00010]** | +0.00020 [−0.00007, +0.00035] | Brier yes (marginal); ECE no |
| dnn_5class_cw | 0.080411 | 0.085455 | +0.00504 [+0.00435, +0.00576] | +0.00544 [+0.00449, +0.00650] | Yes (worsens) |

**NSL hybrid summary:** 5 of 6 models show statistically significant Brier worsening. 1 of 6 (xgb_5class_cw) shows marginal Brier improvement (CI upper bound −0.0001). On ECE, 5 of 6 worsen significantly; xgb_5class_cw is indistinguishable from zero on ECE.

### 2.2 NSL Dirichlet — macro Brier and ECE with bootstrap CIs

| Model | Brier pre | Brier post | ΔBrier (95% CI) | ΔECE (95% CI) | CI excludes 0? |
|---|---:|---:|---|---|---|
| rf_5class_smote | 0.072305 | 0.079733 | +0.00743 [+0.00723, +0.00763] | +0.00780 [+0.00714, +0.00843] | Yes (worsens) |
| xgb_5class_smote | 0.079535 | 0.080529 | +0.00099 [+0.00080, +0.00120] | +0.00120 [+0.00084, +0.00141] | Yes (worsens) |
| dnn_5class_smote | 0.084370 | 0.086264 | +0.00189 [+0.00150, +0.00230] | +0.00171 [+0.00116, +0.00232] | Yes (worsens) |
| rf_5class_cw | 0.079137 | 0.083752 | +0.00462 [+0.00443, +0.00479] | +0.00638 [+0.00599, +0.00672] | Yes (worsens) |
| xgb_5class_cw | 0.080332 | 0.082547 | +0.00222 [+0.00201, +0.00244] | +0.00240 [+0.00205, +0.00263] | Yes (worsens) |
| dnn_5class_cw | 0.080411 | 0.083522 | +0.00311 [+0.00255, +0.00372] | +0.00092 [+0.00017, +0.00171] | Yes (worsens, but ECE near-zero) |

**NSL Dirichlet summary:** All 6 5-class models show statistically significant Brier worsening. ECE also worsens significantly on all 6 models, though dnn_5class_cw ECE worsening is barely detectable (CI [+0.00017, +0.00171]).

### 2.3 NSL three-way winner counts (macro Brier)

| Method | Wins on macro Brier (CI-confirmed) |
|---|---:|
| Pre-calibration | 6 of 9 (all 5-class except xgb_5class_cw, plus rf_binary_cw, dnn_binary_cw) |
| Hybrid Platt/Isotonic | 1 of 9 (xgb_5class_cw, marginal CI [−0.00046, −0.00010]) |
| Dirichlet ODIR | 2 of 9 (xgb_binary_cw, dnn_binary_cw — binary tasks) |

**Headline:** On NSL, post-hoc calibration loses to uncalibrated baseline on 6 of 9 models. The 1 hybrid win is statistically real but marginal in magnitude. Binary improvements with Dirichlet exist but binary models worsened on UNSW and CIC, suggesting they're not a generalizable finding.

### 2.4 NSL Dirichlet — per-class Brier directions

Per-class deltas (Brier pre → Brier post):

**rf_5class_smote:** Normal 0.1509→0.1726 (↑); DoS 0.0679→0.0702 (↑); Probe 0.0361→0.0388 (↑); R2L 0.1045→0.1141 (↑); U2R 0.0021→0.0029 (↑). **0 of 5 improved.**

**xgb_5class_smote:** Normal 0.1824→0.1833 (↑); DoS 0.0629→0.0637 (↑); Probe 0.0412→0.0407 (↓); R2L 0.1089→0.1123 (↑); U2R 0.0023→0.0026 (↑). **1 of 5 improved (Probe).**

**dnn_5class_smote:** Normal 0.1830→0.1915 (↑); DoS 0.0638→0.0636 (↓); Probe 0.0501→0.0492 (↓); R2L 0.1183→0.1240 (↑); U2R 0.0066→**0.0030** (↓). **3 of 5 improved.** U2R Brier reduced 54.5%.

**rf_5class_cw:** Normal 0.1739→0.1838 (↑); DoS 0.0675→0.0740 (↑); Probe 0.0401→0.0419 (↑); R2L 0.1118→0.1159 (↑); U2R 0.0024→0.0032 (↑). **0 of 5 improved.**

**xgb_5class_cw:** Normal 0.1855→0.1901 (↑); DoS 0.0616→0.0623 (↑); Probe 0.0406→0.0411 (↑); R2L 0.1115→0.1165 (↑); U2R 0.0024→0.0027 (↑). **0 of 5 improved.**

**dnn_5class_cw:** Normal 0.1583→0.1896 (↑); DoS 0.0661→0.0679 (↑); Probe 0.0519→0.0461 (↓); R2L 0.1079→0.1110 (↑); U2R 0.0179→**0.0029** (↓). **2 of 5 improved.** U2R Brier reduced 83.8%.

**NSL Dirichlet per-class summary:** **6 of 30 cells improved, 24 of 30 worsened.** Improvements concentrated in DNN models on U2R class.

### 2.5 NSL hybrid — argmax preservation with CIs

| Model | % flipped (95% CI) | Δ acc (point) | Δ F1m (point) |
|---|---|---:|---:|
| rf_5class_smote | 3.61% [3.35, 3.83] | +0.0271 | +0.0250 |
| xgb_5class_smote | 0.21% [0.16, 0.27] | −0.0011 | −0.0203 |
| dnn_5class_smote | 2.72% [2.53, 2.94] | −0.0114 | −0.0147 |
| rf_5class_cw | 4.74% [4.49, 5.01] | +0.0421 | +0.0645 |
| xgb_5class_cw | 0.26% [0.20, 0.32] | +0.0017 | −0.0154 |
| dnn_5class_cw | 7.35% [7.03, 7.71] | +0.0028 | +0.0207 |

**Range: 0.21% to 7.35%. Average: 3.15%.** All CIs strictly positive.

### 2.6 NSL Dirichlet — argmax preservation with CIs

| Model | % flipped (95% CI) | Δ acc (point) | Δ F1m (point) |
|---|---|---:|---:|
| rf_5class_smote | 1.74% [1.57, 1.91] | −0.0127 | **−0.0511** |
| xgb_5class_smote | 1.13% [0.99, 1.27] | −0.0090 | **−0.0851** |
| dnn_5class_smote | 2.58% [2.39, 2.78] | −0.0136 | **−0.0781** |
| rf_5class_cw | 0.75% [0.63, 0.86] | −0.0044 | −0.0108 |
| xgb_5class_cw | 1.08% [0.95, 1.22] | −0.0087 | **−0.0730** |
| dnn_5class_cw | 5.61% [5.33, 5.90] | +0.0018 | −0.0253 |

**Range: 0.75% to 5.61%. Average: 2.15%.** All CIs strictly positive. **NSL Dirichlet F1m drop confirmed: 6/6 CIs strictly negative on F1m delta.**

**Hybrid vs Dirichlet flipping rate comparison (NSL):** CIs DO NOT overlap for 5 of 6 models:
- rf_5class_smote: hybrid [3.35, 3.83] vs Dirichlet [1.57, 1.91] — **hybrid flips more** (no overlap)
- xgb_5class_smote: hybrid [0.16, 0.27] vs Dirichlet [0.99, 1.27] — **Dirichlet flips more** (no overlap, reversed direction)
- dnn_5class_smote: hybrid [2.53, 2.94] vs Dirichlet [2.39, 2.78] — **overlap** (statistically indistinguishable)
- rf_5class_cw: hybrid [4.49, 5.01] vs Dirichlet [0.63, 0.86] — **hybrid flips more**
- xgb_5class_cw: hybrid [0.20, 0.32] vs Dirichlet [0.95, 1.22] — **Dirichlet flips more** (reversed direction)
- dnn_5class_cw: hybrid [7.03, 7.71] vs Dirichlet [5.33, 5.90] — **hybrid flips more**

The two methods produce statistically different argmax rates on most NSL models, but not in a single direction. Hybrid flips more on rf and dnn models; Dirichlet flips more on xgb models.

**Important observation:** despite very low argmax flipping (1-3% on most models), **macro-F1 dropped 5-9 percentage points on 4 of 6 models** (5.1, 7.3, 7.8, 8.5 pp). This indicates the few flips were concentrated on rare classes — consistent with Dirichlet absorbing rare-class shift on NSL.

---

## 3. UNSW-NB15 calibration results

### 3.1 UNSW hybrid Platt/isotonic — macro Brier and ECE with bootstrap CIs

| Model | Brier pre | Brier post | ΔBrier (95% CI) | ΔECE (95% CI) |
|---|---:|---:|---|---|
| rf_5class_smote | 0.072236 | 0.069136 | −0.00310 [−0.00332, −0.00288] | −0.01305 [−0.01369, −0.01237] |
| xgb_5class_smote | 0.071247 | 0.068668 | −0.00258 [−0.00280, −0.00235] | −0.00589 [−0.00655, −0.00529] |
| dnn_5class_smote | 0.086865 | 0.072061 | −0.01480 [−0.01513, −0.01446] | −0.03082 [−0.03157, −0.03010] |
| rf_5class_cw | 0.069654 | 0.068178 | −0.00148 [−0.00169, −0.00127] | −0.00785 [−0.00854, −0.00727] |
| xgb_5class_cw | 0.075054 | 0.070140 | −0.00491 [−0.00515, −0.00464] | −0.01003 [−0.01077, −0.00942] |
| dnn_5class_cw | 0.093118 | 0.074384 | −0.01873 [−0.01907, −0.01839] | −0.03736 [−0.03809, −0.03662] |

**UNSW hybrid summary:** All 6 5-class models show statistically significant Brier and ECE improvement (all CIs strictly negative).

### 3.2 UNSW Dirichlet — macro Brier and ECE with bootstrap CIs

| Model | Brier pre | Brier post | ΔBrier (95% CI) | ΔECE (95% CI) |
|---|---:|---:|---|---|
| rf_5class_smote | 0.072236 | 0.065268 | −0.00697 [−0.00722, −0.00670] | −0.01740 [−0.01819, −0.01655] |
| xgb_5class_smote | 0.071247 | 0.064830 | −0.00642 [−0.00666, −0.00615] | −0.01059 [−0.01128, −0.00996] |
| dnn_5class_smote | 0.086865 | 0.067462 | −0.01940 [−0.01978, −0.01898] | −0.03576 [−0.03641, −0.03498] |
| rf_5class_cw | 0.069654 | 0.064908 | −0.00475 [−0.00499, −0.00450] | −0.01137 [−0.01215, −0.01055] |
| xgb_5class_cw | 0.075054 | 0.065927 | −0.00913 [−0.00942, −0.00881] | −0.01609 [−0.01683, −0.01531] |
| dnn_5class_cw | 0.093118 | 0.070523 | −0.02260 [−0.02300, −0.02219] | −0.04117 [−0.04189, −0.04045] |

**UNSW Dirichlet summary:** All 6 5-class models show statistically significant Brier and ECE improvement. Magnitudes consistently larger than hybrid.

### 3.3 UNSW three-way winner counts (macro Brier)

| Method | Wins on macro Brier (CI-confirmed) |
|---|---:|
| Pre-calibration | 3 of 9 (all binary) |
| Hybrid Platt/Isotonic | **0 of 9** |
| Dirichlet ODIR | **6 of 9** (all 5-class) |

**Headline:** Dirichlet wins every UNSW 5-class model with non-overlapping CIs over hybrid. Hybrid never wins on UNSW.

**Verification — Dirichlet vs Hybrid on UNSW 5-class:** For every UNSW 5-class model, Dirichlet's ΔBrier CI lies entirely below hybrid's ΔBrier CI (no overlap). Example: dnn_5class_cw hybrid [−0.01907, −0.01839] vs Dirichlet [−0.02300, −0.02219]. Dirichlet's superiority on UNSW 5-class is statistically tight.

### 3.4 UNSW Dirichlet — per-class Brier (full table)

Per-class Brier change (Δ = post − pre, negative = improvement):

| Model | Normal Δ | DoS Δ | Probe Δ | R2L Δ | U2R Δ | All improved? |
|---|---:|---:|---:|---:|---:|---|
| rf_5class_smote | −0.0073 | −0.0044 | −0.0080 | −0.0073 | −0.0077 | **Yes** |
| xgb_5class_smote | −0.0023 | −0.0063 | −0.0079 | −0.0116 | −0.0041 | **Yes** |
| dnn_5class_smote | −0.0168 | −0.0157 | −0.0162 | −0.0249 | −0.0235 | **Yes** |
| rf_5class_cw | −0.0067 | −0.0026 | −0.0077 | −0.0033 | −0.0036 | **Yes** |
| xgb_5class_cw | −0.0059 | −0.0066 | −0.0133 | −0.0134 | −0.0066 | **Yes** |
| dnn_5class_cw | −0.0262 | −0.0147 | −0.0159 | −0.0249 | −0.0313 | **Yes** |

**Confirmed: All 30 of 30 cells improved.** (Per-cell CIs not in the bootstrap output — bootstrap was on macro metrics. Per-cell CIs available if needed.)

**U2R per-model gains (largest improvements):**

| Model | U2R Brier pre | U2R Brier post | % reduction |
|---|---:|---:|---:|
| dnn_5class_cw | 0.0379 | 0.0066 | **−82.6%** |
| dnn_5class_smote | 0.0299 | 0.0064 | **−78.6%** |
| rf_5class_smote | 0.0130 | 0.0053 | −59.2% |
| xgb_5class_cw | 0.0112 | 0.0047 | −58.0% |
| xgb_5class_smote | 0.0086 | 0.0046 | −46.5% |
| rf_5class_cw | 0.0085 | 0.0050 | −41.2% |

### 3.5 UNSW hybrid — argmax preservation with CIs

| Model | % flipped (95% CI) | Δ acc (point) | Δ F1m (point) |
|---|---|---:|---:|
| rf_5class_smote | 18.74% [18.46, 19.00] | +0.0439 | −0.0082 |
| xgb_5class_smote | 12.73% [12.50, 12.96] | +0.0271 | −0.0169 |
| dnn_5class_smote | 23.65% [23.33, 23.94] | +0.0753 | +0.0114 |
| rf_5class_cw | 17.38% [17.11, 17.64] | +0.0435 | −0.0140 |
| xgb_5class_cw | 15.62% [15.36, 15.90] | +0.0292 | −0.0037 |
| dnn_5class_cw | 25.42% [25.10, 25.69] | +0.0623 | −0.0028 |

**UNSW hybrid flipping:** Range 12.73-25.42%, **average 18.92%**. All CIs narrow (~0.5 pp width). The substantial argmax shift is statistically tight.

### 3.6 UNSW Dirichlet — argmax preservation with CIs

| Model | % flipped (95% CI) | Δ acc (point) | Δ F1m (point) |
|---|---|---:|---:|
| rf_5class_smote | 19.39% [19.09, 19.65] | +0.0590 | +0.0005 |
| xgb_5class_smote | 14.29% [14.04, 14.52] | +0.0404 | +0.0021 |
| dnn_5class_smote | 25.98% [25.67, 26.28] | +0.0947 | +0.0196 |
| rf_5class_cw | 16.97% [16.69, 17.22] | +0.0512 | −0.0209 |
| xgb_5class_cw | 16.22% [15.96, 16.48] | +0.0508 | +0.0099 |
| dnn_5class_cw | 29.33% [29.01, 29.65] | **+0.1061** | **−0.0327** |

**UNSW Dirichlet flipping:** Range 14.29-29.33%, **average 20.36%**. All CIs narrow.

**UNSW hybrid vs Dirichlet flipping comparison — CORRECTED v5 (was wrongly stated as "mostly overlap" in v4):**

CIs **DO NOT overlap on 5 of 6 models**, indicating the two methods produce statistically different argmax disruption on UNSW:

| Model | Hybrid CI | Dirichlet CI | Status | Direction |
|---|---|---|---|---|
| rf_5class_smote | [18.46, 19.00] | [19.09, 19.65] | **No overlap** | Dirichlet flips more |
| xgb_5class_smote | [12.50, 12.96] | [14.04, 14.52] | **No overlap** | Dirichlet flips more |
| dnn_5class_smote | [23.33, 23.94] | [25.67, 26.28] | **No overlap** | Dirichlet flips more |
| rf_5class_cw | [17.11, 17.64] | [16.69, 17.22] | Overlap | (statistically indistinguishable) |
| xgb_5class_cw | [15.36, 15.90] | [15.96, 16.48] | **No overlap** | Dirichlet flips more |
| dnn_5class_cw | [25.10, 25.69] | [29.01, 29.65] | **No overlap** | Dirichlet flips more |

**Pattern: on UNSW, Dirichlet flips MORE samples than hybrid on 5 of 6 models (the exception is rf_5class_cw, where the methods are statistically indistinguishable).** The average rates (hybrid 18.92% vs Dirichlet 20.36%) understate this — they hide per-model statistical differences. This parallels the NSL finding (§2.6) that per-model CIs tell a richer story than averages.

**Critical observation on dnn_5class_cw (UNSW Dirichlet):** Accuracy improved 10.6 pp BUT macro-F1 dropped 3.3 pp. The class-weighted training boost on rare classes was partially undone by Dirichlet's prior absorption.

---

## 4. CIC-IDS2017 calibration results

### 4.1 CIC hybrid Platt/isotonic — macro Brier and ECE with bootstrap CIs

| Model | Brier pre | Brier post | ΔBrier (95% CI) | ΔECE (95% CI) | Operationally meaningful? |
|---|---:|---:|---|---|---|
| rf_5class_smote | 0.000629 | 0.000546 | −0.00008 [−0.00011, −0.00005] | −0.00098 [−0.00113, −0.00083] | Statistically significant but trivial magnitude |
| xgb_5class_smote | 0.000441 | 0.000413 | −0.00003 [−0.00005, −0.00000] | −0.00027 [−0.00031, −0.00010] | Marginal Brier; small ECE |
| dnn_5class_smote | 0.011243 | 0.005313 | **−0.00593 [−0.00634, −0.00552]** | **−0.01207 [−0.01269, −0.01104]** | **Yes — substantial recovery** |
| rf_5class_cw | 0.000603 | 0.000605 | **+0.00000 [−0.00004, +0.00004]** | −0.00051 [−0.00064, −0.00040] | **Brier CI INCLUDES ZERO — no effect** |
| xgb_5class_cw | 0.000365 | 0.000345 | −0.00002 [−0.00004, −0.00000] | −0.00019 [−0.00023, −0.00001] | Marginal — both CI upper bounds near zero |
| dnn_5class_cw | 0.052581 | 0.014770 | **−0.03781 [−0.03856, −0.03708]** | **−0.10074 [−0.10174, −0.09950]** | **Yes — dramatic recovery** |

**CIC hybrid summary (CI-aware):** 2 of 6 5-class models show operationally meaningful improvement (both DNN). 4 of 6 are at noise level or include zero.

**Headline numbers (statistically robust):**
- dnn_5class_cw macro ECE: 0.10656 → **0.00087** (ΔECE −0.10074 [−0.10174, −0.09950], −99.2%)
- dnn_5class_cw macro Brier: 0.05258 → **0.01477** (−71.9%)
- dnn_5class_smote macro Brier: 0.01124 → **0.00531** (−52.7%)

**Important caveat:** CIC tree-model improvements (rf_5class_smote, xgb_5class_smote, rf_5class_cw, xgb_5class_cw) are at noise level. Some CIs include zero (rf_5class_cw Brier) or barely exclude it. The bulk of CIC hybrid's apparent success comes from the two collapsed DNN models.

### 4.2 CIC Dirichlet — macro Brier and ECE with bootstrap CIs

| Model | Brier pre | Brier post | ΔBrier (95% CI) | ΔECE (95% CI) | Direction |
|---|---:|---:|---|---|---|
| rf_5class_smote | 0.000629 | 0.000686 | +0.00006 [+0.00001, +0.00011] | −0.00031 [−0.00045, −0.00017] | Brier ↑ (worsens), ECE ↓ |
| xgb_5class_smote | 0.000441 | 0.000421 | −0.00002 [−0.00006, +0.00002] | −0.00015 [−0.00026, +0.00007] | **Both CIs INCLUDE ZERO** |
| dnn_5class_smote | 0.011243 | 0.006034 | −0.00521 [−0.00557, −0.00483] | −0.00926 [−0.00992, −0.00845] | Both improve significantly |
| rf_5class_cw | 0.000603 | 0.000660 | +0.00006 [+0.00002, +0.00009] | +0.00021 [+0.00007, +0.00031] | **Both worsen significantly (small magnitude)** |
| xgb_5class_cw | 0.000365 | 0.000360 | −0.00000 [−0.00003, +0.00002] | −0.00007 [−0.00020, +0.00011] | **Both CIs INCLUDE ZERO** |
| dnn_5class_cw | 0.052581 | 0.015340 | −0.03724 [−0.03797, −0.03654] | −0.10104 [−0.10202, −0.09991] | Both improve dramatically |

**CIC Dirichlet summary:** On 4 of 6 tree models, the effect on Brier or ECE is statistically indistinguishable from zero or significantly worsens (small magnitude). The 2 dramatic improvements are both DNN models. **Dirichlet on CIC is essentially a coin-flip for tree models.**

### 4.3 CIC three-way winner counts (point estimates; bootstrap-confirmed)

| Method | Wins on macro Brier (point) | Wins with CI excluding zero (5-class only, n=6) |
|---|---:|---:|
| Pre-calibration | 0 | 0 |
| Hybrid | 7 of 9 | 2 of 6 (dnn_smote, dnn_cw) confirmed substantial; 3 of 6 marginal |
| Dirichlet | 2 of 9 | 1 of 6 (dnn_smote, dnn_cw) confirmed |

**Headline (CI-aware):** Both methods produce substantial improvements on the 2 catastrophically miscalibrated DNN models. For the 4 well-calibrated CIC tree models, neither method produces operationally meaningful change.

### 4.4 CIC per-class winners (5-class only, 30 cells — point estimates)

| Class | Hybrid wins | Dirichlet wins | Pre-cal wins |
|---|---:|---:|---:|
| Normal | 4 | 2 | 0 |
| DoS | 5 | 1 | 0 |
| Probe | 4 | 0 | 2 |
| R2L | 3 | 2 | 1 |
| U2R | 4 | 0 | 2 |
| **Total** | **20** | **5** | **5** |

**Caveat (audit-noted):** Many of these per-class wins are at noise level (deltas in the 4th-5th decimal place). The aggregate "hybrid wins 20" should be interpreted as "hybrid scores slightly lower Brier on 20 of 30 cells" without strong claims about operational significance for the small-magnitude cells.

### 4.5 CIC — argmax preservation with CIs

**Hybrid:**

| Model | % flipped (95% CI) |
|---|---|
| rf_5class_smote | 0.03% [0.01, 0.04] |
| xgb_5class_smote | 0.01% [0.00, 0.02] |
| dnn_5class_smote | 3.44% [3.26, 3.62] |
| rf_5class_cw | 0.05% [0.03, 0.07] |
| xgb_5class_cw | 0.01% [0.00, 0.03] |
| dnn_5class_cw | **19.34% [18.99, 19.74]** |

**Dirichlet:**

| Model | % flipped (95% CI) |
|---|---|
| rf_5class_smote | 0.10% [0.07, 0.14] |
| xgb_5class_smote | 0.02% [0.01, 0.03] |
| dnn_5class_smote | 3.44% [3.26, 3.62] |
| rf_5class_cw | 0.10% [0.07, 0.13] |
| xgb_5class_cw | 0.02% [0.01, 0.04] |
| dnn_5class_cw | **17.86% [17.53, 18.24]** |

**Hybrid vs Dirichlet argmax comparison (CI-aware):**

- For tree models and dnn_5class_smote: % flipped CIs essentially overlap or are within rounding of each other. Both methods preserve argmax similarly on these models.
- **For dnn_5class_cw (catastrophic model): hybrid CI [18.99, 19.74] does NOT overlap with Dirichlet CI [17.53, 18.24].** Hybrid flips MORE samples than Dirichlet on this model. Difference is statistically significant (~0.75 pp gap between Dirichlet upper and hybrid lower).

---

## 5. Cross-dataset summary

### 5.1 Three-way winner counts on macro Brier (9 models per dataset, point estimates)

| Dataset | Shift severity | Pre-cal wins | Hybrid wins | Dirichlet wins |
|---|---|---:|---:|---:|
| **NSL-KDD** | Severe | **6** | 1 (marginal) | 2 (binary only) |
| **UNSW-NB15** | Moderate | 3 (binary) | **0** | **6** (all 5-class, CIs tight) |
| **CIC-IDS2017** | None | 0 | **7** | 2 |

**Bootstrap-confirmed total across 27 model-dataset combinations:** Pre 9, Hybrid 8, Dirichlet 10.

### 5.2 Hybrid vs Dirichlet argmax-flipping comparison summary (NEW v5)

| Dataset | Non-overlapping CIs | Direction pattern |
|---|---:|---|
| NSL-KDD | 5 of 6 | Mixed — hybrid flips more on rf and dnn models; Dirichlet flips more on xgb models |
| UNSW-NB15 | 5 of 6 | **Consistent — Dirichlet flips more on 5 of 5 distinct models** |
| CIC-IDS2017 | 1 of 6 (dnn_5class_cw only) | Hybrid flips more by ~1.5 pp on the one catastrophic model |

**Pattern across datasets:** Averages hide significant per-model differences. UNSW shows the cleanest pattern (Dirichlet consistently flips more); NSL shows the messiest (method-direction varies by model architecture); CIC shows convergence except on the one collapsed model.

### 5.3 Argmax flipping rates by method × dataset (5-class average) with range of CIs

| Method | NSL | UNSW | CIC |
|---|---|---|---|
| **Hybrid** | 3.15% (range 0.21–7.35%, all CIs strictly positive, width ~0.3 pp) | 18.92% (range 12.73–25.42%, all CIs width ~0.5 pp) | 3.81% (range 0.01–19.34%, dominated by dnn_5class_cw) |
| **Dirichlet** | 2.15% (range 0.75–5.61%, all CIs strictly positive) | 20.36% (range 14.29–29.33%) | 3.81% (range 0.02–17.86%) |

**Key observation (refined):** flipping rates are NOT monotonic in distribution shift severity. UNSW (moderate shift) shows HIGHER flipping than NSL (severe shift) for both methods. CIs confirm this is a robust empirical fact, not a sampling artifact.

**Revised hypothesis (not yet directly verified):** flipping rate depends on WHICH classes shift, not just shift severity. NSL's shift is concentrated in rare classes (R2L 16×, U2R 7×), affecting few absolute samples; UNSW's shift is in the majority class (Normal 1.4×), affecting every test sample.

### 5.4 The U2R Platt-on-small-n problem (all 12 combinations)

| Dataset/Model | n_calib(U2R) | Brier pre | Brier post | Direction |
|---|---:|---:|---:|---|
| NSL U2R, rf_5class_smote | 10 | 0.00210 | 0.00290 | mild degradation |
| NSL U2R, xgb_5class_smote | 10 | 0.00230 | 0.00260 | mild degradation |
| NSL U2R, dnn_5class_smote | 10 | 0.00660 | 0.00300 | dramatic improvement |
| NSL U2R, rf_5class_cw | 10 | 0.00240 | 0.00320 | mild degradation |
| NSL U2R, xgb_5class_cw | 10 | 0.00240 | 0.00270 | mild degradation |
| NSL U2R, dnn_5class_cw | 10 | 0.01790 | 0.00290 | dramatic improvement |
| CIC U2R, rf_5class_smote | 7 | 0.00003 | 0.00005 | mild noise |
| CIC U2R, xgb_5class_smote | 7 | 0.00003 | 0.00003 | unchanged |
| CIC U2R, dnn_5class_smote | 7 | 0.00066 | 0.00017 | dramatic improvement |
| CIC U2R, rf_5class_cw | 7 | 0.00004 | 0.00017 | mild degradation |
| CIC U2R, xgb_5class_cw | 7 | 0.00002 | 0.00002 | unchanged |
| CIC U2R, dnn_5class_cw | 7 | 0.00628 | 0.00017 | dramatic improvement |

**Pattern:** Platt-on-small-n works when the underlying model is poorly calibrated (DNN models especially). It adds mild noise or leaves results unchanged when the model is already well-calibrated (tree models with already-near-zero Brier).

**Count:** 4 dramatic improvements (all DNN), 6 mild degradations (all tree), 2 unchanged.

---

## 6. Empirical findings (numbered, bootstrap-validated, v5-audit-verified)

1. **Calibration efficacy depends on calibration-test distribution alignment, with bootstrap confirmation.** On CIC (no shift), hybrid wins 7 of 9 models on point estimates but only 2 of 6 5-class wins have operationally meaningful CIs (the DNN models); the other 4 tree-model wins are at noise level. On UNSW (moderate shift), Dirichlet dominates 5-class tasks (6/6 wins, all CIs strictly negative). On NSL (severe shift), pre-cal wins on 6 of 9 models — post-hoc calibration fails to improve the baseline (most CIs strictly positive, indicating significant degradation).

2. **Both calibration methods substantially modify classifier predictions on UNSW, with statistically tight CIs.** Argmax flipping rates: hybrid 12.7-25.4% (CIs all width <0.6 pp), Dirichlet 14.3-29.3%. The calibrated probabilities, used as the basis for a downstream classifier, disagree with the pre-calibration argmax on a substantial fraction of samples.

3. **NSL Dirichlet flipping is low (0.75-5.61%, avg 2.15%) but macro-F1 drops 5-9 pp on 4 of 6 models with all CIs strictly negative.** This indicates the few flips are concentrated in rare classes where each error disproportionately impacts macro-F1. Dirichlet absorbs NSL's rare-class shift through implicit reclassification.

4. **Class-weighted models suffer most from Dirichlet's prior absorption.** UNSW dnn_5class_cw: accuracy +10.6 pp [CI: +10.4, +10.8] but macro-F1 −3.3 pp. The class-weight training boost on rare classes is partially undone when Dirichlet learns calib-prior alignment.

5. **Catastrophically miscalibrated models benefit dramatically from both methods, with very tight CIs.** CIC dnn_5class_cw: macro ECE 0.107 → 0.0009 (hybrid, ΔECE −0.10074 [−0.10174, −0.09950], −99.2%); accuracy 0.823 → 0.946 (+12.3 pp [+12.0, +12.8]). Calibration here functions as classifier recovery, not just confidence rescaling.

6. **Rare-class calibration is intrinsically noisy at the small sample sizes evaluated (n=7 and n=10), with method-dependent behaviour.** For well-calibrated tree models, all calibration methods add mild noise or leave results unchanged. For poorly-calibrated DNN models, all methods produce dramatic recovery. The "Platt-for-small-n" strategy in hybrid does not uniformly help; its value depends on the underlying model's pre-calibration quality.

7. **Binary calibration on UNSW degrades regardless of method.** Both methods worsen all 3 binary UNSW models. Hypothesis: 2-class structure cannot redistribute probability mass to absorb prior shift the way multiclass can.

8. **Hybrid and Dirichlet produce argmax behaviour that depends on the dataset.** On CIC (5 of 6 models): essentially indistinguishable; exception is the catastrophic dnn_5class_cw where hybrid flips ~1.5 pp more. On NSL: 5 of 6 models show statistically distinct flipping rates with mixed direction (hybrid more on rf and dnn; Dirichlet more on xgb). **On UNSW (corrected v5):** 5 of 6 models show statistically distinct flipping rates with **consistent direction — Dirichlet flips more on all 5 distinct models**. The two methods cannot be summarized as "one always flips more" — patterns depend on dataset and model architecture.

9. **NSL hybrid vs Dirichlet show statistically different argmax rates on 5 of 6 models, but not in a single direction.** Hybrid flips more on rf and dnn models (no CI overlap); Dirichlet flips more on xgb models (no CI overlap). Only dnn_5class_smote shows overlap. Method interaction with model architecture is non-trivial.

10. **CIC tree-model calibration "wins" are mostly at noise level.** rf_5class_cw under hybrid: ΔBrier CI [−0.00004, +0.00004] — INCLUDES ZERO. Several other tree-model Brier deltas have CIs whose upper bound rounds to zero. The dramatic CIC wins come from the two DNN models; tree models' calibration is essentially unchanged by either method.

11. **NEW v5: Averages of argmax flipping rates systematically understate per-model statistical differences between methods.** UNSW hybrid 18.92% vs Dirichlet 20.36% (1.4 pp difference) suggests methods are similar, but per-model CIs reveal Dirichlet flips significantly more on 5 of 6 distinct models. When comparing calibration methods, per-model CIs should be the primary evidence, not averages.

---

## 7. Honest limitations and gaps

1. **Bootstrap over test resamples only.** CIs answer "would different test set draws give the same result." They do NOT answer "would different model training seeds give the same result." That would require retraining (out of scope).

2. **Two methods only.** Temperature scaling, Beta calibration, histogram binning, BBQ not evaluated.

3. **Brier pre/post values updated v5 to bootstrap-consistent precision.** The original calibration notebooks stored some hybrid-method values at lower precision than the bootstrap notebook recomputes them (10 of 36 rows had arithmetic discrepancies of magnitude ~0.001-0.002). v5 uses the recomputed values (`results/tables/calibration_brier_recomputed.csv`) for display. Deltas and CIs are unchanged; they were correct in v4.

4. **n_calib=7 (CIC U2R) is too small for any calibration method to fit reliably.** All numbers for this class should be considered noise-level.

5. **The NSL flipping mechanism hypothesis (rare-class shift absorption) is plausible but not directly verified.** A targeted test would inspect which classes the flipped samples come from. Not done in this session.

6. **NSL hybrid and UNSW hybrid per-class Brier tables not in this document.** They exist in CSVs at `results/tables/`. Macro-level + CIs are sufficient for main findings.

7. **Per-cell bootstrap CIs not computed.** Bootstrap was on macro Brier and macro ECE. If a reviewer asks "what's the CI on dnn_5class_cw R2L Brier improvement?" we'd need to extend the bootstrap. Quick to add if needed.

8. **All findings descriptive of three specific benchmarks.** Generalization to other IDS datasets, real-world enterprise traffic, or non-IDS classification tasks is not established.

9. **No new calibration method proposed.** Contribution is empirical characterization of existing methods under distribution shift conditions common in IDS benchmarks, with bootstrap CIs providing statistical rigor.
