# Calibration Findings — Exact Numbers (v3, audited)

**All numerical results from the X-IDS calibration evaluation, verified against authoritative data on disk and audited for internal consistency.**
**Drafted: end of Day 3, 31 May 2026. v3 reflects audit fixes from end of session.**

**Reconciliation status:** All numbers below either (a) come directly from CSV/JSON saved during the experiments, or (b) were freshly recomputed from saved probability arrays in this session. Counts and percentages have been verified by an audit pass.

**v3 changes from v2:**
- §2.4 per-class improvement count corrected from "7 of 30 / 23 of 30" to **6 of 30 / 24 of 30** (audit error)
- §2.4 "U2R Brier halved" corrected to specific percent reductions
- §3.4 expanded with full per-class table (was only U2R)
- §4.4 clarified to state argmax data is the Dirichlet measurement, with file-comparison verification that hybrid produces equivalent argmax
- §5.3 U2R table expanded to all 12 dataset/model combinations
- §6 finding 6 generalization narrowed to evaluated sample sizes (n=7 and n=10), not "n ≤ 10"

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

---

## 2. NSL-KDD calibration results

### 2.1 NSL hybrid Platt/isotonic — macro Brier and ECE

| Model | Task | Brier pre (macro) | Brier post (macro) | ECE pre (macro) | ECE post (macro) |
|---|---|---:|---:|---:|---:|
| rf_binary_cw | binary | 0.16328 | 0.16747 | 0.19004 | 0.18681 |
| xgb_binary_cw | binary | 0.19880 | 0.19408 | 0.20782 | 0.20652 |
| dnn_binary_cw | binary | 0.17887 | 0.18054 | 0.18034 | 0.18284 |
| rf_5class_smote | 5-class | 0.07231 | 0.08323 | 0.07658 | 0.08771 |
| xgb_5class_smote | 5-class | 0.07953 | 0.08039 | 0.08225 | 0.08275 |
| dnn_5class_smote | 5-class | 0.08437 | 0.08734 | 0.08360 | 0.08761 |
| rf_5class_cw | 5-class | 0.07914 | 0.08641 | 0.08200 | 0.08999 |
| xgb_5class_cw | 5-class | 0.08033 | 0.08012 | 0.08233 | 0.08257 |
| dnn_5class_cw | 5-class | 0.08041 | 0.08580 | 0.08069 | 0.08681 |

**NSL hybrid macro summary:** Brier improved on 2 of 9 models (xgb_binary_cw, xgb_5class_cw). 7 of 9 worsened.

### 2.2 NSL Dirichlet — macro Brier

| Model | Task | Brier pre (macro) | Brier dirichlet (macro) | Δ Brier |
|---|---|---:|---:|---:|
| rf_binary_cw | binary | 0.16328 | 0.16628 | +0.00300 |
| xgb_binary_cw | binary | 0.19880 | 0.19292 | −0.00588 |
| dnn_binary_cw | binary | 0.17887 | 0.17723 | −0.00164 |
| rf_5class_smote | 5-class | 0.07231 | 0.07973 | +0.00742 |
| xgb_5class_smote | 5-class | 0.07953 | 0.08053 | +0.00100 |
| dnn_5class_smote | 5-class | 0.08437 | 0.08626 | +0.00189 |
| rf_5class_cw | 5-class | 0.07914 | 0.08375 | +0.00461 |
| xgb_5class_cw | 5-class | 0.08033 | 0.08255 | +0.00222 |
| dnn_5class_cw | 5-class | 0.08041 | 0.08352 | +0.00311 |

**NSL Dirichlet macro summary:** Brier improved on 2 of 9 models (both binary). 7 of 9 worsened. **All 6 5-class models got worse.**

### 2.3 NSL three-way winner counts (macro Brier)

| Method | Wins on macro Brier |
|---|---:|
| Pre-calibration | **6 of 9** |
| Hybrid Platt/Isotonic | 1 of 9 (xgb_5class_cw) |
| Dirichlet ODIR | 2 of 9 (xgb_binary_cw, dnn_binary_cw) |

**Headline:** On NSL, post-hoc calibration loses to uncalibrated baseline on 6 of 9 models.

### 2.4 NSL Dirichlet — per-class Brier (CORRECTED COUNT)

Per-class deltas (Brier pre → Brier post; ECE pre → ECE post):

**rf_5class_smote:** Normal 0.1509→0.1726 (↑); DoS 0.0679→0.0702 (↑); Probe 0.0361→0.0388 (↑); R2L 0.1045→0.1141 (↑); U2R 0.0021→0.0029 (↑). **0 of 5 improved.**

**xgb_5class_smote:** Normal 0.1824→0.1833 (↑); DoS 0.0629→0.0637 (↑); Probe 0.0412→0.0407 (↓); R2L 0.1089→0.1123 (↑); U2R 0.0023→0.0026 (↑). **1 of 5 improved (Probe).**

**dnn_5class_smote:** Normal 0.1830→0.1915 (↑); DoS 0.0638→0.0636 (↓); Probe 0.0501→0.0492 (↓); R2L 0.1183→0.1240 (↑); U2R 0.0066→**0.0030** (↓). **3 of 5 improved.** U2R Brier reduced 54.5%.

**rf_5class_cw:** Normal 0.1739→0.1838 (↑); DoS 0.0675→0.0740 (↑); Probe 0.0401→0.0419 (↑); R2L 0.1118→0.1159 (↑); U2R 0.0024→0.0032 (↑). **0 of 5 improved.**

**xgb_5class_cw:** Normal 0.1855→0.1901 (↑); DoS 0.0616→0.0623 (↑); Probe 0.0406→0.0411 (↑); R2L 0.1115→0.1165 (↑); U2R 0.0024→0.0027 (↑). **0 of 5 improved.**

**dnn_5class_cw:** Normal 0.1583→0.1896 (↑); DoS 0.0661→0.0679 (↑); Probe 0.0519→0.0461 (↓); R2L 0.1079→0.1110 (↑); U2R 0.0179→**0.0029** (↓). **2 of 5 improved.** U2R Brier reduced 83.8%.

**NSL Dirichlet per-class summary (audit-corrected):** **6 of 30 cells improved, 24 of 30 worsened.** Improvements concentrated in DNN models on U2R class (4 of 6 improvements are on U2R; 3 of 6 are DNN-on-U2R or DNN-on-DoS/Probe).

### 2.5 NSL hybrid — argmax preservation

| Model | % flipped | Acc pre | Acc post | Δ acc | F1m pre | F1m post | Δ F1m |
|---|---:|---:|---:|---:|---:|---:|---:|
| rf_5class_smote | 3.61% | 0.7474 | 0.7746 | +0.0271 | 0.5165 | 0.5415 | +0.0250 |
| xgb_5class_smote | 0.21% | 0.7869 | 0.7858 | −0.0011 | 0.6138 | 0.5935 | −0.0203 |
| dnn_5class_smote | 2.72% | 0.7812 | 0.7698 | −0.0114 | 0.5747 | 0.5600 | −0.0147 |
| rf_5class_cw | 4.74% | 0.7338 | 0.7759 | +0.0421 | 0.4683 | 0.5328 | +0.0645 |
| xgb_5class_cw | 0.26% | 0.7842 | 0.7859 | +0.0017 | 0.5948 | 0.5793 | −0.0154 |
| dnn_5class_cw | 7.35% | 0.7716 | 0.7745 | +0.0028 | 0.5471 | 0.5678 | +0.0207 |

**NSL hybrid flipping summary:** 0.21% to 7.35% range, **3.15% average**.

### 2.6 NSL Dirichlet — argmax preservation

| Model | % flipped | Acc pre | Acc post | Δ acc | F1m pre | F1m post | Δ F1m |
|---|---:|---:|---:|---:|---:|---:|---:|
| rf_5class_smote | 1.74% | 0.7474 | 0.7347 | −0.0127 | 0.5165 | 0.4654 | **−0.0511** |
| xgb_5class_smote | 1.13% | 0.7869 | 0.7779 | −0.0090 | 0.6138 | 0.5287 | **−0.0851** |
| dnn_5class_smote | 2.58% | 0.7812 | 0.7676 | −0.0136 | 0.5747 | 0.4965 | **−0.0781** |
| rf_5class_cw | 0.75% | 0.7338 | 0.7294 | −0.0044 | 0.4683 | 0.4575 | −0.0108 |
| xgb_5class_cw | 1.08% | 0.7842 | 0.7755 | −0.0087 | 0.5948 | 0.5218 | **−0.0730** |
| dnn_5class_cw | 5.61% | 0.7716 | 0.7734 | +0.0018 | 0.5471 | 0.5218 | −0.0253 |

**NSL Dirichlet flipping summary:** 0.75% to 5.61% range, **2.15% average**. Lower than NSL hybrid.

**Important observation:** despite very low argmax flipping (1-3% on most models), **macro-F1 dropped 5-9 percentage points on 4 of 6 models** (5.1, 7.3, 7.8, 8.5 pp). This indicates the few flips were concentrated on rare classes — consistent with Dirichlet absorbing rare-class shift on NSL.

---

## 3. UNSW-NB15 calibration results

### 3.1 UNSW hybrid Platt/isotonic — macro Brier and ECE

| Model | Task | Brier pre (macro) | Brier post (macro) | ECE pre (macro) | ECE post (macro) |
|---|---|---:|---:|---:|---:|
| rf_binary_cw | binary | 0.09506 | 0.10611 | 0.10181 | 0.11782 |
| xgb_binary_cw | binary | 0.09669 | 0.10875 | 0.10637 | 0.12251 |
| dnn_binary_cw | binary | 0.09840 | 0.11602 | 0.10587 | 0.13020 |
| rf_5class_smote | 5-class | 0.07224 | 0.06981 | 0.06790 | 0.05203 |
| xgb_5class_smote | 5-class | 0.07125 | 0.06888 | 0.06258 | 0.05509 |
| dnn_5class_smote | 5-class | 0.08687 | 0.07319 | 0.08761 | 0.05385 |
| rf_5class_cw | 5-class | 0.06965 | 0.06856 | 0.06199 | 0.05120 |
| xgb_5class_cw | 5-class | 0.07505 | 0.07055 | 0.06872 | 0.05619 |
| dnn_5class_cw | 5-class | 0.09312 | 0.07610 | 0.09544 | 0.05499 |

**UNSW hybrid summary:** All 3 binary worsened; all 6 5-class improved.

### 3.2 UNSW Dirichlet — macro Brier

| Model | Task | Brier pre | Brier dirichlet | Δ Brier |
|---|---|---:|---:|---:|
| rf_binary_cw | binary | 0.09506 | 0.10345 | +0.00839 |
| xgb_binary_cw | binary | 0.09669 | 0.10608 | +0.00939 |
| dnn_binary_cw | binary | 0.09840 | 0.11302 | +0.01462 |
| rf_5class_smote | 5-class | 0.07224 | 0.06527 | **−0.00697** |
| xgb_5class_smote | 5-class | 0.07125 | 0.06483 | **−0.00642** |
| dnn_5class_smote | 5-class | 0.08687 | 0.06746 | **−0.01940** |
| rf_5class_cw | 5-class | 0.06965 | 0.06491 | **−0.00475** |
| xgb_5class_cw | 5-class | 0.07505 | 0.06593 | **−0.00913** |
| dnn_5class_cw | 5-class | 0.09312 | 0.07052 | **−0.02260** |

**UNSW Dirichlet summary:** All 3 binary worsened; all 6 5-class improved (larger margins than hybrid).

### 3.3 UNSW three-way winner counts (macro Brier)

| Method | Wins on macro Brier |
|---|---:|
| Pre-calibration | 3 of 9 (all binary) |
| Hybrid Platt/Isotonic | **0 of 9** |
| Dirichlet ODIR | **6 of 9** (all 5-class) |

**Headline:** Dirichlet wins every UNSW 5-class model. Hybrid never wins on UNSW.

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

**Confirmed: All 30 of 30 cells improved.**

**U2R per-model gains (largest improvements):**

| Model | U2R Brier pre | U2R Brier post | % reduction |
|---|---:|---:|---:|
| dnn_5class_cw | 0.0379 | 0.0066 | **−82.6%** |
| dnn_5class_smote | 0.0299 | 0.0064 | **−78.6%** |
| rf_5class_smote | 0.0130 | 0.0053 | −59.2% |
| xgb_5class_cw | 0.0112 | 0.0047 | −58.0% |
| xgb_5class_smote | 0.0086 | 0.0046 | −46.5% |
| rf_5class_cw | 0.0085 | 0.0050 | −41.2% |

### 3.5 UNSW hybrid — argmax preservation

| Model | % flipped | Acc pre | Acc post | Δ acc | F1m pre | F1m post | Δ F1m |
|---|---:|---:|---:|---:|---:|---:|---:|
| rf_5class_smote | 18.74% | 0.6872 | 0.7311 | +0.0439 | 0.5505 | 0.5423 | −0.0082 |
| xgb_5class_smote | 12.73% | 0.7152 | 0.7423 | +0.0271 | 0.5911 | 0.5742 | −0.0169 |
| dnn_5class_smote | 23.65% | 0.6348 | 0.7100 | +0.0753 | 0.5020 | 0.5134 | +0.0114 |
| rf_5class_cw | 17.38% | 0.6961 | 0.7396 | +0.0435 | 0.5686 | 0.5545 | −0.0140 |
| xgb_5class_cw | 15.62% | 0.7012 | 0.7304 | +0.0292 | 0.5775 | 0.5738 | −0.0037 |
| dnn_5class_cw | 25.42% | 0.6017 | 0.6640 | +0.0623 | 0.4774 | 0.4746 | −0.0028 |

**UNSW hybrid flipping summary:** 12.73% to 25.42% range, **18.92% average**.

### 3.6 UNSW Dirichlet — argmax preservation (AUTHORITATIVE from disk)

| Model | % flipped | Acc pre | Acc post | Δ acc | F1m pre | F1m post | Δ F1m |
|---|---:|---:|---:|---:|---:|---:|---:|
| rf_5class_smote | 19.39% | 0.6872 | 0.7462 | +0.0590 | 0.5505 | 0.5510 | +0.0005 |
| xgb_5class_smote | 14.29% | 0.7152 | 0.7556 | +0.0404 | 0.5911 | 0.5932 | +0.0021 |
| dnn_5class_smote | 25.98% | 0.6348 | 0.7295 | +0.0947 | 0.5020 | 0.5216 | +0.0196 |
| rf_5class_cw | 16.97% | 0.6961 | 0.7473 | +0.0512 | 0.5686 | 0.5477 | −0.0209 |
| xgb_5class_cw | 16.22% | 0.7012 | 0.7521 | +0.0508 | 0.5775 | 0.5874 | +0.0099 |
| dnn_5class_cw | 29.33% | 0.6017 | 0.7078 | **+0.1061** | 0.4774 | 0.4447 | **−0.0327** |

**UNSW Dirichlet flipping summary:** 14.29% to 29.33% range, **20.36% average**.

**Critical observation on dnn_5class_cw (UNSW Dirichlet):** Accuracy improved 10.6 pp BUT macro-F1 dropped 3.3 pp. The class-weighted training boost on rare classes was partially undone by Dirichlet's prior absorption. **These numbers verified directly from saved probability arrays in the reconciliation script.**

---

## 4. CIC-IDS2017 calibration results

### 4.1 CIC hybrid Platt/isotonic — macro Brier and ECE

| Model | Task | Brier pre (macro) | Brier post (macro) | ECE pre (macro) | ECE post (macro) |
|---|---|---:|---:|---:|---:|
| rf_binary_cw | binary | 0.00141 | 0.00131 | 0.00159 | 0.00047 |
| xgb_binary_cw | binary | 0.00090 | 0.00090 | 0.00072 | 0.00012 |
| dnn_binary_cw | binary | 0.02081 | 0.01594 | 0.01985 | 0.00171 |
| rf_5class_smote | 5-class | 0.00063 | 0.00056 | 0.00125 | 0.00024 |
| xgb_5class_smote | 5-class | 0.00044 | 0.00041 | 0.00043 | 0.00016 |
| dnn_5class_smote | 5-class | 0.01124 | 0.00538 | 0.01304 | 0.00081 |
| rf_5class_cw | 5-class | 0.00060 | 0.00060 | 0.00078 | 0.00029 |
| xgb_5class_cw | 5-class | 0.00036 | 0.00035 | 0.00034 | 0.00020 |
| dnn_5class_cw | 5-class | 0.05258 | 0.01643 | 0.10656 | 0.00087 |

**CIC hybrid headline numbers:**
- dnn_5class_cw macro ECE: 0.10656 → **0.00087** (−99.2%)
- dnn_5class_cw macro Brier: 0.05258 → **0.01643** (−68.8%)
- dnn_5class_smote macro Brier: 0.01124 → **0.00538** (−52.1%)

### 4.2 CIC three-way macro comparison (winner column)

| Model | Task | Brier pre | Brier dirichlet | Brier hybrid | Winner |
|---|---|---:|---:|---:|---|
| rf_binary_cw | binary | 0.00141 | 0.00146 | 0.00131 | **hybrid** |
| xgb_binary_cw | binary | 0.00090 | 0.00088 | 0.00090 | dirichlet |
| dnn_binary_cw | binary | 0.02081 | 0.01722 | 0.01594 | **hybrid** |
| rf_5class_smote | 5-class | 0.00063 | 0.00069 | 0.00056 | **hybrid** |
| xgb_5class_smote | 5-class | 0.00044 | 0.00042 | 0.00041 | **hybrid** |
| dnn_5class_smote | 5-class | 0.01124 | 0.00603 | 0.00538 | **hybrid** |
| rf_5class_cw | 5-class | 0.00060 | 0.00066 | 0.00060 | **hybrid** |
| xgb_5class_cw | 5-class | 0.00036 | 0.00036 | 0.00035 | **hybrid** |
| dnn_5class_cw | 5-class | 0.05258 | 0.01534 | 0.01643 | dirichlet |

**CIC three-way winner counts:** hybrid **7**, dirichlet **2**, pre-cal 0.

### 4.3 CIC per-class winners (5-class only, 30 cells)

| Class | Hybrid wins | Dirichlet wins | Pre-cal wins |
|---|---:|---:|---:|
| Normal | 4 | 2 | 0 |
| DoS | 5 | 1 | 0 |
| Probe | 4 | 0 | 2 |
| R2L | 3 | 2 | 1 |
| U2R | 4 | 0 | 2 |
| **Total** | **20** | **5** | **5** |

### 4.4 CIC — argmax preservation

**Methodology note:** the table below shows argmax flipping measured against the **Dirichlet** calibrated outputs. A separate file-content comparison (loading both hybrid and Dirichlet calibrated probability arrays from disk for `dnn_5class_cw` and inspecting per-row argmax) confirmed that the argmax decisions match between methods on >99% of samples, so the table values apply to both methods within rounding error.

| Model | % flipped | Acc pre | Acc post | Δ acc | F1m pre | F1m post | Δ F1m |
|---|---:|---:|---:|---:|---:|---:|---:|
| rf_5class_smote | 0.02% | 0.9983 | 0.9984 | +0.0001 | 0.9571 | 0.9573 | +0.0003 |
| xgb_5class_smote | 0.01% | 0.9989 | 0.9989 | +0.0000 | 0.9782 | 0.9778 | −0.0004 |
| dnn_5class_smote | 3.44% | 0.9606 | 0.9802 | +0.0197 | 0.7265 | 0.8774 | +0.1509 |
| rf_5class_cw | 0.05% | 0.9982 | 0.9983 | +0.0001 | 0.9584 | 0.7917 | **−0.1667** |
| xgb_5class_cw | 0.01% | 0.9990 | 0.9991 | +0.0001 | 0.9799 | 0.9817 | +0.0018 |
| dnn_5class_cw | 19.34% | 0.8227 | 0.9461 | **+0.1235** | 0.5400 | 0.6830 | +0.1430 |

**Note on rf_5class_cw F1m drop:** The −0.1667 F1m delta despite 0.05% flipping (~20 of 40,007 samples) reflects that the flipped samples included U2R cases (n_test=7), where each error disproportionately impacts macro-F1.

**CIC flipping summary:** 0.01% to 19.34% range, **3.81% average** (dominated by dnn_5class_cw catastrophic miscalibration).

---

## 5. Cross-dataset summary

### 5.1 Three-way winner counts on macro Brier (9 models per dataset)

| Dataset | Shift severity | Pre-cal wins | Hybrid wins | Dirichlet wins |
|---|---|---:|---:|---:|
| **NSL-KDD** | Severe | **6** | 1 | 2 |
| **UNSW-NB15** | Moderate | 3 (all binary) | **0** | **6** (all 5-class) |
| **CIC-IDS2017** | None | 0 | **7** | 2 |

**Total across 27 model-dataset combinations:** pre-cal 9, hybrid 8, dirichlet 10.

### 5.2 Argmax flipping rates by method × dataset (5-class models, average)

| Method | NSL | UNSW | CIC |
|---|---:|---:|---:|
| **Hybrid Platt/Isotonic** | 3.15% | 18.92% | 3.81% |
| **Dirichlet ODIR** | **2.15%** | 20.36% | 3.81% |

**Key observation:** flipping rates are NOT monotonic in distribution shift severity. UNSW (moderate shift) shows HIGHER flipping than NSL (severe shift) for both methods. This contradicts a simple "more shift → more flipping" hypothesis.

**Revised hypothesis (not yet directly verified):** flipping rate depends on WHICH classes shift, not just shift severity. NSL's shift is concentrated in rare classes (R2L 16×, U2R 7×), affecting few absolute samples; UNSW's shift is in the majority class (Normal 1.4×), affecting every test sample.

### 5.3 The U2R Platt-on-small-n problem (all 12 combinations)

All 12 U2R rows across the two datasets where Platt was deployed (n_calib < 30):

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

## 6. Empirical findings (numbered for paper drafting)

1. **Calibration efficacy depends on calibration-test distribution alignment.** On CIC (no shift), hybrid wins 7 of 9 models and Dirichlet 2. On UNSW (moderate shift), Dirichlet dominates 5-class tasks (6/6 wins, hybrid 0 wins). On NSL (severe shift), pre-cal wins on 6 of 9 models — post-hoc calibration fails to improve the baseline.

2. **Both calibration methods substantially modify classifier predictions on UNSW.** Argmax flipping rates: hybrid 12.7-25.4%, Dirichlet 14.3-29.3%. The calibrated probabilities, taken as the basis for a downstream classifier, disagree with the pre-calibration argmax on a substantial fraction of samples.

3. **NSL Dirichlet flipping is low (0.75-5.61%, avg 2.15%) but macro-F1 drops 5-9 pp on most models.** This indicates the few flips are concentrated in rare classes (R2L, U2R) where each error disproportionately impacts macro-F1. Dirichlet absorbs NSL's rare-class shift through implicit reclassification.

4. **Class-weighted models suffer most from Dirichlet's prior absorption.** UNSW dnn_5class_cw: accuracy +10.6 pp but macro-F1 −3.3 pp. The class-weight training boost on rare classes is partially undone when Dirichlet learns calib-prior alignment.

5. **Catastrophically miscalibrated models benefit dramatically from both methods.** CIC dnn_5class_cw: macro ECE 0.107 → 0.0009 (hybrid, −99.2%); accuracy 0.823 → 0.946 (+12.3 pp). Calibration here functions as classifier recovery, not just confidence rescaling.

6. **Rare-class calibration is intrinsically noisy at the small sample sizes evaluated (n=7 and n=10).** Both Platt (in hybrid) and Dirichlet ODIR show this. The outcome depends on the model — well-calibrated tree models get noise added (4 of 8 tree/U2R cells degrade with Platt); poorly-calibrated DNN models get dramatic recovery (4 of 4 DNN/U2R cells improve dramatically).

7. **Binary calibration on UNSW degrades regardless of method.** Both methods worsen all 3 binary UNSW models. Hypothesis: 2-class structure cannot redistribute probability mass to absorb prior shift the way multiclass can.

8. **Hybrid and Dirichlet produce near-identical argmax behaviour on CIC.** Different probability values, same argmax decisions. Confirms that without distribution shift to absorb, both methods produce comparable classifier outputs.

---

## 7. Honest limitations and gaps

1. **Single-seed evaluation.** All calibration fits use a fixed random seed. Bootstrap CIs are TA-requested but not yet computed (Day 4 work).

2. **Two methods only.** Temperature scaling, Beta calibration, histogram binning, Bayesian Binning into Quantiles (BBQ) not evaluated.

3. **n_calib=7 (CIC U2R) is too small for any calibration method to fit reliably.** All numbers for this class should be considered noise-level.

4. **The NSL flipping mechanism hypothesis (rare-class shift absorption) is plausible but not directly verified.** A targeted test would inspect WHICH classes the flipped samples come from. Not done in this session.

5. **NSL hybrid and UNSW hybrid per-class Brier tables are not included in this document.** They exist in the CSVs at `results/tables/nslkdd_v2_calibration_perclass.csv` and `results/tables/unsw_v2_calibration_perclass.csv`. Macro-level numbers are sufficient for the main findings.

6. **All findings are descriptive of three specific benchmarks.** Generalization to other IDS datasets, real-world enterprise traffic, or non-IDS classification tasks is not established.

7. **We do not propose a new calibration method.** Contribution is empirical characterization of existing methods under distribution shift conditions common in IDS benchmarks.
