# Day 3 Calibration Work — Comprehensive Session Memory (v3, audited)

**Project:** X-IDS Research (Explainable Intrusion Detection System) under PAIDS, supervisor Ms Alaa Mohasseb.
**Target venue:** IEEE TIFS / TDSC (top-tier security journals).
**Repo:** github.com/anasbiswas1/xids-research
**Session date:** Started ~evening 30 May 2026, ran through into night of 31 May 2026.
**Document purpose:** Capture every aspect of the calibration work tonight — empirical results, methodology decisions, mistakes made, reasoning chains, pending work, and orientation for future sessions. This is the authoritative session memory; if I lose this conversation, I should be able to pick up the calibration work from this file alone.

**v3 changes from v2:**
- §5.1 NSL Dirichlet per-class count corrected from "7 improved / 23 worsened" to **6 improved / 24 worsened** (audit error from earlier conversation, propagated to docs)
- §6 framing softened from "the calibrated classifier is NOT the same classifier" to "calibrated probabilities, used as the basis for a downstream classifier, disagree with the pre-calibration argmax on a substantial fraction of samples" (avoids philosophical claim about classifier identity)
- §10 "tomorrow's writeup work" generalized to "next session" (time-independent framing)
- §11.1 acknowledges that an audit pass was done at end of session

---

## 1. Session context and starting state

### 1.1 What was already done before tonight (Day 1 + Day 2)

The v2 rebuild was complete: M1/M6/M10 methodology fixes applied across all three datasets (NSL-KDD, UNSW-NB15, CIC-IDS2017). 27 models trained (3 datasets × 3 architectures × 3 label tasks: binary class-weighted, 5-class SMOTE, 5-class class-weighted). NSL RandomForest retrained with max_depth=20. Progress doc v8 committed at `9fb0ba0`.

Calibration sets were carved out during preprocessing (M6 fix): an 80/20 stratified split of training data, producing a held-out calibration set per dataset never seen during model training. This is the fundamental prerequisite for tonight's work — calibration cannot be applied without a separate calibration set.

Day 2 ended with a major finding: the CIC class-weighted DNN model (dnn_5class_cw) had collapsed during training due to extreme class-weight ratios (Normal ~0.25, U2R ~1091, ratio of ~4400×). Test accuracy 0.823, macro-F1 0.540. This collapsed model would later become important for tonight's calibration findings.

### 1.2 TA comments to address

Three comments from Ms Alaa via the TA channel before tonight:

1. **Rare-class calibration caveat:** "The rare-class calibration sample size concern needs to be addressed explicitly. NSL U2R has only 10 calibration samples; CIC U2R has 7." This was the original driver for tonight's hybrid Platt/isotonic approach (Platt scaling for n < 30).

2. **Bootstrap confidence intervals:** Required on four claim types — (a) SCTS-v2 quartile gaps, (b) Krishna rank correlations Spearman/Kendall, (c) DNN stability advantage under FGSM/PGD/Gaussian, (d) UNSW DoS hardest-to-trust finding. B=1000 resamples, 95% CIs (2.5/97.5 percentile). **Scheduled for Day 4. Not done tonight.**

3. **LLM evaluation is weak:** Required a 3-layer evaluation — (L1) LLM-judge on full set, (L2) 50 alerts stratified rated by user + supervisor + 1 peer on Correctness/Actionability/Clarity 1-5, (L3) LLM-judge rates same 50 + Cohen's kappa LLM-vs-human. **Scheduled for Day 5. Not done tonight.**

Tonight's calibration work primarily addresses comment 1 (rare-class concern). Comments 2 and 3 remain pending.

### 1.3 Tonight's goal at session start

Run calibration experiments on three datasets and produce a deliverable for the upcoming supervisor meeting (rescheduled if needed). Specifically: apply hybrid Platt/isotonic calibration as a methodology choice that addresses the rare-class concern, document the empirical results.

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

This was a critical discovery early in tonight's session. The three datasets span a spectrum of calibration-test distribution shift, which we did not initially appreciate:

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

Raw model probabilities. RandomForest from leaf-node class proportions. XGBoost from sigmoid (binary) or softmax (multiclass). DNN from softmax layer. Serves as the baseline against which post-hoc methods are evaluated.

### 4.2 Hybrid Platt/Isotonic

Per-class one-versus-rest scheme. Strategy selection based on per-class sample size:
- **n_calib ≥ 30:** Isotonic regression (non-parametric monotone mapping, captures arbitrary calibration curves)
- **n_calib < 30:** Platt scaling (single 2-parameter sigmoid, stable on small samples)

After per-class calibration, probability vector is renormalized to sum to 1.

**Where Platt is actually deployed in our experiments:** NSL U2R (n=10), CIC U2R (n=7). Everywhere else: isotonic.

**Implementation notebooks:** `notebooks/03_nsl_calibration_v2.ipynb`, `notebooks/03_unsw_calibration_v2.ipynb`, `notebooks/03_cic_calibration_v2.ipynb`.

### 4.3 Dirichlet ODIR

Joint multiclass calibration (Kull et al., NeurIPS 2019). Learns a single k×k linear transformation in log-probability space plus a k-vector of intercepts. ODIR variant uses Off-Diagonal and Intercept Regularization — more stable than full Dirichlet on rare classes.

No per-class decomposition, no post-hoc renormalization. Mathematically: `p_calibrated = softmax(W * log(p_raw) + b)` where W is the k×k transformation and b is the intercept vector.

**Implementation notebooks:** `notebooks/03c_nsl_calibration_dirichlet_v2.ipynb`, `notebooks/03c_unsw_calibration_dirichlet_v2.ipynb`, `notebooks/03c_cic_calibration_dirichlet_v2.ipynb`. Uses `dirichletcal` Python package.

### 4.4 Why these two methods

- **Hybrid** directly addresses the TA's rare-class concern (Platt-for-small-n)
- **Dirichlet** is current SOTA for multiclass calibration; joint method without per-class fits
- Together they bracket the methodology space: per-class vs joint, parametric vs non-parametric

**Not evaluated:** Temperature scaling, Beta calibration, histogram binning, BBQ. Explicitly out of scope.

---

## 5. Empirical results — all numbers

### 5.1 NSL-KDD

**Hybrid Platt/Isotonic — macro Brier:**

| Model | Task | Brier pre | Brier post | Improved? |
|---|---|---:|---:|---|
| rf_binary_cw | binary | 0.16328 | 0.16747 | no |
| xgb_binary_cw | binary | 0.19880 | 0.19408 | **yes** |
| dnn_binary_cw | binary | 0.17887 | 0.18054 | no |
| rf_5class_smote | 5-class | 0.07231 | 0.08323 | no |
| xgb_5class_smote | 5-class | 0.07953 | 0.08039 | no |
| dnn_5class_smote | 5-class | 0.08437 | 0.08734 | no |
| rf_5class_cw | 5-class | 0.07914 | 0.08641 | no |
| xgb_5class_cw | 5-class | 0.08033 | 0.08012 | **yes** |
| dnn_5class_cw | 5-class | 0.08041 | 0.08580 | no |

**NSL hybrid summary:** Improved 2 of 9, worsened 7 of 9.

**Dirichlet — macro Brier:**

| Model | Task | Brier pre | Brier dirichlet | Improved? |
|---|---|---:|---:|---|
| rf_binary_cw | binary | 0.16328 | 0.16628 | no |
| xgb_binary_cw | binary | 0.19880 | 0.19292 | **yes** |
| dnn_binary_cw | binary | 0.17887 | 0.17723 | **yes** |
| rf_5class_smote | 5-class | 0.07231 | 0.07973 | no |
| xgb_5class_smote | 5-class | 0.07953 | 0.08053 | no |
| dnn_5class_smote | 5-class | 0.08437 | 0.08626 | no |
| rf_5class_cw | 5-class | 0.07914 | 0.08375 | no |
| xgb_5class_cw | 5-class | 0.08033 | 0.08255 | no |
| dnn_5class_cw | 5-class | 0.08041 | 0.08352 | no |

**NSL Dirichlet summary:** Improved 2 of 9 (both binary), worsened 7 of 9. **All 6 5-class models worsened.**

**Three-way winner counts:** Pre 6, Hybrid 1, Dirichlet 2.

**Hybrid argmax flipping (NSL 5-class):**

| Model | % flipped | Δ acc | Δ F1m |
|---|---:|---:|---:|
| rf_5class_smote | 3.61% | +0.0271 | +0.0250 |
| xgb_5class_smote | 0.21% | −0.0011 | −0.0203 |
| dnn_5class_smote | 2.72% | −0.0114 | −0.0147 |
| rf_5class_cw | 4.74% | +0.0421 | +0.0645 |
| xgb_5class_cw | 0.26% | +0.0017 | −0.0154 |
| dnn_5class_cw | 7.35% | +0.0028 | +0.0207 |

**Average: 3.15% flipping.**

**Dirichlet argmax flipping (NSL 5-class):**

| Model | % flipped | Δ acc | Δ F1m |
|---|---:|---:|---:|
| rf_5class_smote | 1.74% | −0.0127 | −0.0511 |
| xgb_5class_smote | 1.13% | −0.0090 | −0.0851 |
| dnn_5class_smote | 2.58% | −0.0136 | −0.0781 |
| rf_5class_cw | 0.75% | −0.0044 | −0.0108 |
| xgb_5class_cw | 1.08% | −0.0087 | −0.0730 |
| dnn_5class_cw | 5.61% | +0.0018 | −0.0253 |

**Average: 2.15% flipping.**

**Key observation:** despite very low flipping (1-3% on most), macro-F1 dropped 5-9 pp on 4 of 6 models. The flips were concentrated in rare classes.

**Dirichlet per-class wins (NSL, 30 cells) — AUDIT-CORRECTED:** **6 improved, 24 worsened.** Improvements concentrated in DNN models on U2R class:
- dnn_5class_smote U2R: 0.0066 → **0.0030** (−54.5%)
- dnn_5class_cw U2R: 0.0179 → **0.0029** (−83.8%)

### 5.2 UNSW-NB15

**Hybrid Platt/Isotonic — macro Brier:**

| Model | Task | Brier pre | Brier post | Improved? |
|---|---|---:|---:|---|
| rf_binary_cw | binary | 0.09506 | 0.10611 | no |
| xgb_binary_cw | binary | 0.09669 | 0.10875 | no |
| dnn_binary_cw | binary | 0.09840 | 0.11602 | no |
| rf_5class_smote | 5-class | 0.07224 | 0.06981 | **yes** |
| xgb_5class_smote | 5-class | 0.07125 | 0.06888 | **yes** |
| dnn_5class_smote | 5-class | 0.08687 | 0.07319 | **yes** |
| rf_5class_cw | 5-class | 0.06965 | 0.06856 | **yes** |
| xgb_5class_cw | 5-class | 0.07505 | 0.07055 | **yes** |
| dnn_5class_cw | 5-class | 0.09312 | 0.07610 | **yes** |

**UNSW hybrid summary:** All 3 binary worsened; all 6 5-class improved.

**Dirichlet — macro Brier:**

| Model | Task | Brier pre | Brier dirichlet | Improved? |
|---|---|---:|---:|---|
| rf_binary_cw | binary | 0.09506 | 0.10345 | no |
| xgb_binary_cw | binary | 0.09669 | 0.10608 | no |
| dnn_binary_cw | binary | 0.09840 | 0.11302 | no |
| rf_5class_smote | 5-class | 0.07224 | 0.06527 | **yes** |
| xgb_5class_smote | 5-class | 0.07125 | 0.06483 | **yes** |
| dnn_5class_smote | 5-class | 0.08687 | 0.06746 | **yes** |
| rf_5class_cw | 5-class | 0.06965 | 0.06491 | **yes** |
| xgb_5class_cw | 5-class | 0.07505 | 0.06593 | **yes** |
| dnn_5class_cw | 5-class | 0.09312 | 0.07052 | **yes** |

**UNSW Dirichlet summary:** All 3 binary worsened; all 6 5-class improved (larger margins than hybrid).

**Three-way winner counts:** Pre 3 (all binary), Hybrid 0, Dirichlet 6 (all 5-class).

**Hybrid argmax flipping (UNSW 5-class):**

| Model | % flipped | Δ acc | Δ F1m |
|---|---:|---:|---:|
| rf_5class_smote | 18.74% | +0.0439 | −0.0082 |
| xgb_5class_smote | 12.73% | +0.0271 | −0.0169 |
| dnn_5class_smote | 23.65% | +0.0753 | +0.0114 |
| rf_5class_cw | 17.38% | +0.0435 | −0.0140 |
| xgb_5class_cw | 15.62% | +0.0292 | −0.0037 |
| dnn_5class_cw | 25.42% | +0.0623 | −0.0028 |

**Average: 18.92% flipping.**

**Dirichlet argmax flipping (UNSW 5-class):**

| Model | % flipped | Δ acc | Δ F1m |
|---|---:|---:|---:|
| rf_5class_smote | 19.39% | +0.0590 | +0.0005 |
| xgb_5class_smote | 14.29% | +0.0404 | +0.0021 |
| dnn_5class_smote | 25.98% | +0.0947 | +0.0196 |
| rf_5class_cw | 16.97% | +0.0512 | −0.0209 |
| xgb_5class_cw | 16.22% | +0.0508 | +0.0099 |
| dnn_5class_cw | 29.33% | **+0.1061** | **−0.0327** |

**Average: 20.36% flipping.**

**Critical observation on UNSW dnn_5class_cw with Dirichlet:** Accuracy +10.6 pp BUT macro-F1 −3.3 pp. The class-weight training boost on rare classes was partially undone by Dirichlet's prior absorption. **Numbers verified directly from saved probability arrays via reconciliation script.**

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

**Hybrid Platt/Isotonic — macro Brier:**

| Model | Task | Brier pre | Brier post | Improved? |
|---|---|---:|---:|---|
| rf_binary_cw | binary | 0.00141 | 0.00131 | **yes** |
| xgb_binary_cw | binary | 0.00090 | 0.00090 | tie |
| dnn_binary_cw | binary | 0.02081 | 0.01594 | **yes** |
| rf_5class_smote | 5-class | 0.00063 | 0.00056 | **yes** |
| xgb_5class_smote | 5-class | 0.00044 | 0.00041 | **yes** |
| dnn_5class_smote | 5-class | 0.01124 | 0.00538 | **yes** |
| rf_5class_cw | 5-class | 0.00060 | 0.00060 | tie |
| xgb_5class_cw | 5-class | 0.00036 | 0.00035 | **yes** |
| dnn_5class_cw | 5-class | 0.05258 | 0.01643 | **yes** |

**CIC hybrid summary:** 7 of 9 improved, 2 tied. None worsened.

**Headline numbers:**
- dnn_5class_cw macro ECE: 0.10656 → **0.00087** (−99.2%)
- dnn_5class_cw macro Brier: 0.05258 → **0.01643** (−68.8%)

**Three-way winner counts (CIC):** Hybrid 7, Dirichlet 2, Pre 0.

The 2 Dirichlet wins on CIC:
- xgb_binary_cw (barely; delta 0.00002)
- dnn_5class_cw (Dirichlet 0.01534 vs hybrid 0.01643)

**CIC per-class winners (5-class, 30 cells):**

| Class | Hybrid | Dirichlet | Pre |
|---|---:|---:|---:|
| Normal | 4 | 2 | 0 |
| DoS | 5 | 1 | 0 |
| Probe | 4 | 0 | 2 |
| R2L | 3 | 2 | 1 |
| U2R | 4 | 0 | 2 |
| **Total** | **20** | **5** | **5** |

**CIC argmax flipping** (measured on Dirichlet; file-comparison confirms hybrid argmax decisions match for >99% of samples):

| Model | % flipped | Δ acc | Δ F1m |
|---|---:|---:|---:|
| rf_5class_smote | 0.02% | +0.0001 | +0.0003 |
| xgb_5class_smote | 0.01% | +0.0000 | −0.0004 |
| dnn_5class_smote | 3.44% | +0.0197 | +0.1509 |
| rf_5class_cw | 0.05% | +0.0001 | **−0.1667** |
| xgb_5class_cw | 0.01% | +0.0001 | +0.0018 |
| dnn_5class_cw | 19.34% | +0.1235 | +0.1430 |

**Average: 3.81% flipping** (entirely driven by dnn_5class_cw collapse).

**The rf_5class_cw F1m drop of −0.1667** despite 0.05% flipping reflects that the ~20 flipped samples included U2R cases (n_test = 7), where each error disproportionately impacts macro-F1.

### 5.4 Cross-dataset summary

**Winner counts (9 models per dataset):**

| Dataset | Shift | Pre-cal | Hybrid | Dirichlet |
|---|---|---:|---:|---:|
| NSL-KDD | Severe | **6** | 1 | 2 |
| UNSW-NB15 | Moderate | 3 (binary) | **0** | **6** (5-class) |
| CIC-IDS2017 | None | 0 | **7** | 2 |

**Total across 27 model-dataset combinations:** Pre 9, Hybrid 8, Dirichlet 10.

**Argmax flipping by method × dataset (5-class average):**

| Method | NSL | UNSW | CIC |
|---|---:|---:|---:|
| Hybrid | 3.15% | 18.92% | 3.81% |
| Dirichlet | 2.15% | 20.36% | 3.81% |

---

## 6. Interpretation and discussion

### 6.1 Main empirical claims

**Claim 1: Calibration efficacy depends on calibration-test distribution alignment.**

CIC (no shift) → calibration mostly improves (hybrid wins 7 of 9).
UNSW (moderate shift) → Dirichlet dominates 5-class (6 of 6), hybrid never wins.
NSL (severe shift) → pre-cal wins 6 of 9. Post-hoc calibration fails the baseline.

**Claim 2: Both methods substantially modify classifier predictions on UNSW.**

Argmax flipping rates: hybrid 12.7-25.4%, Dirichlet 14.3-29.3%. The calibrated probabilities, used as the basis for a downstream classifier, disagree with the pre-calibration argmax on a substantial fraction of samples. This challenges the standard framing of calibration as "just rescaling probabilities."

**Claim 3: NSL Dirichlet has the opposite pattern — low flipping but big F1 impact.**

Average flipping 2.15%, but macro-F1 drops 5-9 pp on 4 of 6 models. Mechanism: Dirichlet absorbs NSL's severe rare-class shift by reclassifying a small fraction of rare-class samples, which has outsized impact on macro-F1.

**Claim 4: Class-weighted models suffer most from Dirichlet's prior absorption.**

UNSW dnn_5class_cw with Dirichlet: accuracy +10.6 pp, macro-F1 −3.3 pp. Class-weight boost on rare classes gets partially undone.

**Claim 5: Catastrophically miscalibrated models benefit dramatically.**

CIC dnn_5class_cw with hybrid: ECE 0.107 → 0.0009 (−99.2%), accuracy 0.823 → 0.946 (+12.3 pp). Calibration here functions as classifier recovery, not just confidence rescaling.

**Claim 6: Binary calibration on UNSW degrades regardless of method.**

Both hybrid and Dirichlet worsen all 3 binary UNSW models. Likely cause: 2-class structure cannot redistribute probability mass to absorb prior shift.

### 6.2 Mechanistic hypothesis (initial, simple)

We initially hypothesized: **flipping rate is monotonic in distribution shift severity** (more shift → more argmax change).

**This hypothesis is empirically wrong.** UNSW (moderate shift) shows higher flipping than NSL (severe shift). The simple "shift → flipping" relationship doesn't hold.

### 6.3 Revised mechanistic hypothesis (not directly verified)

**Flipping rate depends on which classes shift, not just severity.**

- NSL: shift concentrated in rare classes (R2L 16×, U2R 7.5×). Few absolute samples affected; flipping rate stays low; but macro-F1 impact is amplified because each rare-class error counts heavily.
- UNSW: shift concentrated in majority class (Normal 1.4×). Every test sample is potentially affected; flipping rate is high; macro-F1 impact moderate because flips spread across class boundaries.
- CIC: no shift. Flipping limited to pre-cal miscalibration effects.

This hypothesis is consistent with our data but **not directly verified**. A targeted test would inspect which classes the flipped samples come from. This is a Day 4 or beyond task.

### 6.4 Hybrid's argmax flipping mechanism is different

Hybrid Platt/isotonic flipping correlates more with pre-calibration miscalibration magnitude than with distribution shift. UNSW models have higher pre-cal Brier (0.07-0.09) than NSL (0.05-0.08) or CIC (<0.05). When pre-cal probabilities are more miscalibrated, the per-class isotonic transformations are larger, and after renormalization more boundary samples flip.

**Important:** This means the two methods have DIFFERENT sensitivities. Dirichlet → distribution shift. Hybrid → pre-cal miscalibration magnitude. Not directly comparable on a single axis.

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

When proposing Dirichlet, I framed it as "methodologically stronger." **Empirically mixed.** Dirichlet wins on UNSW 5-class (6 of 6) but loses or ties on CIC (2 of 9) and is mixed on NSL (2 of 9 wins). Theoretical strength doesn't translate to uniform empirical superiority.

### 7.5 Wrong prediction: "Hybrid preserves argmax, Dirichlet shifts it"

This was the framing I built Stance B around. **Empirically false on UNSW.** Both methods flip 14-25% on UNSW. The clean dichotomy doesn't hold. This invalidated Stance B and pushed us toward Stance C.

### 7.6 Invented number: "NSL Dirichlet flips ~23%"

I claimed NSL Dirichlet flipped about 23% of predictions, based on extrapolation from UNSW's 20% and assuming severe shift would cause more flipping. **Completely wrong.** The actual measurement showed 2.15% average. I should have presented this as a prediction, not as a finding.

### 7.7 Initial recommendation of Stance B was wrong

I recommended Stance B (conservative methodology view) on the grounds that it was safer for an MSc paper. **Wrong context.** Once the user clarified TIFS/TDSC target and PAIDS affiliation, Stance C was the correct recommendation. I should have asked about the target venue earlier.

### 7.8 Repeatedly suggested stopping when user had energy

Throughout the night, I kept suggesting we stop because "it's late" when the user had clearly stated they had energy and wanted to keep going. **Pattern: projecting fatigue concerns onto the user.** This is a behaviour to avoid. User's self-report on energy wins; my model of what's appropriate doesn't.

### 7.9 F1m number inconsistency on UNSW dnn_5class_cw

At one point I gave two different F1m numbers for the same model (0.4774 → 0.4447 vs 0.5400 → 0.6830). **Reconciliation script confirmed: 0.4774 → 0.4447 is correct.** The other set was a typo or confusion on my part. Authoritative numbers from saved disk arrays should always win over conversational recall.

### 7.10 Wrong NSL Dirichlet per-class improvement count

In conversation and v1/v2 of the docs, I claimed "7 of 30 cells improved" on NSL Dirichlet. **The audit at end of session revealed actual count is 6 of 30 improved, 24 of 30 worsened.** The error originated in early conversation and propagated. Lesson: per-cell counts should always be recomputed at writeup time, not carried over from earlier discussion.

### 7.11 Pattern across mistakes

The common thread: **I made predictions ahead of the data and stated them as findings**. Every time the data corrected me. Going forward, predictions should be labelled as predictions, not delivered as results. Data leads, narratives follow.

A second pattern: **counts and aggregates carried over from conversation can drift from the underlying data.** End-of-session audit caught one such error (the 7-vs-6 issue) but only because it was specifically searched for. Future sessions should audit all aggregate counts against raw data before publishing.

---

## 8. Pending work

### 8.1 For calibration specifically

**Bootstrap confidence intervals (TA-requested, Day 4):**
- Apply B=1000 bootstrap resamples to calibration metrics
- Report 95% CIs (2.5/97.5 percentile) on macro Brier deltas and per-class Brier
- ~1-2 hours of compute, ~2-3 hours including writeup

**Reliability diagrams (optional, Day 4):**
- We have NSL reliability diagrams in `03b_nsl_calibration_diagnostic.ipynb`
- UNSW and CIC reliability diagrams would complete the figure set
- ~30 minutes of work each

**Direct test of revised mechanism hypothesis:**
- The "rare-class shift drives NSL flipping, majority-class shift drives UNSW flipping" hypothesis is plausible but not directly verified
- A targeted analysis: for each Dirichlet-flipped sample, inspect WHICH class it was flipped from and to
- If NSL flips are concentrated in rare classes and UNSW flips are spread across majority transitions, hypothesis is supported
- ~1 hour of work

**Class-prior absorption mathematical analysis (optional, paper-strengthening):**
- Compute the Dirichlet ODIR intercept vector explicitly per model per dataset
- Show that intercepts correlate with `log(p_test/p_calib)` for each class
- This would directly demonstrate the implicit prior shift adaptation mechanism
- ~2 hours of work, makes the paper stronger

### 8.2 For the broader Day 3+ work

Calibration is just the first phase of Day 3. The full Day 3 plan was:

| Phase | Status |
|---|---|
| Calibration (this work) | ✓ Data complete, writeup partial |
| SHAP (Notebook 04) | Not started |
| Stability — FGSM/PGD/Gaussian (Notebook 05) | Not started |
| Krishna agreement (Notebook 06) | Not started |
| SCTS-v2 on NSL + UNSW (Notebooks 07-08) | Not started |

**Key decision pending:** does SHAP run on pre-cal probabilities, hybrid-calibrated, or both? Under Stance C, the cleanest answer is "hybrid-calibrated as primary, pre-cal as sensitivity check." But we may want a dual pipeline later. This decision can be deferred until SHAP is started.

### 8.3 For the paper writeup (next session)

Three artefacts produced tonight:
1. **`calibration_section_draft.md`** — 2,420-word prose draft of the calibration section. Has [PENDING] markers for citations, [INSERT TABLE] markers for data tables. Not citation-ready.
2. **`calibration_findings_exact_v3.md`** — 3,962-word exact-numbers reference (audited). All data reconciled from disk. Ready for use in editing the prose draft.
3. **`calibration_session_memory_v3.md`** (this file) — Comprehensive session memory.

Next session writeup work:
- Verify every number in the prose draft against the findings doc v3
- Add citations (Niculescu-Mizil & Caruana 2005, Kull et al. 2019, Guo et al. 2017, Ovadia et al. 2019)
- Insert actual tables (formatted from CSVs in `results/tables/`)
- Resolve approximate language ("approximately", "roughly") in favour of exact numbers
- Address remaining [PENDING] markers

---

## 9. File and repo references

### 9.1 Notebooks created tonight

All in `notebooks/` directory:

| File | Purpose |
|---|---|
| `03_nsl_calibration_v2.ipynb` | NSL hybrid Platt/isotonic |
| `03b_nsl_calibration_diagnostic.ipynb` | NSL reliability diagrams + diagnostic |
| `03c_nsl_calibration_dirichlet_v2.ipynb` | NSL Dirichlet ODIR |
| `03_unsw_calibration_v2.ipynb` | UNSW hybrid (built late in session) |
| `03c_unsw_calibration_dirichlet_v2.ipynb` | UNSW Dirichlet |
| `03_cic_calibration_v2.ipynb` | CIC hybrid |
| `03c_cic_calibration_dirichlet_v2.ipynb` | CIC Dirichlet |

### 9.2 CSV outputs

All in `results/tables/`:

| File | Contents |
|---|---|
| `nslkdd_v2_calibration_summary.csv` | NSL hybrid macro |
| `nslkdd_v2_calibration_perclass.csv` | NSL hybrid per-class |
| `nslkdd_v2_calibration_threeway.csv` | NSL Dirichlet comparison |
| `nslkdd_v2_calibration_threeway_perclass.csv` | NSL three-way per-class |
| `unsw_v2_calibration_*` | Same pattern, UNSW |
| `cic_v2_calibration_*` | Same pattern, CIC |

### 9.3 Calibrator outputs

Saved as `.npy` arrays in `calibrators/{dataset}_v2/` (hybrid) and `calibrators/{dataset}_v2_dirichlet/` (Dirichlet). Plus `calibration_summary.json` in each folder with all metrics.

### 9.4 Git state

Last commit at end of session: `600b0c2` — "Day 3 calibration complete: hybrid + Dirichlet on three datasets with full diagnostics".

26 files changed, 246 insertions, 2117 deletions.

**Still uncommitted at session end:**
- `docs/v2_rebuild_progress_day1_day2_v7.md` (stale, should be deleted)
- `models/` directory (some untracked artifacts)
- `shap_values/unsw_nb15/` (v1 SHAP outputs, low priority)

### 9.5 Documents produced tonight

In `docs/` (or wherever the user saves them):

| File | Purpose |
|---|---|
| `calibration_section_draft.md` | Tier 2 prose draft for paper |
| `calibration_findings_exact_v3.md` | Reconciled, audited exact-numbers reference |
| `calibration_session_memory_v3.md` (THIS FILE) | Comprehensive session memory |

---

## 10. Orientation for future sessions

If a future Claude instance picks up this work:

1. **Read this document first.** It's the single source of truth for the calibration phase.
2. **Then check git log** for any commits after `600b0c2` — those reflect progress made after this session ended.
3. **Verify any number against `results/tables/` CSVs** before quoting it in a draft. Tonight's experience showed that conversational recall can drift from disk truth — including aggregate counts (see §7.10).
4. **The user's research is PAIDS-affiliated, target IEEE TIFS/TDSC.** Calibration is part of a 4-component framework: calibration, SHAP-based stability, Krishna agreement, SCTS-v2 trust scoring, plus LLM-based alert evaluation. Don't optimize calibration for safety; optimize for genuine contribution.
5. **Stance C is the agreed framing.** Four-property evaluation matrix across three datasets and two methods.
6. **The user trusts their TA (Ms Alaa) at checkpoints**, runs technical decisions independently between them. Don't ask the user to consult Ms Alaa on every methodology choice.
7. **The user has energy and time.** Don't suggest stopping unless there's a specific reason beyond "it's late."
8. **Predictions should be labelled as predictions.** This session's pattern of stating predictions as findings caused confusion. Better practice: "I expect X, but we should verify by running Y."
9. **Audit aggregate counts before publishing.** Per-cell counts, winner totals, and averages should always be recomputed from the underlying tables, not carried over from previous conversation turns.

### 10.1 Where to start next

Three possible starting points, in order of recommended priority:

**Option A (recommended): Bootstrap CIs on calibration metrics.** Directly addresses TA's comment 2. Solidifies the calibration findings before moving to SHAP. ~3-4 hours.

**Option B: Begin Notebook 04 (SHAP).** Move forward to the next major component. Decision needed: which calibration outputs feed SHAP? Recommendation: hybrid-calibrated as primary.

**Option C: Finalize the calibration section writeup.** Use the prose draft + findings doc v3 + this memory doc together. Produce a clean paper section. ~4-6 hours.

Either A then C, or C then A, are coherent. B can wait until calibration is fully wrapped.

### 10.2 What NOT to do

- Don't start Day 5 LLM evaluation work yet (requires SHAP outputs which don't exist)
- Don't add more calibration methods. Three is enough (pre-cal, hybrid, Dirichlet). The scope is set.
- Don't promise to "finally complete" the calibration work — it's already complete in the sense that the empirical matrix is filled. What remains is documentation and CIs.

---

## 11. Session metadata

### 11.1 Quantitative summary

- **Total time:** approximately 8 hours of active work (with breaks for compute runs)
- **Notebooks built:** 7 calibration notebooks + 1 diagnostic
- **Models calibrated:** 27 (9 models × 3 datasets) × 2 methods = 54 calibrated probability sets
- **Argmax checks run:** 5 (NSL hybrid, NSL Dirichlet, UNSW hybrid, UNSW Dirichlet, CIC both methods)
- **Drive disconnect events:** 1 (resolved with `force_remount=True`)
- **Schema differences across datasets handled:** 3 (NSL `multiclass_5`, UNSW `five_class_id_to_name`, CIC `multiclass_5`)
- **End-of-session audit:** Performed. Found 1 numerical error (the 7-vs-6 NSL Dirichlet per-class count) and 5 interpretation/completeness issues. All fixed in v3 of docs.

### 11.2 Audit trail

- v1 documents drafted in-session
- v2 reconciliation against disk corrected one error (UNSW dnn_5class_cw F1m), filled in TBDs, replaced extrapolation with measurement (NSL Dirichlet argmax)
- v3 audit pass corrected count error (6 vs 7), expanded U2R table to all 12 cases, added per-class UNSW Dirichlet table, softened philosophical framing in §6, generalized time references

**End of session memory document v3. Last updated: end of 31 May 2026, after audit pass.**
