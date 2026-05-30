# Calibration → SHAP → Krishna → Stability → SCTS-v2: Session Memory v11

**Status as of 2026-05-30 (Day 3 complete + framing decision):** All Day 3 components sealed including SCTS-v2 trust score with Mondrian per-class conformal. Paper framing decision locked: Framing B (multi-dataset honest evaluation; NSL as cautionary case study).

**Repository:** github.com/anasbiswas1/xids-research
**Last commits (in order):**
- `a1f9be9` Notebook 04c: canonical SHAP recompute on 1000 shared samples per dataset
- `d61fbdc` Notebook 06v3: Krishna cross-model agreement (canonical, K-sweep 5/10/15/20)
- `512b1f2` Canonical eval indices force-added (was hitting .gitignore)
- `d7df8e7` Notebook 05c: adversarial stability (canonical, magnitudes verified, Lipschitz with XGB log-odds note)
- `4a2012c` Session memory v9: canonical samples + Krishna sealed
- `8c8cfbb` Session memory v10: adversarial stability sealed
- `4e80987` Notebook 07c: SCTS-v2 trust score on canonical samples (initial marginal-conformal version)
- `6258826` Notebook 07c: SCTS-v2 with Mondrian per-class conformal (intermediate)
- **`bc85188` Notebook 07c: SCTS-v2 with Mondrian per-class conformal (final committed version + 5 result files)**

## What v11 adds beyond v10

v10 closed with Day 3 needing **Notebook 07c (SCTS-v2)** still to build. v11 documents that work and the paper-framing decision it forced.

This session iterated through three SCTS-v2 designs (marginal → Mondrian → tested SCTS-v3 feasibility), found that **none fix the underlying problem** that SCTS-v2 measures prediction self-coherence rather than correctness, and committed Mondrian 07c with honest documentation of where it works and where it doesn't. The framing decision (Framing B: present all 3 datasets including the failure as a case study) makes the paper reviewer-bulletproof.

Four new audit catches added (#12 — #15). Audit total now 15.

---

# §18 — SCTS-v2 Trust Score (Notebook 07c)

## §18.1 — What got built and where it lives

**Notebook:** `notebooks/07c_scts_canonical.ipynb` (committed at `bc85188`)

Also: `notebooks/07c_scts_canonical_mondrian_v2.ipynb` untracked sibling (identical content; the v2-named file was the one uploaded and run; the main filename was renamed-into-place from it via Colab's file upload routing).

**Outputs (all in `results/tables/`):**
- `scts_v2_canonical.csv` (1.98 MB, 18,000 rows = 18 models × 1000 canonical samples; per-sample SCTS, c₁, c₂, c₃, correct flag, predicted/true class)
- `scts_v2_per_class_summary.csv` (11.3 KB, 90 rows; per (dataset, model, true_class) mean SCTS and components)
- `scts_v2_alpha_sensitivity.csv` (6.2 KB, 54 rows; α ∈ {0.05, 0.10, 0.20} per model)
- `scts_v2_validation.csv` (5.8 KB, 72 rows; SCTS quartile bins × per-model accuracy)
- `scts_v2_summary.json` (12.1 KB; aggregate stats, conformal thresholds dict, alpha sensitivity)

**Methodology (final committed version):**
- c₁ = calibrated probability of predicted class (from pre-saved `_test_proba_calibrated.npy`, verified identical to v8 refit hybrid calibrators)
- c₂ = worst-case (min) per-sample Jaccard top-10 across {Gaussian, FGSM, PGD} perturbations from 05c
- c₃ = per-sample Mondrian conformal score at α=0.05 (per-class threshold stratified by PREDICTED class; fallback to marginal threshold if n_calib_class < 30)
- SCTS-v2 = (c₁ · c₂ · c₃)^(1/3) · 100 with eps=1e-6 floor

**Conformal calibration split:** All test samples NOT in canonical 1000 are used to fit Mondrian thresholds (~21,544 NSL / 62,461 UNSW / 39,007 CIC). Canonical 1000 per dataset are scored. No sample leakage.

## §18.2 — First implementation: marginal split-conformal

Initial 07c used marginal (single-threshold) split-conformal per Romano et al. 2019. Threshold = ceil((n+1)(1-α))/n quantile of nonconformity scores `1 - p̂(y_true)`.

**Headline numbers (marginal version):**
- Overall mean SCTS: 66.6 (median 74.3, std 21.1, range 0.36–100)
- Mean components: c₁=0.894, c₂=0.478, c₃=0.804
- Mean Pearson correlation SCTS-vs-correctness across 18 models: **+0.272**
- 17/18 models with positive Pearson (only NSL rf_5class_smote slightly negative at -0.056)

**Per-dataset Pearson (marginal):**
- NSL mean: +0.11 (range -0.06 to +0.26) — weak
- UNSW mean: +0.52 (range +0.47 to +0.57) — strong, SCTS works
- CIC mean: +0.18 (range +0.10 to +0.43, DNN best) — moderate

**Conformal threshold patterns (marginal, α=0.05):**
- NSL: threshold = 1.000 across ALL 6 models, empirical coverage = 1.000 (degenerate)
- UNSW: threshold = 0.88-0.89, empirical coverage = 0.88-0.93 (healthy)
- CIC: threshold = 0.001-0.006 for tree models, 0.19-0.49 for DNN (tree thresholds are over-strict)

**Per-class ordering across 18 models (marginal):**
| Class | mean SCTS | mean c₁ | mean c₂ | mean c₃ | accuracy | n |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 75.85 | 0.950 | 0.512 | 0.932 | 0.889 | 4890 |
| DoS | 63.44 | 0.842 | 0.497 | 0.731 | 0.615 | 4068 |
| Probe | 63.60 | 0.908 | 0.446 | 0.789 | 0.793 | 3720 |
| R2L | 60.97 | 0.875 | 0.467 | 0.721 | 0.598 | 3678 |
| U2R | 59.77 | 0.872 | 0.478 | 0.703 | 0.456 | 1644 |

Across all 18 models, SCTS correctly orders classes by trustworthiness: Normal (highest SCTS, highest accuracy) → DoS/Probe (middle) → R2L → U2R (lowest SCTS, lowest accuracy). **Gap Normal–U2R: 16.1 points.** This is the publishable per-class finding.

## §18.3 — Diagnosis: NSL conformal degeneracy

The threshold=1.0 on all 6 NSL models led to diagnostic. Conformal quantile at α=0.05 hits 1.0 iff ≥5% of calibration samples have nonconformity score `1 - p̂(y_true) = 1.0`, i.e., `p̂(y_true) = 0`.

**Measured fraction with score=1.0 on NSL test set (21,544 samples, rf_5class_smote):**
- Overall: 17.7% (3,812 samples)
- Per-class breakdown:
  - Normal: 2.4% (224 of 9,450)
  - DoS: 14.1% (1,020 of 7,212)
  - Probe: 6.7% (147 of 2,210)
  - **R2L: 90.6% (2,421 of 2,672)** ← catastrophic
  - U2R: 0% (no U2R samples in calibration set; all 67 are in canonical 1000)

**Interpretation:** The v8 hybrid calibrators (refit, verified working) assign calibrated probability of zero to the true class for **90.6% of R2L test samples**. The model's one-vs-rest calibrator for R2L learns from only 199 calibration samples and effectively predicts "this is NOT R2L" with probability 1.0 for most actual R2L cases.

This is a **multi-class calibration limit on imbalanced data**, not a 07c bug. The v8 MODERATE verdict on calibration robustness was about rank stability of SHAP attributions, not about absolute probability mass on rare true classes.

## §18.4 — Three investigations rejected before Mondrian

Before implementing Mondrian, three simpler tweaks were tested:

**Investigation A — c₂ min vs mean:** Replaced worst-case (min) Jaccard with average across perturbations. Mean Pearson: marginal=0.272, mean-c₂=0.277. **Δ = +0.005 (noise level).** Choice doesn't matter. Min retained for consistency with existing 07_scts_v2.

**Investigation B — Threshold floor (raise low thresholds):** Tested floor ∈ {0.3, 0.5} to rescue CIC's over-strict thresholds. Effect on CIC Pearson at floor=0.5: -0.036 (c₂=mean) to -0.048 (c₂=min) vs no-floor baseline (hurts in both cases). Effect on NSL: zero (NSL's 1.0 threshold is already above any floor; floor cannot reduce a threshold). Per-class gap (Normal–U2R) compressed from 16.1 → 6.5 (also hurts). **Rejected.**

**Investigation C — Threshold ceiling (cap high thresholds):** Tested ceiling ∈ {0.3, 0.5, 0.7} to rescue NSL's degenerate 1.0 threshold. NSL Pearson improvement:
- No ceiling: +0.113
- Ceiling=0.7: +0.125 (Δ=+0.012)
- Ceiling=0.5: +0.135 (Δ=+0.022)
- Ceiling=0.3: +0.139 (Δ=+0.026)

Even aggressive ceiling of 0.3 (cutting threshold by 70%) gave only +0.026 Pearson — within noise. **NSL's weak SCTS-correctness correlation is not because of conformal threshold value; it's because c₁ is overconfident for misclassified rare-class predictions and c₂/c₃ can't compensate.** Rejected.

These three investigations took ~25 minutes and saved building a much larger fix that wouldn't have rescued NSL either.

## §18.5 — Mondrian per-class conformal (the chosen methodology)

Stratify conformal threshold by **predicted class** (deployment-feasible since true class is unknown at test time). Per Vovk et al. 2005 and Boström 2020. For each predicted class c:
1. Subset calibration samples where model predicted class c
2. Compute their nonconformity scores `1 - p̂(y_true)`
3. Threshold for samples predicted as c = 95th percentile of those scores
4. **Fallback rule:** If n_calib_c < 30 (matching v5/v8 hybrid calibrator boundary), use marginal threshold for that class

At test time, for each sample, look up threshold by predicted class.

**Why Mondrian over Adaptive Prediction Sets (APS) of Romano et al. 2020:** APS uses cumulative softmax probability mass as the nonconformity score. For our specific failure mode — calibrators assigning `p̂(y_true) = 0` to misclassified rare-class samples — APS faces the same degeneracy as marginal split-conformal: when the true class has zero probability, both `1 − p̂(y_true)` (LAC/marginal) and the cumulative-sum-to-true-class (APS) hit 1.0. Mondrian addresses a different axis of the problem (heterogeneity across predicted classes), which is the part of the failure mode that can be addressed without re-fitting the underlying calibrators. RAPS (Angelopoulos et al. 2021) was also considered but adds a regularization hyperparameter our small-data-per-class regime would struggle to tune.

**Why fallback threshold = 30:** Matches the same statistical floor used in v5/v8 hybrid calibration (Platt-vs-isotonic boundary). Conventional small-sample practice for stable quantile estimation places the floor near n=30, which corresponds to the rule-of-thumb for trustworthy percentile estimation. Using the same constant across the pipeline maintains methodological consistency that's easier to defend in a paper.

### Per-class threshold patterns (Mondrian, α=0.05)

**NSL (all 6 models show same pattern):**
| Predicted class | Mondrian threshold | Note |
|---|---:|---|
| Normal | 1.00 | Still degenerate — R2L-predicted-as-Normal samples push q95 to 1 |
| DoS | 0.29–0.49 | Functional — Mondrian works for DoS predictions |
| Probe | 0.99–1.00 | Still degenerate |
| R2L | 0.38–0.49 | Functional — Mondrian works for R2L predictions |
| U2R | 1.00 (fallback) | n_calib_U2R < 30, falls back to marginal=1.0 |

`fb=1/5` is the typical pattern (only U2R falls back). One model (dnn_5class_smote) showed `fb=0/5` with Mondrian U2R threshold = 1.00 — this means n_calib_U2R was ≥30 (no fallback triggered), but the actual per-class q95 still hit 1.0 from real samples in the predicted-U2R group. Either edge-case behavior; both produce the same downstream effect (degenerate threshold for U2R predictions). Per-class n values logged in `conformal_thresholds.json` for verification.

**UNSW (typical):** `mondrian=[0.49, 0.69, 0.91, 0.90, 1.00]` — diverse, healthy thresholds per class

**CIC (typical):** `mondrian=[0.00-0.01, 0.01-0.02, 0.01-0.04, 0.00-1.00, 0.00-0.01]` with fallback often triggered (`fb=2/5`) — per-class still mostly over-strict but more variation than marginal

### Mondrian results: per-dataset Pearson change

| Model | Marginal Pearson | Mondrian Pearson | Δ |
|---|---:|---:|---:|
| NSL rf_5class_cw | +0.008 | -0.032 | -0.04 |
| NSL xgb_5class_cw | +0.210 | +0.120 | -0.09 |
| NSL dnn_5class_cw | +0.040 | -0.005 | -0.04 |
| NSL rf_5class_smote | -0.056 | -0.100 | -0.04 |
| NSL xgb_5class_smote | +0.264 | +0.146 | -0.12 |
| NSL dnn_5class_smote | +0.214 | +0.202 | -0.01 |
| **NSL mean** | **+0.11** | **+0.06** | **-0.06** |
| UNSW rf_5class_cw | +0.496 | +0.453 | -0.04 |
| UNSW xgb_5class_cw | +0.538 | +0.474 | -0.06 |
| UNSW dnn_5class_cw | +0.466 | +0.435 | -0.03 |
| UNSW rf_5class_smote | +0.555 | +0.504 | -0.05 |
| UNSW xgb_5class_smote | +0.572 | +0.508 | -0.06 |
| UNSW dnn_5class_smote | +0.496 | +0.416 | -0.08 |
| **UNSW mean** | **+0.52** | **+0.47** | **-0.05** |
| CIC rf_5class_cw | +0.116 | +0.200 | +0.08 |
| CIC xgb_5class_cw | +0.104 | +0.217 | +0.11 |
| CIC dnn_5class_cw | +0.429 | +0.484 | +0.05 |
| CIC rf_5class_smote | +0.114 | +0.282 | +0.17 |
| CIC xgb_5class_smote | +0.096 | +0.257 | +0.16 |
| CIC dnn_5class_smote | +0.242 | +0.471 | +0.23 |
| **CIC mean** | **+0.18** | **+0.32** | **+0.14** |
| **OVERALL mean** | **+0.272** | **+0.280** | **+0.01** |
| Models with positive Pearson | 17/18 | 15/18 | -2 |

**Honest Mondrian summary:** Substantial improvement on CIC (+0.14 mean Pearson), small regression on UNSW (-0.05), neutral-to-worse on NSL (-0.06). Net Pearson nearly unchanged (+0.008). **3 NSL models flipped from positive to negative Pearson under Mondrian.**

### Mondrian results: per-dataset, per-class

**NSL aggregated across 6 models (n in each = 6 × per-class count):**
| True class | mean SCTS | mean c₁ | mean c₂ | mean c₃ | accuracy | n |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 80.9 | 0.995 | 0.544 | 0.994 | 0.960 | 1566 |
| DoS | 77.5 | 0.960 | 0.560 | 0.927 | 0.839 | 1488 |
| Probe | 78.8 | 0.925 | 0.596 | 0.919 | 0.677 | 1266 |
| **R2L** | **77.2** | 0.954 | 0.546 | 0.935 | **0.067** | 1278 |
| U2R | 74.6 | 0.914 | 0.534 | 0.896 | 0.169 | 402 |

**The R2L row is the critical failure:** Mondrian SCTS=77.2 but accuracy=6.7%. Reviewer-visible "trust score says trust 77/100, model is wrong 93% of the time." Marginal version was SCTS=77.4 / acc=7.5% — Mondrian barely moved this.

**UNSW aggregated across 6 models:**
| True class | mean SCTS | mean c₁ | mean c₂ | mean c₃ | accuracy | n |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 68.7 | 0.864 | 0.512 | 0.809 | 0.716 | 1200 |
| DoS | 53.4 | 0.594 | 0.490 | 0.546 | 0.043 | 1200 |
| Probe | 63.3 | 0.837 | 0.400 | 0.820 | 0.713 | 1200 |
| R2L | 59.7 | 0.716 | 0.493 | 0.660 | 0.882 | 1200 |
| U2R | 58.1 | 0.740 | 0.384 | 0.724 | 0.650 | 1200 |

UNSW DoS has the strikingly low accuracy (0.043) on canonical 1000 — this is the "hardest-to-trust" UNSW finding scheduled for Day 4 bootstrap CI work.

**CIC aggregated across 6 models:**
| True class | mean SCTS | mean c₁ | mean c₂ | mean c₃ | accuracy | n |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 71.4 | 0.991 | 0.480 | 0.882 | 0.991 | 2124 |
| DoS | 66.8 | 0.972 | 0.442 | 0.844 | 0.962 | 1380 |
| Probe | 61.0 | 0.961 | 0.344 | 0.798 | 0.990 | 1254 |
| R2L | 52.9 | 0.956 | 0.363 | 0.685 | 0.844 | 1200 |
| U2R | 41.7 | 0.963 | 0.516 | 0.441 | 0.548 | 42 |

CIC ordering is monotonic and tracks accuracy well. SCTS Normal > DoS > Probe > R2L > U2R, accuracy follows same order (with U2R drop to 0.548 reflected in SCTS drop to 41.7).

### Mondrian alpha sensitivity (mean across 18 models)

| α | Mean per-class threshold | Empirical coverage (Mondrian) | Mean SCTS |
|---|---:|---:|---:|
| 0.05 | 0.600 | 0.940 | 67.8 |
| 0.10 | 0.548 | 0.899 | 61.3 |
| 0.20 | 0.423 | 0.837 | 52.8 |

Empirical coverage 0.940 at α=0.05 (target 0.95) — closer than marginal's 0.916.

### Quartile validation (aggregate across 18 models)

| SCTS bin | n | mean accuracy |
|---|---:|---:|
| [0, 25) | 752 | 0.702 |
| [25, 50) | 1823 | 0.615 |
| [50, 75) | 6841 | 0.666 |
| [75, 101) | 8584 | 0.860 |

Closer to monotonic than marginal (0.848, 0.507, 0.680, 0.859). The [0, 25) bin drops from 0.848 → 0.702 — the artifact from cross-dataset scale issues is reduced under Mondrian.

## §18.6 — Cross-architecture disagreement diagnostic (SCTS-v3 dead on arrival)

After Mondrian failed to rescue NSL R2L, considered adding c₄ = per-sample cross-architecture SHAP disagreement (from 06v3's `sample_disagreement_canonical.csv`). Hypothesis: when RF, XGB, DNN disagree about which features matter for a sample, that signals model uncertainty even when each model is individually confident — could catch the R2L misclassification problem.

**Diagnostic before building SCTS-v3:** For NSL R2L samples (n=213 per model), tested whether wrong predictions have higher cross-architecture disagreement than correct predictions.

**Result (NSL R2L, all 6 models):**
| Model | Mean disagreement (correct) | Mean disagreement (wrong) | Direction |
|---|---:|---:|:---:|
| rf_5class_cw | 0.631 (n=7) | 0.607 (n=206) | wrong ↓ |
| xgb_5class_cw | 0.652 (n=13) | 0.605 (n=200) | wrong ↓ |
| dnn_5class_cw | 0.635 (n=14) | 0.606 (n=199) | wrong ↓ |
| rf_5class_smote | 0.614 (n=16) | 0.588 (n=197) | wrong ↓ |
| xgb_5class_smote | 0.607 (n=21) | 0.588 (n=192) | wrong ↓ |
| dnn_5class_smote | 0.615 (n=14) | 0.589 (n=199) | wrong ↓ |

**All 6 models: wrong R2L predictions have LOWER cross-architecture disagreement than correct R2L predictions.** Pearson(disagreement, correct) = +0.075 to +0.197 across models — disagreement weakly predicts correctness, not error.

**Interpretation:** When the model wrongly classifies R2L as Normal, all three architectures (RF, XGB, DNN) agree on the wrong attribution because they're all using similar features to identify "Normal." The disagreement signal points the OPPOSITE direction we'd need to catch R2L misclassifications.

**Adding c₄ would HURT SCTS-v2, not help it.** SCTS-v3 abandoned without building. Diagnostic took 5 minutes and saved ~45 minutes of notebook construction that would have made the problem worse.

## §18.7 — Why NO single-classifier-self-coherence metric can fix R2L

After three rejected investigations (§18.3) and one rejected feature (§18.6), the root cause is now clear:

**SCTS-v2 measures prediction self-coherence, not correctness:**
- c₁ = predicted-class confidence — high when model is confident, regardless of correctness
- c₂ = explanation stability under perturbation — stable explanations of WRONG predictions still get high c₂
- c₃ = predicted-class conformal score — high when predicted-class probability is high, regardless of correctness

**All three components measure aspects of the prediction WITHOUT reference to whether the prediction is correct.** No combination/weighting/threshold-tweaking of these three components can catch the case "the model is confidently and consistently wrong." That requires a signal orthogonal to the classifier's own outputs.

**NSL R2L is the specific failure mode** where the calibrator was fit on 199 R2L samples, sharpens probability so aggressively that p̂(R2L) = 0 for 90.6% of actual R2L test cases. Models then predict Normal/DoS for those R2L samples with high confidence. SCTS-v2 dutifully reports "high trust" because all three components are high. The trust score correctly reports self-coherence; it's just that self-coherence does not equal correctness here.

**What would actually work:** An out-of-distribution detector or an ensemble disagreement metric that uses GROUND-TRUTH-ORTHOGONAL signals (not the classifier's own attributions). Cross-architecture SHAP disagreement is NOT ground-truth-orthogonal — it inherits the same input features and similar learned patterns. Future SCTS-v3 design would need OOD scoring from a separately-trained anomaly detector, or true ensemble disagreement at prediction level (not attribution level).

This is documented as future work in the paper.

## §18.8 — Paper framing decision: Framing B

**Three framings considered:**

- **Framing A (try first, fall back to B if fails):** Build SCTS-v3 with cross-arch disagreement; rescue NSL R2L. **Tested and rejected** per §18.6 — cross-arch disagreement points wrong direction.

- **Framing B (chosen):** Present all 3 datasets honestly. UNSW = success (Pearson +0.47). CIC = moderate success (Pearson +0.32). NSL = cautionary case study (R2L SCTS=77.2, acc=6.7% — explained as inherited from calibration overconfidence).

- **Framing C:** Focus paper on UNSW only. Rejected — loses the multi-dataset contribution that's central to the work.

**Framing B rationale:**
1. The R2L failure case is itself a contribution. "Trust scores inherit calibration quality, and calibrators fail on rare classes in multi-class imbalanced data" is a real finding the field needs.
2. Reviewers respect candor about limitations. Owning the R2L row in the table is much better than hiding it and being caught.
3. The paper claim becomes verifiable: "SCTS-v2 works when calibration is healthy (Pearson +0.47 UNSW); fails to flag confidently-wrong predictions when calibration is overconfident on rare-class boundaries (NSL R2L)."
4. The component framework still works as designed — SCTS-v2 measures internal coherence. We don't pretend otherwise.

**Paper section sketch (Framing B):**
- §X.1 Methodology (SCTS-v2 components, Mondrian conformal per Vovk 2005)
- §X.2 Validation on UNSW (Pearson +0.47, quartile validation, per-class ordering) — the strong evidence
- §X.3 Validation on CIC (Pearson +0.32; tree models limited by ceiling-effect from 98-99% accuracy; DNN best)
- §X.4 NSL case study: SCTS-v2 vs calibration cliffs (the R2L failure, threshold degeneracy explained, what the trust score can and cannot do)
- §X.5 Discussion: trust scores inherit calibration quality; future work on ground-truth-orthogonal trust components

**Key interpretive guidance to write into §X.5:** SCTS-v2 is a relative ranking metric within a fixed (model, dataset) context, NOT an absolute trust score on a 0–100 scale. A SCTS of 75 on UNSW does not mean the same thing as a SCTS of 75 on NSL — the score's calibration to actual correctness depends on the underlying classifier's calibration quality on the relevant class. The case study on NSL R2L (SCTS=77.2, accuracy=6.7%) is concrete evidence that SCTS values across datasets/models cannot be compared on an absolute scale. The trust score's interpretation must be paired with diagnostic information about the underlying calibration quality on the prediction's class. **Reviewers who treat SCTS as an absolute 0–100 trust scale will be misled; this is why we lead with the limitation rather than reporting a single headline number.**

## §18.9 — Marginal results preserved for paper comparison

The marginal SCTS-v2 numbers (from the initial 07c run) are NOT preserved as separate CSVs on disk — they were overwritten by the Mondrian run. The numbers ARE preserved in this v11 §18.2 documentation for paper comparison.

**Reproducibility path for the marginal numbers:** If a future session needs to regenerate marginal CSVs for paper-table comparison (e.g., a side-by-side methodology table), one of:
1. Check out commit `4e80987` (the original marginal-only 07c), run it, save outputs with `_marginal_PRE_MONDRIAN_*` suffix
2. Modify the current 07c temporarily: in cell 4, replace `component_3_mondrian(...)` with `component_3(..., marginal_thresh)` and replace `empirical_coverage_mondrian(...)` with `empirical_coverage(..., marginal_thresh)`. Run, save outputs separately. Revert.

The marginal per-model Pearson values in §18.5's comparison table are from the actual first run (cell 4 output captured in the session transcript). They are not synthesized; if regenerated, they will match deterministically (SEED=42, no randomness in conformal threshold fitting).

If a side-by-side methodology table is needed for the paper (showing why Mondrian was chosen over marginal), the per-dataset Pearson numbers in §18.5 are the authoritative source.

---

# §19 — New Audit Catches (12–15)

## Catch #12 — Test hypothesis before implementing fix (renamed/reinforced from session memory v10's discipline)

**Context:** When marginal SCTS-v2 showed NSL conformal degeneracy, three "fixes" presented themselves: threshold floor, threshold ceiling, c₂ mean-vs-min. Could have implemented any of them in 07c and re-run. Could have built SCTS-v3 with cross-arch disagreement.

**Discipline applied:** Each candidate fix was tested via standalone diagnostic before any code modification to 07c. Each diagnostic took 3-5 minutes. Results:
- c₂ choice: no effect
- Threshold floor: wrong direction (NSL needs ceiling, not floor)
- Threshold ceiling: noise-level improvement (+0.026 Pearson)
- Cross-arch disagreement: WRONG DIRECTION on NSL R2L

**Net savings:** ~2 hours of notebook rebuilding that would have produced no improvement (or made things worse). The diagnostic-first discipline from v10 § 11 catches #7-11 paid for itself 4× this session.

## Catch #13 — Filename caching trap

**Context:** First Mondrian implementation of 07c was named `07c_scts_canonical.ipynb` (same name as marginal version). User opened the file in Colab and was shown the OLD cached marginal version (Drive caches notebook files by filename for snappy reopens, even when underlying file changes).

**Symptom:** User said "i'm not getting new one, opening old one." Assistant initially didn't catch why.

**Fix:** Versioned filename `07c_scts_canonical_mondrian_v2.ipynb`. User downloaded the explicitly-versioned file; Colab couldn't pull cached old version because the filename was new. Successful upload, run, results.

**Discipline:** When iterating on a notebook within a session, **never reuse the same filename for two different versions in the same chat**. Always version-suffix (`_v2`, `_mondrian`, etc.) to dodge browser/Colab caching by filename.

## Catch #14 — Cross-dataset quartile aggregation produces misleading U-shape

**Context:** The marginal SCTS-v2 quartile validation showed accuracy by SCTS bin: [0,25)=0.848, [25,50)=0.507, [50,75)=0.680, [75,101)=0.859. This is U-shaped, NOT monotonic, which superficially suggested SCTS was broken.

**Root cause:** SCTS scales differ across datasets. CIC RF samples have low SCTS (~45) due to terrible stability, but high accuracy (98-99%). They contaminate the low-SCTS bins of the aggregate validation. Pattern is NOT seen in per-model Pearson (the right unit of analysis).

**Discipline:** Cross-dataset/cross-model aggregation of any metric where the scale varies systematically should be flagged as potentially misleading. **Per-model correlation is the correct unit of analysis** for trust-score validity claims, not per-bin accuracy across pooled samples.

Mondrian version's quartile validation is closer to monotonic ([0,25)=0.702, [75,101)=0.860) because Mondrian compresses the variance, but the underlying issue remains.

## Catch #15 — Misleading commit diff for overwritten notebook

**Context:** Commit `bc85188` showed `2 files changed, 2 insertions(+), 579 deletions(-)` and `rewrite notebooks/07c_scts_canonical.ipynb (99%)`. Looked alarming — 579 deletions and only 2 insertions sounds like the Mondrian notebook is essentially empty.

**Reality:** The previous commit (`6258826`) had committed the OLD marginal-version 07c with full Colab run outputs (many lines of stored cell outputs in JSON). The Mondrian-version commit `bc85188` overwrote the notebook with the same code structure but **cleared outputs** (Colab cell outputs ARE cleared when uploading a fresh notebook from a different version). The 579-deletion diff reflects the cleared output JSON, NOT the code structure.

**Verification:** Diagnostic confirmed `mondrian_conformal_thresholds`, `component_3_mondrian`, and `empirical_coverage_mondrian` are all in the committed notebook code. The Mondrian methodology IS on git as of `bc85188`.

**Discipline:** When a commit's diff stats look surprising, verify by inspecting the file contents (or git show) rather than assuming the worst. For notebooks specifically: cell output clears can produce large deletion diffs that don't reflect code loss.

---

# §20 — Day 3 Final State

| Component | Notebook | Commit | Mean Pearson (where applicable) | Status |
|---|---|---|---:|---|
| Canonical SHAP | 04c | `a1f9be9` | — | ✓ |
| Krishna agreement | 06v3 | `d61fbdc` | RC 0.50 (per v9) | ✓ |
| Stability v2 | 05c | `d7df8e7` | mean Jaccard 0.55 | ✓ |
| SCTS-v2 (Mondrian) | 07c | `bc85188` | +0.28 overall, +0.47 UNSW, +0.32 CIC, +0.06 NSL | ✓ |
| Session memory | v9, v10, **v11** | `4a2012c`, `8c8cfbb`, [pending] | — | ✓ |

All Day 3 deliverables sealed. 5 SCTS-v2 result files committed at `bc85188`. 15 audit catches documented.

# §21 — Day 4 Scope (Bootstrap CIs) — Updated from v10

v10 listed 4 bootstrap CI targets. With SCTS-v2 numbers now available, the list is:

1. **Calibration sensitivity claims** (Brier, ECE per-class; already had these from §15 in v8/v9)
2. **Krishna rank correlations** (Spearman, Kendall — from 06v3)
3. **DNN stability advantage** (11/18 dataset-stratified from 05c)
4. **SCTS quartile gaps** — now with UNSW Pearson +0.47, CIC Pearson +0.32, NSL Pearson +0.06 (Mondrian numbers)
5. **NEW: SCTS Normal-vs-U2R per-class gap** (Normal 73.7 vs U2R 58.1 in aggregate; bootstrap CI on the 15.6-point gap to confirm not noise)

Bootstrap protocol: B=1000 stratified resamples, 95% CIs at 2.5/97.5 percentiles. Same protocol as 03d.

Estimated time: 3-4h.

# §22 — Day 5 Scope (3-layer LLM evaluation) — Unchanged from v10

L1: full-set LLM-judge on per-alert SCTS narratives.
L2: stratified 50-alert human rating (Anas + supervisor + 1 peer rater) on Correctness / Actionability / Clarity 1-5 scale.
L3: LLM-judge on same 50 alerts; Cohen's kappa LLM-vs-human.

Peer rater recruitment is a priority blocker. Sampling strategy for the 50 alerts: stratified by SCTS quartile and by dataset (NSL/UNSW/CIC). Include the NSL R2L high-SCTS-low-accuracy cases explicitly to test whether LLM-as-judge catches what SCTS-v2 misses.

# §23 — Paper Outline Update (Framing B)

Sections affected by today's findings:

- §SCTS — Restructured per §18.8 (UNSW success, CIC moderate, NSL case study)
- §Calibration — Add forward reference: "Calibration overconfidence on rare-class boundaries propagates to trust scores (see §SCTS.4)"
- §Methodology — Mention Mondrian per-class conformal with Vovk 2005 / Boström 2020 citations
- §Discussion — Add: "Trust scores inherit calibration quality. Future work: ground-truth-orthogonal trust components."
- §Future Work — Add: "SCTS-v3 with OOD detection or ensemble-prediction disagreement"
- §Limitations — Add the following statements:
  - "SCTS-v2's three components all measure prediction self-coherence; they cannot flag confidently-wrong predictions when the underlying classifier and calibrator are jointly miscalibrated on rare classes (demonstrated on NSL R2L: SCTS=77.2, accuracy=6.7%)."
  - "SCTS-v2 should be interpreted as a relative ranking metric within a fixed (model, dataset) context, not as an absolute trust score on a universal 0–100 scale. Score values across datasets are not directly comparable."
  - "The Mondrian per-class conformal calibration requires n_calib ≥ 30 per predicted class for stable quantile estimation; in highly imbalanced settings (e.g., NSL U2R), the rule falls back to marginal threshold and effectively suspends the conformal component for that class."

---

# Provenance

This document supersedes v10 in scope but does not replace it. v10 covered components 04c, 06v3, 05c and adversarial stability findings. v11 adds 07c SCTS-v2 (§18) and the paper framing decision (§18.8).

Run-throughs and intermediate calculations preserved in chat transcript at `/mnt/transcripts/2026-05-30-01-45-26-2026-05-30-xids-day3-scts-mondrian.txt` (compacted from earlier sessions; this v11 was authored after that compaction).

Repository at the end of this session:
- Notebooks: 04c, 06v3, 05c, 07c all sealed
- Models: gitignored (still untracked, intentional)
- Results: 5 SCTS-v2 files in `results/tables/`
- Docs: v9, v10 committed; v11 pending (this file)
- Untracked sibling: `notebooks/07c_scts_canonical_mondrian_v2.ipynb` (duplicate, can stay or be removed)

Audit catches: 15 total (1-11 from prior sessions; 12-15 documented in §19 above).

End of v11.
