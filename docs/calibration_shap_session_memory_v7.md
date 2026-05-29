# Day 3 Calibration + SHAP Session Memory (v7, audit-corrected)

**Project:** X-IDS Research (Explainable Intrusion Detection System) under PAIDS, supervisor Ms Alaa Mohasseb.
**Target venue:** IEEE TIFS / TDSC (top-tier security journals).
**Repo:** github.com/anasbiswas1/xids-research
**Session date:** Started ~evening 30 May 2026, calibration completed by night of 31 May 2026, SHAP completed ~13:15 on 1 June 2026.
**Document purpose:** Comprehensive session memory for Day 3 of the v2 rebuild. Now covers both calibration (§1-11) and SHAP analysis (§12). Future sessions can resume from this single document.

**v7 changes from v6 (audit pass):**
- §12.1 total compute time corrected: **~90 minutes** (68.3 min initial + 21.6 min retry), not "~111 minutes" as v6 incorrectly stated. The "89.7 min initial run" in v6 was an invented figure with no source — actual first run was 68.3 min per the Colab log.
- §12.5 Issue 3 reframed to reflect the actual prediction history (predicted ~13 min for 4 DNNs, actual was 21.6 min)
- §12.7 Observation 2 updated to specify the XGB variant (SMOTE, since the quoted 0.045/0.014 numbers come from `xgb_5class_smote`; CW shows 0.041/0.014)
- §12.4 augmented with explicit cw-vs-smote comparison for cases where they diverge meaningfully
- Class labels verified against `data/processed/{dataset}/class_mappings.json` — all three datasets use `{0:Normal, 1:DoS, 2:Probe, 3:R2L, 4:U2R}` convention

---

## (§1-§11 unchanged from v5 — calibration content remains authoritative)

For all calibration-related context, methodology, results, mistakes, and pending work, see v5 sections §1 through §11. This document only adds §12 covering the SHAP phase, plus minor reference updates.

---

## 12. SHAP analysis (Notebook 04) — completed 1 June 2026

### 12.1 Phase summary

**Status:** ✓ Complete. 18 of 18 5-class models have SHAP values computed and saved.

**Notebook:** `notebooks/04_shap_analysis.ipynb` (committed at `3921f46`).

**Output location:** `shap_values/{dataset}_v2/{model_name}_*.npy` on Drive.

**Total compute time:** **~90 minutes wall clock** across two sessions:
- Initial run (31 May, overnight): 68.3 min total. Successfully completed 14 of 18 models; 4 UNSW/CIC DNNs failed with `KeyError: 'in_dim'` due to checkpoint format mismatch.
- Retry run (1 June, after fix): 21.6 min. Skipped the 14 already-completed models, computed the 4 missing DNNs.

### 12.2 Methodology

**Sampling:**
- Background: 100 stratified samples from calibration set per model
- Evaluation: 1000 stratified samples from test set per model
- Deterministic per-model seed: `SEED + abs(hash(model_name)) % 1000`

**Explainers:**
- **TreeExplainer** for Random Forest and XGBoost (exact SHAP, fast)
- **GradientExplainer** for PyTorch DNN (approximate, more stable than DeepExplainer for PyTorch in current SHAP versions)

**DNN handling note:** The DNN architecture is `[Linear → BatchNorm1d → ReLU → Dropout] × 4 → Linear` outputting logits. For SHAP, a `DNNWithSoftmax` wrapper was applied so SHAP values reflect probability contributions (consistent with Stance C framing where downstream analyses use probabilities).

**SHAP target:** Raw model probabilities. Hybrid calibration is a post-hoc transformation that sits outside the model — SHAP cannot see inside the calibrator. The SHAP values show what features drive the underlying model predictions, which then get calibrated downstream. This is documented in each `_meta.json` file.

**Class label convention (verified from `class_mappings.json`):**

All three datasets use the same 5-class encoding:
| Class index | Label |
|---:|---|
| 0 | Normal |
| 1 | DoS |
| 2 | Probe |
| 3 | R2L |
| 4 | U2R |

UNSW additionally drops the `Generic` category before this mapping is applied (see `data/processed/unsw_nb15_v2/class_mappings.json` for the `unsw_to_five_class` map).

### 12.3 Output structure

Per model (18 total), three files saved:
- `{model}_shap.npy` — SHAP values, shape `(1000, n_features, 5)`
- `{model}_eval_idx.npy` — Test set indices used (shape `(1000,)`), so downstream Krishna/SCTS can subset the test set consistently
- `{model}_meta.json` — Configuration, shape, time taken, explainer used, methodology note

**Per-dataset file sizes (verified on Drive):**

| Dataset | n_features | Total size on Drive |
|---|---:|---:|
| NSL-KDD | 122 | 23.3 MB (6 arrays) |
| UNSW-NB15 | 194 | 37.0 MB (6 arrays) |
| CIC-IDS2017 | 78 | 14.9 MB (6 arrays) |
| **Total** | | **75.2 MB** |

**Git status:** Arrays are NOT in git (excluded by `.gitignore: shap_values/**/*.npy`). This is intentional — large derived artifacts belong on Drive, not in git history. Notebooks + metadata + run logs ARE in git, so reproducibility is preserved via deterministic re-run.

### 12.4 Per-class SHAP magnitudes (sanity check)

All 18 models produced non-zero SHAP values for all 5 classes. Full per-model, per-class mean|SHAP| values:

| Dataset | Model | Normal | DoS | Probe | R2L | U2R |
|---|---|---:|---:|---:|---:|---:|
| NSL | rf_5class_smote | 0.0056 | 0.0037 | 0.0039 | 0.0027 | 0.0026 |
| NSL | rf_5class_cw | 0.0057 | 0.0036 | 0.0037 | 0.0025 | 0.0023 |
| NSL | xgb_5class_smote | 0.0670 | 0.0773 | 0.0815 | 0.0720 | 0.0774 |
| NSL | xgb_5class_cw | 0.0645 | 0.0731 | 0.0767 | 0.0635 | 0.0739 |
| NSL | dnn_5class_smote | 0.0116 | 0.0050 | 0.0051 | 0.0082 | 0.0036 |
| NSL | dnn_5class_cw | 0.0079 | 0.0050 | 0.0042 | 0.0046 | 0.0022 |
| UNSW | rf_5class_smote | 0.0021 | 0.0015 | 0.0017 | 0.0017 | 0.0022 |
| UNSW | rf_5class_cw | 0.0021 | 0.0015 | 0.0016 | 0.0018 | 0.0019 |
| UNSW | xgb_5class_smote | 0.0340 | 0.0178 | 0.0275 | 0.0139 | 0.0453 |
| UNSW | xgb_5class_cw | 0.0337 | 0.0176 | 0.0268 | 0.0139 | 0.0414 |
| UNSW | dnn_5class_smote | 0.0030 | 0.0033 | 0.0040 | 0.0049 | 0.0053 |
| UNSW | dnn_5class_cw | 0.0024 | 0.0026 | 0.0037 | 0.0033 | 0.0045 |
| CIC | rf_5class_smote | 0.0074 | 0.0051 | 0.0052 | 0.0046 | 0.0032 |
| CIC | rf_5class_cw | 0.0075 | 0.0050 | 0.0050 | 0.0044 | 0.0028 |
| CIC | xgb_5class_smote | 0.1144 | 0.1338 | 0.1337 | 0.1386 | 0.0997 |
| CIC | xgb_5class_cw | 0.1075 | 0.1164 | 0.1151 | 0.1152 | 0.0928 |
| CIC | dnn_5class_smote | 0.0141 | 0.0086 | 0.0061 | 0.0072 | 0.0029 |
| CIC | dnn_5class_cw | 0.0065 | 0.0052 | 0.0055 | 0.0065 | 0.0014 |

**Observations:**
- XGBoost SHAP values are an order of magnitude larger than RF/DNN — this is expected (XGBoost TreeExplainer outputs raw log-odds contributions while RF dilutes across many trees and DNN GradientExplainer with softmax wrapper gives bounded probability gradients)
- For Krishna agreement (Notebook 06), the scaling difference is irrelevant — Krishna uses **rank correlations**, not absolute values

**SMOTE vs CW within-architecture comparison:**

For most (dataset, architecture) pairs, the SMOTE and CW variants produce nearly identical mean|SHAP| magnitudes. **Exception:** CIC DNN shows substantial divergence:
- CIC DNN smote Normal-class mean|SHAP| = 0.0141
- CIC DNN cw Normal-class mean|SHAP| = 0.0065
- Ratio: 0.46 (CW magnitudes are roughly half SMOTE)

This is consistent with the calibration finding (v5 §5.3): the CIC dnn_5class_cw model is catastrophically miscalibrated (pre-cal macro Brier 0.053 vs SMOTE's 0.011). The collapsed model produces less-discriminating SHAP attributions, reflected in the smaller magnitudes.

### 12.5 Methodology issues caught during SHAP phase

**Issue 1: DNN checkpoint format inconsistency across datasets**

NSL DNN checkpoints stored architecture metadata (`in_dim`, `n_classes`, `hidden`, `dropout`) alongside state_dict. UNSW and CIC stored bare state_dicts only — no metadata wrapper.

Initial notebook v1 assumed Keras `.h5` format (wrong); v2 fixed to PyTorch `.pt` but only handled NSL's wrapped format; v3 patch (applied during retry) added inference of architecture from the first Linear layer's shape when metadata is missing.

**Resolution:** `load_pytorch_dnn` function inspects the loaded dict — if `state_dict` key present, use wrapped metadata; otherwise treat the dict as a bare state_dict and infer `in_dim` from `net.0.weight.shape[1]`. Architecture (hidden=(256,128,64,32), dropout=0.3, n_classes=5) is identical across all three datasets, so inference is safe.

**Lesson:** Don't assume checkpoint format consistency across datasets even when training code is shared. Inspect actual file structure before writing load logic.

**Issue 2: First run lost 4 of 18 models due to Drive disconnect at end of 68-minute run**

The notebook's per-model save-immediately design meant 14 successful models were preserved on Drive, but 4 DNNs that hadn't yet computed failed with `KeyError: 'in_dim'`. After fixing the load function and re-running, the resumable design correctly skipped the 14 done models and only computed the 4 missing ones in 21.6 minutes.

**Lesson:** Per-model save-immediately + resumable design saved hours of recompute. Worth keeping as a pattern for any overnight notebook.

**Issue 3: DNN SHAP compute time underestimated by ~2x**

Going into the retry session, I predicted the 4 remaining DNNs would take ~13 minutes total (later relaxed to "8-15 minutes"). Actual was **21.6 minutes** — roughly 5.5 min per UNSW DNN (330s, 327s) and 5.3 min per CIC DNN (322s, 317s).

The estimate was based on extrapolating tree-model SHAP times (RF on NSL took ~150s; XGB took ~13s) without accounting for the very different computational profile of GradientExplainer.

**Lesson:** GradientExplainer time is dominated by model forward/backward complexity, not input feature count. UNSW (194 features) and CIC (78 features) took similar per-model time because the DNN architecture is identical — depth and parameter count matter more than input dimensionality for backward-pass computation.

### 12.6 What's now possible (unblocked next steps)

The SHAP outputs feed three downstream components:

1. **Notebook 06 — Krishna agreement** (~1 hour)
   - Compares feature attribution rankings across model architectures
   - Uses Spearman/Kendall rank correlation, not absolute SHAP magnitudes
   - Per-dataset and per-class comparisons

2. **Notebooks 07-08 — SCTS-v2** (~1 hour combined for NSL + UNSW)
   - Trust scoring framework using SHAP attribution coherence
   - Requires per-sample SHAP arrays at the (1000 samples × n_features × n_classes) shape we have

3. **Notebook 05 — Adversarial stability** (~1.5 hours)
   - Does NOT depend on SHAP outputs (operates on raw predictions under perturbation)
   - Can run independently in parallel

Plus **Day 5 LLM evaluation** (3-layer with peer rater) is blocked on Notebooks 06 and 07-08, since trust scores feed alert generation.

### 12.7 Inspection observations (pre-Krishna preview)

Two early observations from inspecting the per-class mean|SHAP| table. Both are **predictions to verify in Krishna agreement (Notebook 06)**, not findings.

**Observation 1 (verify in Krishna):** Within each dataset, RF and XGB likely produce highly correlated feature rankings (predicted >0.8 Spearman). The per-class mean|SHAP| patterns within each tree method are consistent across SMOTE/CW variants and across the two tree architectures, suggesting they latch onto similar features. Krishna agreement is the right place to test this rigorously.

**Observation 2 (verify in Krishna):** UNSW XGBoost (both SMOTE and CW variants) shows substantial per-class magnitude asymmetry across attack types. Quantitatively:
- xgb_5class_smote: U2R = 0.0453, R2L = 0.0139 (ratio 3.3×)
- xgb_5class_cw:    U2R = 0.0414, R2L = 0.0139 (ratio 3.0×)

This asymmetry could indicate XGBoost putting more discriminative "work" into rare classes (U2R has n_calib=253, R2L has n_calib=10,665 on UNSW — the rare class gets more SHAP magnitude despite having fewer samples). If verified, this is worth investigating in the paper's interpretability section. CIC XGBoost shows the opposite pattern: U2R is the LOWEST per-class magnitude (~0.09 vs ~0.13 for other classes), consistent with CIC having only n_calib=7 U2R samples — the model has barely-meaningful U2R signatures to weight.

These predictions are tightly tied to the dataset's class-distribution shape (see v5 §3.2). If Krishna agreement confirms them, it supports the broader thesis that calibration AND attribution behavior both depend on class-distribution structure.

---

## §13. Updated state summary (Day 3 progress)

| Component | Status | Output location | Time taken |
|---|---|---|---|
| Calibration (§2-§7 of v5) | ✓ Complete, 3 audit passes | `calibrators/`, docs in repo | ~10h |
| Bootstrap CIs on calibration | ✓ Complete | `results/tables/calibration_bootstrap_cis.csv` | ~10 min |
| Brier recompute (v5 audit fix) | ✓ Complete | `results/tables/calibration_brier_recomputed.csv` | ~1 min |
| **SHAP analysis (§12 of v7)** | **✓ Complete (18/18)** | **`shap_values/`** | **~90 min** |
| Adversarial stability (Notebook 05) | ⏳ Not started | — | est. ~1.5h |
| Krishna agreement (Notebook 06) | ⏳ Not started, ready | — | est. ~1h |
| SCTS-v2 NSL (Notebook 07) | ⏳ Not started, ready | — | est. ~30 min |
| SCTS-v2 UNSW (Notebook 08) | ⏳ Not started, ready | — | est. ~30 min |
| Bootstrap CIs on other claims | ⏳ Day 4 work | — | est. ~3h |
| 3-layer LLM evaluation | ⏳ Day 5 work | — | est. ~5h + 2h external |

### 13.1 Where to start next session

Three equally reasonable starting points:

**Option A: Notebook 06 (Krishna agreement)** — uses fresh SHAP outputs immediately. Will test the two predictions in §12.7. ~1 hour.

**Option B: Notebook 05 (Stability)** — independent of SHAP, gives breadth to Day 3 progress. ~1.5 hours. Outputs feed Day 4 bootstrap CIs on stability claims.

**Option C: Notebooks 07-08 (SCTS-v2)** — also uses SHAP outputs. Builds the trust scoring layer that downstream LLM evaluation depends on. ~1 hour combined.

**Recommended ordering:** A → C → B. Reason: A and C both use SHAP outputs while they're fresh and the methodology is in active memory. B is independent and can be slotted in any time before Day 4.

---

## §11.1 audit trail (updated for v7)

- **v1-v5:** See calibration audit trail in v5 §11.2 (calibration phase, 3 audit passes)
- **v6:** SHAP analysis phase added (§12). Methodology issues from SHAP phase documented (§12.5).
- **v7 (this version, audit pass on §12):**
  - Caught and fixed wrong total compute time in §12.1 (was "~111 min / 89.7 min initial", actual is ~90 min / 68.3 min initial)
  - Clarified §12.7 Observation 2 to specify XGB variant for quoted magnitudes (SMOTE values shown; CW also reported)
  - Added §12.4 SMOTE-vs-CW divergence subsection with verified CIC DNN ratio (0.46)
  - Verified all class label assertions against `class_mappings.json` for all three datasets
  - Added explicit class-label table (§12.2) for unambiguous reference
  - Rewrote §12.5 Issue 3 to reflect actual prediction history rather than invented "~3-5 min" framing
  - Added bonus note in §12.2 about UNSW's Generic-category drop (visible in `class_mappings.json`)

**Pattern caught in v7 audit:** Inventing precise-sounding numbers ("89.7 min") when only approximate ones were known. The actual number (68.3 min) was in the conversation log — should have looked it up, not estimated. Same anti-pattern as v3-era count error and v4-era UNSW comparison error. Adding to the v5 §7 meta-lessons: **never write a precise number from memory when the source is logged.**

**End of session memory v7. Last updated: 1 June 2026, after SHAP audit pass.**
