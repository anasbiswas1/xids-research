# Day 3 Session Memory v9 — Canonical Samples + Cross-Model Agreement

**Project:** X-IDS Research (Explainable Intrusion Detection System) under PAIDS, supervisor Ms Alaa Mohasseb.
**Target venue:** IEEE TIFS / TDSC.
**Repo:** github.com/anasbiswas1/xids-research
**Session date:** 1 June 2026.
**Document purpose:** Comprehensive session memory covering calibration (§1-11, sealed in v5), SHAP (§12, sealed in v7), calibration sensitivity (§14, sealed in v8), canonical sample recompute (§15, new in v9), and cross-model Krishna agreement (§16, new in v9).

**v9 changes from v8:**
- §15 added: Notebook 04c — SHAP recomputed on canonical shared sample indices. Fixes a methodology gap discovered when attempting cross-architecture agreement analysis: Notebook 04 used per-model hash-seeded sampling, causing different architectures to evaluate different test subsets.
- §16 added: Notebook 06v3 — Cross-model SHAP agreement using the six Krishna et al. (2022) metrics on canonical samples. Both v7 §12.7 predictions tested.
- §13 state summary updated.
- Audit trail extended (Issues 7-8 documented).

**v9 audit catches applied before deployment:**
- Issue 7 documented: Notebook 06v2 sample misalignment caught by hard assertion in v3.
- Issue 8 documented: numpy.bool_ silent count bug (printed 0 of 6 when actual was 5 of 6). Caught by row-vs-total inconsistency.
- All numerical claims cross-verified against actual experiment data (preserved in pasted Colab output).

---

## (§1-§11 unchanged from v5 — calibration content remains authoritative)
## (§12 unchanged from v7 — SHAP analysis content remains authoritative)
## (§14 unchanged from v8 — calibration sensitivity content remains authoritative)

For prior content, see v5/v7/v8 documents. v9 only adds §15-§16 and audit trail updates.

---

## 15. Notebook 04c — SHAP recompute on canonical shared samples — completed 1 June 2026

### 15.1 Why this was needed: sample-alignment failure in Notebook 04

When building Notebook 06 (cross-model SHAP agreement using Krishna et al. metrics), the cell-4 data-load sanity check revealed that the three architectures (RF, XGB, DNN) within each (dataset, variant) had been evaluated on **different test samples** in Notebook 04. The cross-architecture overlap was approximately 100-235 samples per pair (out of 1000 each), and the overlap was **structurally dominated by the rarest class** (most overlap was U2R/R2L samples that there are few of in total).

Cross-model agreement metrics (Krishna et al. 2022) compare per-sample feature attributions assuming identical inputs. Comparing "RF's explanation for sample A" to "DNN's explanation for sample B" is methodologically void. Without alignment, all 06v2 numbers would be uninterpretable.

### 15.2 Root cause: per-model hash-seeded sampling in Notebook 04

Inspection of Notebook 04's `compute_shap_for_model` function revealed this line:

```python
rng_local = np.random.default_rng(SEED + abs(hash(model_name)) % 1000)
```

The random seed was **derived from the model name**. Different model names produce different seeds, so each architecture's stratified 1000-sample selection picked different test indices. This was inherited from earlier single-dataset experiments where cross-architecture comparison wasn't planned.

### 15.3 Fix: Notebook 04c with canonical per-dataset sampling

**Notebook:** `notebooks/04c_shap_canonical.ipynb` (committed at `a1f9be9`).

**Methodology change vs Notebook 04:** A single `eval_idx` and `bg_idx` are computed **once per dataset** using `SEED=42` (no per-model jitter), then **all three architectures within that dataset use those same indices**. Everything else (TreeExplainer for RF/XGB, GradientExplainer for DNN, stratified sampling with `stratified_sample`, shape standardization, save format) is identical to Notebook 04.

**Canonical sample composition per dataset (verified at runtime, preserved in commit):**

| Dataset | Normal | DoS | Probe | R2L | U2R | Total |
|---|---:|---:|---:|---:|---:|---:|
| NSL | 261 | 248 | 211 | 213 | 67 | 1000 |
| UNSW | 200 | 200 | 200 | 200 | 200 | 1000 |
| CIC | 354 | 230 | 209 | 200 | 7 | 1000 |

NSL and CIC are unbalanced because U2R has only 67 (NSL) and 7 (CIC) samples available in the test set. The `stratified_sample` function takes `min(200, n_available)` per class, then pads the remainder with random non-stratified samples. UNSW has ≥200 samples per class, so all classes get the target 200.

**The CIC U2R=7 constraint is binding** — cannot be relaxed without enlarging the test set. Per-class analysis on CIC U2R is inherently noise-limited.

### 15.4 Verification

After all 18 models computed, an alignment verification cell loops through all `_eval_idx_shared.npy` files and confirms `np.array_equal(saved, canonical)` for each. **All 18 verified TRUE.** Committed only after verification passed.

### 15.5 Compute timing

Total wall clock: **91.4 minutes** on T4 GPU.

Breakdown:

| Dataset | Total | RF (smote/cw) | XGB (smote/cw) | DNN (smote/cw) |
|---|---:|---:|---:|---:|
| NSL | 15.9 min | 167s / 131s | 14s / 9s | 319s / 314s |
| UNSW | 61.3 min | 1636s / 1362s | 2s / 2s | 333s / 346s |
| CIC | 14.2 min | 122s / 83s | 1s / 2s | 323s / 321s |

Notable patterns:
- **XGB is ~100× faster than RF** at TreeExplainer (probably model size and TreeExplainer XGB-specific optimization)
- **UNSW RF dominates the runtime** (~27 min per RF model). UNSW's 194 features create deeper trees with more SHAP propagation paths.
- **DNN times are consistent** (~5-6 min) across datasets — GradientExplainer scales linearly with samples.

### 15.6 Output files

| File | Purpose | Status |
|---|---|---|
| `notebooks/04c_shap_canonical.ipynb` | Recompute notebook | Committed `a1f9be9` |
| `shap_values/{ds}/canonical_eval_idx.npy` (3 files) | Per-dataset canonical 1000 eval indices | Committed `512b1f2` (force-added past gitignore) |
| `shap_values/{ds}/canonical_bg_idx.npy` (3 files) | Per-dataset canonical 50 background indices | Committed `512b1f2` (force-added) |
| `shap_values/{ds}/{model}_shap_shared.npy` (18 files) | Recomputed SHAP arrays | On Drive, gitignored (large binaries) |
| `shap_values/{ds}/{model}_eval_idx_shared.npy` (18 files) | Per-model copy of canonical eval indices for verification | On Drive, gitignored |
| `shap_values/{ds}/{model}_shap_shared_meta.json` (18 files) | Per-model metadata | Committed `a1f9be9` |

Notebook 04 outputs (the misaligned versions) preserved on Drive but **superseded by `_shared` variants for all v2 downstream analysis**.

### 15.7 Methodological note

The misalignment was not deliberate but reflects an early-pipeline assumption that cross-model comparison wasn't planned. The cell-4 sanity check in 06v2 ("same_indices=False" diagnostic) caught it before any results were committed. Documented as Issue 7 in §16.4.

---

## 16. Notebook 06v3 — Cross-Model SHAP Agreement (Krishna Metrics) — completed 1 June 2026

### 16.1 Methodology

**Notebook:** `notebooks/06_krishna_agreement_v3.ipynb` (committed at `d61fbdc`).

**Six Krishna metrics (Krishna et al. 2022, TMLR):**
- **FA** (Feature Agreement): `|top_k(A) ∩ top_k(B)| / k`
- **RA** (Rank Agreement): fraction of top-k features in same rank position
- **SA** (Sign Agreement): of top-k features in both, fraction with same sign
- **SRA** (Signed Rank Agreement): same feature AND same sign at each rank position
- **RC** (Rank Correlation): Spearman correlation between top-k feature rankings
- **PRA** (Pairwise Rank Agreement): for each pair in top-k(A), fraction where B agrees on ordering

**Scope:**
- 3 datasets × 2 variants (5class_cw, 5class_smote) × 3 architecture pairs (rf-xgb, rf-dnn, xgb-dnn) × 3 K values (5, 10, 15) = 54 aggregate pair-K cells
- Per-class breakdown (5 classes): 270 per-class cells
- Per-sample disagreement scores at K=10 aggregate: 18,000 records (for Day 5 LLM evaluation)

**Sample alignment:** All comparisons use the canonical 1000-sample shared indices from Notebook 04c. **Hard assertion in cell 4** stops the notebook if any model's eval indices don't match canonical.

**Total compute:** ~2 minutes (pure numpy + scipy on CPU).

### 16.2 Aggregate K=10 statistics across 18 model-pair combos

| Metric | Mean | Min | Max | Median |
|---|---:|---:|---:|---:|
| FA (Feature Agreement) | 0.471 | 0.286 | 0.743 | 0.454 |
| RA (Rank Agreement) | 0.081 | 0.032 | 0.162 | 0.067 |
| SA (Sign Agreement) | **0.996** | 0.949 | 1.000 | 1.000 |
| SRA (Signed Rank Agreement) | 0.081 | 0.032 | 0.162 | 0.067 |
| RC (Rank Correlation) | 0.228 | **-0.193** | 0.607 | 0.257 |
| PRA (Pairwise Rank Agreement) | 0.587 | 0.433 | 0.732 | 0.595 |

Three observations are immediately visible from the aggregate stats:

**1. Sign Agreement is extraordinarily high (mean 0.996, min 0.949, median 1.000).** When two architectures both include a feature in their top-10, they agree on whether it pushes prediction toward "attack" or "normal" 99.6% of the time. This is the highest agreement metric by a wide margin.

**2. Rank Agreement is near-zero (mean 0.081).** Exact same feature at the exact same rank position essentially never happens across architectures. The features overlap (FA mean 0.47) and have consistent signs (SA mean 1.0) but their relative ordering shifts substantially.

**3. RC range spans from -0.193 to 0.607.** Some architecture pairs are *anti-correlated* on feature importance ranking — they don't just disagree, they disagree in opposite directions. The mean RC (0.228) is positive but small.

### 16.3 Per-pair K=10 RC table (the headline data)

The clearest single view of cross-architecture agreement:

| Dataset | Variant | rf-xgb RC | rf-dnn RC | xgb-dnn RC |
|---|---|---:|---:|---:|
| NSL | CW | **0.569** | 0.015 | **-0.193** |
| NSL | SMOTE | **0.607** | 0.012 | -0.068 |
| UNSW | CW | **0.536** | 0.189 | 0.274 |
| UNSW | SMOTE | **0.576** | 0.169 | 0.247 |
| CIC | CW | 0.268 | -0.040 | **0.277** |
| CIC | SMOTE | 0.363 | 0.001 | 0.303 |

Bolded values highlight the winning pair per row. **rf-xgb wins in 5 of 6 rows.** The exception is CIC CW where xgb-dnn (0.277) edges out rf-xgb (0.268) — a 0.009 difference that is essentially a tie.

### 16.4 Test of v7 §12.7 Prediction 1: Tree-tree > tree-DNN agreement

Read the table above by row:
- NSL CW: rf-xgb (0.569) > max(rf-dnn 0.015, xgb-dnn -0.193) — **supported**
- NSL SMOTE: rf-xgb (0.607) > max(0.012, -0.068) — **supported**
- UNSW CW: rf-xgb (0.536) > max(0.189, 0.274) — **supported**
- UNSW SMOTE: rf-xgb (0.576) > max(0.169, 0.247) — **supported**
- CIC CW: rf-xgb (0.268) ≯ max(-0.040, **0.277**) — **NOT supported** (xgb-dnn higher by 0.009)
- CIC SMOTE: rf-xgb (0.363) > max(0.001, 0.303) — **supported**

**Prediction 1 supported in 5 of 6 dataset-variant combos.** The single exception (CIC CW) is a near-tie. Interpretation: tree-tree attribution agreement consistently exceeds tree-DNN agreement, with the strongest effect on NSL where tree-DNN RC is essentially zero or negative.

### 16.5 Test of v7 §12.7 Prediction 2: Rare-class agreement < common-class agreement

For each (dataset, variant), mean K=10 RC was computed across all 3 architecture pairs separately for common classes (Normal, DoS) and rare classes (R2L, U2R):

| Dataset | Variant | Common RC (Normal/DoS) | Rare RC (R2L/U2R) | Supports? |
|---|---|---:|---:|---|
| NSL | CW | 0.342 | 0.293 | **Yes** (rare < common) |
| NSL | SMOTE | 0.329 | 0.293 | **Yes** |
| UNSW | CW | 0.316 | 0.255 | **Yes** |
| UNSW | SMOTE | 0.277 | 0.213 | **Yes** |
| CIC | CW | 0.200 | **0.287** | **No** (rare > common) |
| CIC | SMOTE | 0.284 | 0.176 | **Yes** |

**Prediction 2 supported in 5 of 6 dataset-variant combos.** Again, CIC CW is the exception — and in this case rare-class RC (0.287) is *higher* than common-class RC (0.200), the opposite of the predicted direction.

### 16.6 The CIC CW outlier

CIC CW is the systematic exception to both predictions. Hypothesis (not validated): CIC CW's class-weighting on a heavily imbalanced 5-class dataset interacts with the small feature space (78 features) to produce attribution patterns that don't follow the typical architecture-grouping or rare-class-degradation patterns seen on NSL and UNSW. Notably, CIC CW is NOT the row with the weakest aggregate cross-pair RC (NSL CW has the lowest mean at 0.130, with CIC CW at 0.168 and UNSW CW at 0.333). What distinguishes CIC CW is not low overall agreement but rather a flat distribution across pairs — rf-xgb (0.268), rf-dnn (-0.040), and xgb-dnn (0.277) are clustered together rather than the expected sharply-decreasing rf-xgb >> rf-dnn ≈ xgb-dnn pattern seen on NSL and UNSW.

This is worth flagging in the paper as a dataset-specific edge case but does not undermine the broad findings.

### 16.7 Per-sample disagreement scores for Day 5 LLM evaluation

Cell 5 saved per-sample disagreement scores at K=10 aggregate for all 18,000 records (3 datasets × 2 variants × 3 pairs × 1000 samples). The score is `1 - mean(six Krishna metrics)`.

**Top-10 most-disagreed samples per dataset** (averaged across pairs and variants, ranked by disagreement):

| Dataset | Top-10 class distribution | Highest score |
|---|---|---:|
| NSL | Probe×4, DoS×3, U2R×2, R2L×1 | 0.721 |
| UNSW | R2L×8, Normal×1, DoS×1 | 0.738 |
| CIC | Probe×4, DoS×3, Normal×2, R2L×1 | 0.744 |

UNSW's R2L domination of the top-10 is striking. UNSW has 200 R2L samples in the canonical set (one of the larger classes) yet they overwhelmingly populate the disagreement top — suggesting architectures genuinely disagree on what makes UNSW R2L predictions tick.

These 30 samples (10 per dataset) are the prime candidates for the Day 5 LLM evaluation pipeline, where LLM-judge and human-judge will rate alert quality on Correctness/Actionability/Clarity.

### 16.8 Paper framing supported by the data

**Recommended framing:**

> "Cross-architecture SHAP attribution agreement was evaluated using the six metrics from Krishna et al. (2022) across 18 dataset-variant-architecture-pair combinations on canonical shared sample indices. Tree-tree pairs (RF-XGB) showed stronger rank correlation than tree-DNN pairs in 5 of 6 cases (mean RC 0.487 for rf-xgb vs 0.058 for rf-dnn and 0.140 for xgb-dnn), supporting the hypothesis that architecturally-similar models agree more on feature importance. Cross-architecture agreement also dropped on rare classes (R2L, U2R) in 5 of 6 cases. A notable robust finding is the consistency of sign agreement (mean 0.996, near-unanimous): when architectures both attribute a feature, they agree almost without exception on whether it pushes prediction toward 'attack' or 'normal,' even when they disagree about whether the feature is in the top 10 at all. This suggests that calibration-stable interpretations of *direction* are achievable across architectures, while *magnitude-based rankings* are not. The CIC CW configuration is the single exception to both predictions and warrants separate discussion."

### 16.9 Methodological limitations

**Limitation 1: 18 dataset-variant-pair combinations is a small sample for architecture-level claims.** The prediction tests use 6 (dataset, variant) combos as the unit of "supported" — 5 of 6 is statistically suggestive but not conclusive.

**Limitation 2: Architecture pairs share SHAP estimators within their type.** RF and XGB both use TreeExplainer; DNN uses GradientExplainer. Some of the rf-xgb agreement advantage could reflect explainer-method consistency rather than true architecture-level attribution similarity. This confound mirrors the issue acknowledged in v8 §14.9 for the calibration sensitivity experiment.

**Limitation 3: K=10 may not be the right K for all datasets.** CIC has 78 features (top-10 is ~13% of feature space); UNSW has 194 (top-10 is ~5%). The same K means different fractional coverage. K=5 and K=15 were also computed and are in the CSV but the headline analysis used K=10 for consistency with the original Krishna paper.

**Limitation 4: Per-class analysis on rare classes is statistically thin.** CIC U2R has only 7 samples in the canonical set; per-class K=10 metrics on 7 samples are point estimates without confidence intervals. Day 4 bootstrap CIs will quantify this uncertainty.

### 16.10 Files produced

| File | Purpose | Status |
|---|---|---|
| `notebooks/06_krishna_agreement_v3.ipynb` | Krishna analysis notebook | Committed `d61fbdc` |
| `results/tables/krishna_agreement_canonical.csv` | 324 rows (aggregate + per-class) | Committed `d61fbdc` |
| `results/tables/sample_disagreement_canonical.csv` | 18,000 per-sample disagreement records | Committed `d61fbdc` |
| `results/tables/krishna_agreement_canonical_summary.json` | Aggregate stats + prediction test verdicts | Committed `d61fbdc` (corrected counts in subsequent in-memory patch; values match the JSON in repo) |

### 16.11 Audit catches during 06v3 development

**Issue 7: Notebook 06v2 sample misalignment.**

Pre-existing pipeline assumption that Notebook 04 outputs used shared sample indices proved false. The cell-4 sanity check in 06v2 (which the original 06v2 didn't have but I added per cautious-by-default design) printed `same_indices=False` for all 18 (dataset, variant) combos. Caught before any meaningless metrics were computed. Led to building Notebook 04c (§15).

**Lesson:** When pre-existing pipeline outputs are reused, verify the shared assumptions explicitly. Notebook 04's per-model-hash seeding was reasonable for the original single-architecture analyses but invalidated downstream cross-architecture comparison. The cost of a 5-line sanity check is trivial compared to the cost of pretending the data is aligned.

**Issue 8: numpy.bool_ silent count bug in Cell 7.**

The prediction-count line `p2_yes = sum(1 for r in pred2_results if r['supports_pred2'] is True)` returned 0 because numpy comparisons (`rare < common`) produce `numpy.bool_`, and `numpy.bool_(True) is True` evaluates to `False` (different object identity from Python's `True`). The per-row print statements used truthy evaluation, which works correctly with `numpy.bool_`, so the row outputs showed `supports=True` correctly. But the total count was 0.

The bug was caught only because the row prints and the total were inconsistent ("5 of 6 rows show True, total reports 0 of 6"). Had only the total been reported, this would have been a silent wrong answer.

**Resolution:** Patched in a post-hoc cell with `bool()` conversion: `if bool(r['supports_pred2'])`. Counts corrected to 5 of 6 for both predictions before the summary JSON was saved.

**Lesson:** `is True` and `is False` identity checks fail silently on numpy boolean values. Use truthy evaluation (`if x`) or explicit `bool(x)` conversion when working with mixed numpy/Python boolean contexts. **This is the 5th audit catch in the cumulative document chain** (v3 quartile count, v5 mostly-overlap framing, v7 first-run timing, v8 highest-flipping claim, v8 hybrid/Dirichlet mismatch). The previous four were caught at deeper audit passes; this one was caught only because the same data was printed twice in different forms and the inconsistency was visible.

### 16.12 The same-data-printed-twice principle

Issue 8 underlines a useful principle: **when a result will be cited downstream, print it in two different formats during development.** Per-row output and aggregate count should be derived from the same underlying data via independent code paths. If they disagree, the bug is visible. If only one is reported, silent bugs can propagate to publication.

This principle was implicitly used in v8 audit (citing the v5 §3.5 hybrid flipping rates from a different table location than the original calibration findings file). It explicitly saved us here. Worth applying systematically going forward.

### 16.13 Two additional audit catches during v9 deep-audit pass

After the v9 first-draft audit passed simple value-present checks, a user-requested deeper audit (claim-vs-data verification, not just value-presence) caught two more errors before commit.

**Issue 9: §16.6 wrong qualitative framing about CIC CW's aggregate RC.**

Pre-audit v9 §16.6 claimed: "CIC CW also showed the weakest aggregate RC pattern (mean across the row: 0.168 vs UNSW CW 0.333 and NSL CW 0.130)."

The three numerical means (0.130, 0.333, 0.168) are correctly computed. But the qualitative claim "CIC CW also showed the weakest aggregate RC pattern" is wrong: 0.130 (NSL CW) < 0.168 (CIC CW), so NSL CW is the weakest, not CIC CW.

**Resolution:** Reframed §16.6 to acknowledge that CIC CW is the only row failing both predictions but NOT the row with the weakest mean RC. The new framing identifies a different distinguishing feature: CIC CW shows an unusually flat distribution across the three pair RCs (clustered around 0.27, with one outlier at -0.040), rather than the sharply-decreasing rf-xgb >> rf-dnn ≈ xgb-dnn pattern seen on NSL and UNSW.

**Lesson:** Same pattern family as v8 Issue 4 ("highest Dirichlet flipping" claim). I remember the qualitative pattern correctly (CIC CW is the predictions-failure exception) but extrapolate to additional quantitative claims (it's also the weakest in aggregate) without verifying. Cross-row comparisons require explicit computation, not pattern-recall.

**Issue 10: §16.8 wrong arithmetic mean of rf-xgb RC across rows.**

Pre-audit v9 §16.8 paper-framing paragraph claimed: "mean RC 0.532 for rf-xgb vs 0.058 for rf-dnn and 0.140 for xgb-dnn."

Verification: (0.569 + 0.607 + 0.536 + 0.576 + 0.268 + 0.363) / 6 = 0.4865, not 0.532. The other two means (0.058 and 0.140) check out perfectly.

I clearly did not compute the rf-xgb mean and wrote an approximate value from eyeballing the rows. The 0.046 error is large enough that a reviewer who recomputes (which they should) would catch it immediately.

**Resolution:** Corrected to 0.487.

**Lesson:** This is the exact same error mode as Issue 8 (numpy.bool_ silent count) — citing numbers that aren't computed. The fact that the other two means in the same sentence are correct made the mistake harder to spot at first glance, because the pattern "looks right." Going forward, every aggregate number in framing prose must be explicitly cross-computed, not eyeballed.

**Cumulative pattern:** 7 of the 10 audit catches across the v3-v9 chain are arithmetic-or-qualitative-claim errors near correctly-cited individual numbers. The single-value checks pass; the cross-value or claim checks reveal the mistakes. Audit checks need to verify what the document **says about** the numbers, not just whether the numbers appear somewhere.

---

## §13. Updated state summary (Day 3 progress, v9)

| Component | Status | Output | Time |
|---|---|---|---|
| Calibration (§2-§7, v5) | ✓ Sealed | `calibrators/`, docs | ~10h |
| Bootstrap CIs on calibration | ✓ Sealed | `results/tables/calibration_bootstrap_cis.csv` | ~10 min |
| Brier recompute | ✓ Sealed | `results/tables/calibration_brier_recomputed.csv` | ~1 min |
| SHAP analysis (§12, v7) | ✓ Sealed (18/18) | `shap_values/` | ~90 min |
| Refit calibrators (§14.3, v8) | ✓ Sealed (18/18 verified) | `calibrators/{ds}/*_hybrid_fitted.joblib` | ~3 min |
| SHAP raw-vs-calibrated validation (§14.4, v8) | ✓ Sealed (9/9, MODERATE) | `results/tables/shap_raw_vs_calibrated.csv` | ~10 min |
| **SHAP recompute on canonical samples (§15, v9)** | **✓ Sealed (18/18 aligned)** | **`shap_values/{ds}/{model}_shap_shared.npy`** | **~91 min** |
| **Cross-model Krishna agreement (§16, v9)** | **✓ Sealed (5/6 + 5/6 both predictions)** | **`results/tables/krishna_agreement_canonical.csv`** | **~2 min** |
| Adversarial stability (Notebook 05c) | ⏳ In progress | `results/tables/stability_v2.csv` (pending) | est. ~2-3h |
| SCTS-v2 (Notebook 07c) | ⏳ Not started, depends on 05c | — | est. ~30 min |
| Bootstrap CIs on Day 3 claims | ⏳ Day 4 work | — | est. ~3h |
| 3-layer LLM evaluation | ⏳ Day 5 work | — | est. ~5h + 2h external |

### 13.1 Where to start next session

**If 05c completes during this session:** Build Notebook 07c (SCTS-v2 on canonical samples). 05c output `stability_v2_per_sample_jaccard.csv` is the required input for SCTS-v2 component c₂.

**If 05c is not yet complete or has issues:** Resume 05c, then 07c. Both pre-existing single-dataset versions exist (`05_stability_tests.ipynb`, `07_scts_v2.ipynb`) as methodology references.

---

## §11.1 audit trail (updated for v9)

- **v1-v5:** Calibration phase, 3 audit passes
- **v6:** SHAP added; later found to contain wrong total time and unverified class labels
- **v7:** SHAP audit pass corrected v6 errors
- **v8:** Calibration sensitivity added. Three audit catches before deployment: wrong "highest Dirichlet flipping" claim (Issue 4), hybrid/Dirichlet method mismatch (Issue 5), causal language overreach with small n (Issue 6).
- **v9 (this version):** Canonical sample recompute (§15) and cross-model Krishna agreement (§16) added. Two more audit catches: Issue 7 (06v2 sample misalignment caught by cell-4 sanity check before any results were committed) and Issue 8 (numpy.bool_ silent count bug caught by row-vs-total inconsistency in cell 7 output).

**v9 audit performed before commit (TWO passes):**

*Pass 1 (value-present checks):*
- All 16 critical numerical values cross-checked as present in v9
- Method consistency: §16 contains zero hybrid flipping rates (no repeat of v8 Issue 5)
- Both Issue 7 and Issue 8 documented

*Pass 2 (claim-vs-data deep audit, user-requested):*
- **Issue 9 caught:** §16.6 incorrectly claimed CIC CW "showed the weakest aggregate RC pattern." Actual weakest is NSL CW (0.130 < CIC CW's 0.168). Numbers were correct; framing was wrong. Reframed.
- **Issue 10 caught:** §16.8 incorrectly stated mean rf-xgb RC = 0.532. Actual = 0.487. The error was eyeballed-not-computed. Corrected.
- All other rows in §16.3 RC table re-verified against PRED1_DATA row by row.
- Per-class data in §16.5 re-verified against PRED2_DATA.
- Top-disagreement composition strings cross-checked against expected.

**Cumulative audit catches across v3-v9:**

| Pass | Issue type | Example |
|---|---|---|
| v3 audit | Wrong cell count | 7 of 30 → actual 6 of 30 |
| v4 audit | Wrong "mostly" framing | "mostly overlap" → actual 5 of 6 differ |
| v6 audit | Wrong total time | 89.7 min → actual 68.3 min |
| v7 audit | Class label assertions unverified | (corrected before commit) |
| v8 audit Pass 1 | Wrong "highest" claim (Issue 4) | UNSW XGB CW "highest Dirichlet" → actually 5th of 6 |
| v8 audit Pass 2 | Method-data mismatch (Issue 5) | Hybrid SHAP cited against Dirichlet flipping table |
| v8 audit Pass 2 | Causal overreach (Issue 6) | n=3 mechanism claim hedged to "one possible interpretation" |
| v9 audit | Sample misalignment caught at cell-4 (Issue 7) | (prevented bad results from being computed) |
| v9 audit | numpy.bool_ silent count (Issue 8) | 0 of 6 reported, actual 5 of 6 |
| **v9 deep-audit (Issue 9)** | **Wrong "weakest" framing** | **CIC CW "weakest aggregate RC" → actually NSL CW is weakest** |
| **v9 deep-audit (Issue 10)** | **Wrong arithmetic mean** | **rf-xgb mean 0.532 → actually 0.487** |

**Pattern:** 7 of 10 catches are arithmetic-or-qualitative-claim errors **adjacent to correctly-cited individual numbers**. The pre-deployment discipline of running BOTH value-present checks AND claim-vs-data verification is now established as essential.

**End of session memory v9 (covering 04c + 06v3). Stability and SCTS-v2 work in progress; will be added in v10.**
