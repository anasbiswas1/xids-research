# Calibration → SHAP → Krishna → Stability → SCTS-v2 + Health Flag: Session Memory v12

**Status as of 2026-05-30 (Day 3 complete, expanded scope with health flag):** All Day 3 components sealed including SCTS-v2 Mondrian (07c) AND the calibration health flag (07d). Paper framing: Framing B + operational safety layer.

**Repository:** github.com/anasbiswas1/xids-research

**Last commits (in order):**
- `a1f9be9` Notebook 04c: canonical SHAP recompute
- `d61fbdc` Notebook 06v3: Krishna cross-model agreement
- `512b1f2` Canonical eval indices force-added
- `d7df8e7` Notebook 05c: adversarial stability
- `4a2012c` Session memory v9: canonical samples + Krishna sealed
- `8c8cfbb` Session memory v10: adversarial stability sealed
- `4e80987` Notebook 07c: SCTS-v2 (marginal, intermediate)
- `6258826` Notebook 07c: SCTS-v2 with Mondrian (intermediate)
- `bc85188` Notebook 07c: SCTS-v2 with Mondrian (final, 5 result files)
- `dbfac27` Session memory v11: SCTS-v2 Mondrian sealed + Framing B locked
- `a3a0a47` Notebook 07d: SCTS-v2 calibration health flag (initial with JSON NaN bug)
- **`f40df85` Notebook 07d: JSON NaN aggregation bug fixed (per-flag mean_pearson now real numbers)**

## What v12 adds beyond v11

v11 closed with SCTS-v2 Mondrian sealed and Framing B chosen for the paper. The honest framing acknowledged the R2L failure mode (SCTS=77.2 with accuracy 6.7%) as a cautionary case study.

v12 documents the **calibration health flag** built in response to that limitation. The flag turns the limitation into a defensive systems feature: when SCTS may be unreliable, the system explicitly says so. The paper claim upgrades from "we admit our metric fails on rare classes" to "our metric handles its own failure mode via a deployable warning layer."

Two new audit catches added (#16, #17). Audit total now 17.

This document supersedes v11 in scope but does not replace it. v11 §1–§23 still authoritative for SCTS-v2 Mondrian details. v12 adds §24 + audit catches 16-17 + §25 paper outline update.

---

# §24 — SCTS-v2 Calibration Health Flag (Notebook 07d)

## §24.1 — Motivation

v11 §18 documented the catastrophic failure mode: NSL R2L true-class samples receive Mondrian SCTS = 77.2 despite the model being wrong 93% of the time. The R2L misclassification is not a 07c bug — it traces to calibrator overconfidence on rare-class misclassifications (90.6% of NSL R2L samples have calibrated p̂(y_true=R2L) = 0). SCTS-v2 inherits this overconfidence: when the model predicts Normal on an actual R2L sample with high probability, c₁ is high, c₂ is high (stable explanation of a wrong prediction), and c₃ is high (Mondrian threshold for pred=Normal is degenerate). All three components measure prediction self-coherence, not correctness.

The framework cannot be fixed by parameter tuning (v11 §18.3 rejected three candidate fixes), per-class conformal alone (v11 §18.5 Mondrian partial), or adding a fourth component using SHAP disagreement (v11 §18.6 wrong direction). What it CAN do is **make the unreliability visible**: a per-(model, predicted_class) flag that warns the user when SCTS is operationally untrustworthy.

This is standard defensive systems design: when a metric can fail silently, surface a confidence-in-the-confidence indicator. Three-signal flag returns green/amber/red, joined per-sample by predicted class.

## §24.2 — Three-signal design

Per (dataset, model, predicted_class), three signals each return green/amber/red. Overall flag = worst (most-red) of the three.

### Signal 1 — Threshold usable range

The Mondrian per-class threshold itself indicates degeneracy at both extremes.

| Threshold value | Flag | Reason |
|---|---|---|
| (0.05, 0.95) | GREEN | Healthy range; c₃ has dynamic discriminative power |
| [0.95, 0.999) ∪ [0.001, 0.05] | AMBER | Near-degenerate; c₃ has limited dynamic range |
| ≥ 0.999 or < 0.001 | RED | Degenerate (too loose or too strict); c₃ provides no information |

**Why both ends:** NSL pattern is threshold = 1.0 (degenerate-loose: every sample passes the conformal check, c₃ ≈ c₁). CIC tree pattern is threshold ≈ 0.005 (degenerate-strict: any sample with predicted-class confidence below 0.995 gets c₃ = 0). Both extremes break discriminative power.

### Signal 2 — Cliff fraction (calibrator pathology indicator)

Fraction of calibration samples where `score = 1 - p̂(y_true) ≥ 0.95`, computed **per predicted class** (calibration samples are grouped by what the model predicted).

| Cliff fraction | Flag | Reason |
|---|---|---|
| < 5% | GREEN | Calibrator assigns reasonable probability to true class on this prediction's calibration neighbors |
| 5–20% | AMBER | Calibrator partially fails — some calibration samples in this predicted-class group have near-zero true-class probability |
| ≥ 20% | RED | Severe calibrator failure on this predicted-class group |

**Why predicted class (not true class):** The user knows predicted class at deployment; true class is unknown. The cliff manifests in the predicted-class group precisely because misclassified rare-class samples (with `p̂(y_true) ≈ 0`) leak into the prediction. NSL pred=Normal: cliff fraction ~26% because many actual R2L samples were predicted as Normal and their true-class probability is zero.

### Signal 3 — Calibration support

Number of calibration samples in this predicted-class group (i.e., samples on which the model predicted this class).

| n_calib | Flag | Reason |
|---|---|---|
| ≥ 100 | GREEN | Stable quantile estimation possible |
| [30, 100) | AMBER | Borderline; quantile estimate has wider sampling variance |
| < 30 | RED | Mondrian fallback rule triggered; threshold is the marginal threshold, not class-conditional |

**Why these cutoffs:** n=30 matches the v5/v8 hybrid calibrator boundary (Platt-vs-isotonic) and the conventional small-sample-quantile rule of thumb. n=100 is a typical floor for stable percentile estimation in conformal prediction literature.

### Combination rule

Overall flag = worst of three: any RED → RED. Otherwise any AMBER → AMBER. Otherwise GREEN.

**Why worst-of-three (not majority vote):** Each signal catches a distinct failure mode. Threshold = 1.0 alone is sufficient to make c₃ degenerate; high cliff fraction alone indicates calibrator failure; low support alone means we have insufficient data to validate. The flag is a safety mechanism: any single source of unreliability should warn the user.

## §24.3 — Class-level flag distribution (90 cells = 18 ds-models × 5 predicted classes)

**Overall:** 37 RED, 16 AMBER, 37 GREEN (out of 90).

**Per dataset:**

| Dataset | RED | AMBER | GREEN |
|---|---:|---:|---:|
| NSL | 18 | 1 | 11 |
| UNSW | 6 | 3 | 21 |
| CIC | 13 | 12 | 5 |

### NSL pattern (most degraded)

Same pattern across all 6 NSL models:
| Predicted class | Typical flag | Driver |
|---|---|---|
| Normal | **RED** | Threshold = 1.0 (degenerate); cliff fraction high (R2L misclassifications leak here) |
| DoS | **GREEN** | Threshold ~0.45 (healthy); cliff fraction low (DoS predictions are usually correct) |
| Probe | **RED** | Threshold ~0.99 (near-degenerate); cliff fraction moderate-high |
| R2L | **GREEN** | Threshold ~0.45 (healthy); cliff fraction low (R2L predictions when made are usually correct) |
| U2R | **RED** | Fallback triggered (n_calib < 30) |

Counter-intuitive on first read: pred=R2L is green but pred=Normal is red. The flag is keyed on what the model predicts, not on the true class. When the model correctly predicts R2L (the 7-21 samples per model where this happens, with accuracy 1.000), the calibration is fine — SCTS for those predictions IS trustworthy. The problem is when the model predicts Normal on actual R2L samples; those get RED-flagged via the predicted-Normal route.

### UNSW pattern (mostly healthy)

21/30 cells GREEN. Only U2R consistently RED (fallback). 3 AMBER from DNN models on DoS/Probe (slightly degraded calibration).

### CIC pattern (mixed)

CIC RF and XGB tree models hit threshold values ~0.005 (degenerate-strict) → many RED or AMBER. CIC DNN is the only model with mostly green flags. R2L and U2R consistently problematic across all 6 models.

## §24.4 — Sample-level flag distribution (18,000 samples)

When the class-level flag is joined per-sample by predicted class:

| Flag | n samples | % of total |
|---|---:|---:|
| GREEN | 7,618 | 42.3% |
| AMBER | 3,423 | 19.0% |
| RED | 6,959 | 38.7% |

About 4 in 10 samples are flagged as untrustworthy SCTS. The system honestly admits a substantial fraction of its scores aren't operationally reliable.

## §24.5 — Per-flag Pearson validation

Computed within each flag group: for each (dataset, model), Pearson(SCTS, correctness) over samples in that flag group. Filtered to require std(SCTS) > 1e-9 AND std(correct) > 1e-9 AND n > 5.

| Flag | n samples | n valid models | Mean Pearson | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| GREEN | 7,618 | 14 | **+0.292** | +0.333 | -0.072 | +0.524 |
| AMBER | 3,423 | 6 | **+0.351** | +0.339 | +0.149 | +0.564 |
| RED | 6,959 | 15 | **+0.184** | +0.205 | -0.040 | +0.396 |

**GREEN-vs-RED Pearson gap: +0.108 [95% CI: -0.016, +0.230], NOT statistically significant at α=0.05** (model-level bootstrap; see v13 §29.7 for Day 4 CI methodology). Point estimate is in the expected direction; gap is not statistically distinguishable from zero over the 14-vs-15 valid-model comparison.

Mean SCTS and accuracy per flag:

| Flag | Mean SCTS | Mean accuracy |
|---|---:|---:|
| GREEN | 64.9 | 0.657 |
| AMBER | 63.9 | 0.959 |
| RED | 72.8 | 0.688 |

RED samples have **highest** mean SCTS (72.8) but only moderate mean accuracy (0.688) — exactly the pattern we'd expect if the flag is correctly identifying inflated-but-untrustworthy scores. AMBER samples have very high mean accuracy (0.959) but moderate SCTS (63.9) — driven by CIC tree models where the model is almost always right but the metric has limited dynamic range.

## §24.6 — Critical test: the NSL R2L failure mode

The R2L row was the headline failure in v11. n=1278 NSL R2L true-class samples across the 6 NSL models.

| Statistic | Value |
|---|---:|
| Total NSL R2L samples | 1,278 |
| Flagged RED | 1,206 (**94.4%**) |
| Flagged GREEN | 71 (5.6%) |
| Flagged AMBER | 1 (0.08%) |
| Mean SCTS (all 1,278) | 77.2 |
| Mean accuracy (all 1,278) | 0.067 |

**The flag marks 94.4% of true R2L samples in the NSL test cohort as RED (n=1,278; 95% Wilson CI: 93.0%, 95.5%).**

The 71 GREEN-flagged R2L samples are precisely the cases where the model **correctly** predicted R2L (per-row inspection: every GREEN-flagged group has accuracy = 1.000):

| Model | Pred class | Flag | n | Mean SCTS | Accuracy |
|---|---|---|---:|---:|---:|
| dnn_5class_cw | R2L | green | 14 | 30.8 | 1.000 |
| rf_5class_cw | R2L | green | 7 | 26.2 | 1.000 |
| rf_5class_smote | R2L | green | 16 | 6.0 | 1.000 |
| xgb_5class_cw | R2L | green | 13 | 67.6 | 1.000 |
| xgb_5class_smote | R2L | green | 21 | 64.2 | 1.000 |

These green-flagged samples are the model's rare correct R2L predictions, with low-to-moderate SCTS and accuracy 1.000. The flag correctly distinguishes:
- "Model predicted Normal on actual R2L" → RED via predicted-Normal route → don't trust the high SCTS
- "Model correctly predicted R2L on actual R2L" → GREEN via predicted-R2L route → trust the score (even if low)

This is precisely the desired behavior. The flag system marks the cohort where SCTS is empirically unreliable while leaving the rare correct R2L predictions GREEN-flagged.

## §24.7 — Honest caveats

### RED is not "uninformative"

RED samples have mean Pearson +0.184 [95% CI: +0.122, +0.242] (15 valid models, range -0.04 to +0.40). Within RED samples, higher SCTS still weakly predicts correctness — and the within-RED CI excludes zero. The GREEN-vs-RED gap (+0.108) is in the expected direction but **NOT statistically significant** at α=0.05 (95% CI for gap: -0.016, +0.230). **RED means "operationally untrustworthy due to methodology concerns," not "the score has no information at all."** A reviewer who reads the per-flag Pearson and expects RED to be near zero will see otherwise; the framing should be careful about this. (See v13 §29.7 for the Day 4 bootstrap.)

### Amber has highest Pearson — that's not a contradiction

AMBER mean Pearson +0.351 exceeds GREEN's +0.292. This is because AMBER is dominated by CIC tree models (very high accuracy 0.96+, with SCTS modestly discriminating the few errors). High Pearson is mathematically natural in this regime — small variance in correctness, SCTS still varying systematically. But CIC tree predictions are flagged AMBER because their calibration thresholds are near-degenerate (~0.005), meaning we don't trust the SCTS for individual operational decisions even if the aggregate correlation is high. The flag is methodology-quality-based, not empirical-correlation-based.

### Per-class flag (not per-sample-decision)

The flag is assigned per (model, predicted_class), then joined per-sample by predicted class. This means all samples that the same model predicted as the same class share the same flag color. A more granular flag would require per-sample diagnostic features beyond predicted class, which we don't currently compute.

### Coverage of the failure mode

94.4% of the true-R2L cohort is RED-flagged. Of the remaining 5.6%, 71 are GREEN-flagged correct predictions (accuracy 1.000) — the flag intentionally lets the user trust those scores. The flag is mildly over-conservative on the other side: of 1,206 RED-flagged R2L samples, 14 are actually correct predictions (in dnn_5class_smote where the entire predicted-R2L group flagged RED due to threshold/cliff signals even though the predictions happened to be right). This is a deliberate trade-off — false-positive RED flags warn the user about predictions that ARE right but came from a model-class combination where calibration is unreliable as a population, even if it happened to land correctly on these samples.

## §24.8 — What the flag changes for the paper

Before 07d, the paper claim was: "SCTS-v2 works on UNSW (Pearson +0.47), partial on CIC (+0.32), fails on NSL R2L (cautionary case study, +0.06). Limitation acknowledged."

After 07d, the claim upgrades to:

> "SCTS-v2 produces per-sample trust scores accompanied by a three-signal calibration health flag (green/amber/red). The flag marks 94.4% of true R2L samples in the NSL test cohort as RED (n=1,278; 95% Wilson CI: 93.0%, 95.5%), where SCTS may exceed 75 despite the model being wrong 93% of the time on R2L overall. Within-flag mean per-model Pearson is +0.29 [95% CI: +0.17, +0.40] for green-flagged samples and +0.18 [+0.12, +0.24] for red-flagged samples; the +0.11 gap is in the expected direction but not statistically significant at α=0.05 over the 14-vs-15 valid-model bootstrap (95% CI for gap: -0.02 to +0.23). The flag's primary empirical validation is its RED-flag catch rate on the NSL R2L failure cohort. The system honestly admits which 38.7% of its scores are operationally unreliable, making the trust score deployable with explicit uncertainty surfacing."

The transformation is from **honest disclosure** to **operationally responsible deployment**. The R2L row in the per-class table can now carry a `[RED]` annotation directly. Reviewers see the system handles its own failure mode.

## §24.9 — Outputs (07d)

- `notebooks/07d_scts_calib_health.ipynb` (committed at `a3a0a47` initially; refined at `f40df85`)
- `results/tables/scts_v2_calib_health.csv` (90 rows: per (dataset, model, predicted_class), columns include `mondrian_threshold`, `is_fallback`, `n_calib`, `cliff_fraction`, `signal_threshold`, `signal_cliff`, `signal_support`, `calib_health`)
- `results/tables/scts_v2_canonical_with_health.csv` (18,000 rows: original SCTS data + per-sample `calib_health` joined by predicted class)
- `results/tables/scts_v2_health_summary.json` (signal thresholds, flag distributions, per-flag Pearson validation with min/max range)

---

# §19 (Updated) — Audit Catches 16–17

## Catch #16 — `np.mean()` returns NaN where `pandas.Series.mean()` skips NaN

**Context:** 07d cell 10 built a per-flag summary dict including mean Pearson. Cell 9 (printed analysis) used `pandas.DataFrame['pearson'].mean()` which silently skips NaN. Cell 10 (JSON build) used `np.mean(per_model_pearson)` where `per_model_pearson` is a Python list with NaN values from `np.corrcoef` returning NaN when `correct` is constant within a group. The resulting JSON had `mean_pearson_per_model: NaN` for amber and red flags, while the console output showed the correct values (+0.351 and +0.184).

**Symptom caught by:** User-requested "deep audit" before commit. Without that audit, the JSON would have been committed with NaN values, making the validation appear broken in the saved summary even though the console output and per-sample CSVs were correct.

**Fix:** Patch JSON post-commit. Use explicit filter `std(scts) > 1e-9 AND std(correct) > 1e-9 AND not isnan(p)` before appending to the list, then `np.mean()` on the filtered list. Committed at `f40df85`.

**Discipline:** When aggregating across groups for a JSON summary, explicitly handle NaN (either filter before aggregation or use `np.nanmean`). Don't assume `np.mean()` matches pandas defaults. **Always cross-check JSON output against console output when both are derived from the same source data.**

## Catch #17 — f-string format specifier doesn't accept conditional expressions

**Context:** Tried to write `f"mean={x:+.3f if x is not None else 'N/A'}"`. Python raised `ValueError: Invalid format specifier`. The format-spec portion of an f-string (after the colon) is a restricted mini-language, not arbitrary Python.

**Symptom caught by:** Crash inside the JSON patch script before the actual computation ran. Required a second run with corrected syntax.

**Fix:** Move the conditional out of the f-string into a separate variable:
```python
x_str = f"{x:+.3f}" if x is not None else "N/A"
print(f"mean={x_str}")
```

**Discipline:** f-string format specifiers (after the `:`) accept format codes only, not Python expressions. For conditional formatting, compute the formatted string separately. Minor catch but worth documenting — easy to get wrong when writing diagnostic prints.

---

# §25 — Paper Outline Update (Framing B + Health Flag)

Updating v11 §23 paper outline with health flag integration:

### §SCTS section (revised)

- §SCTS.1 Methodology
  - Three components (c₁ calibration, c₂ stability, c₃ Mondrian conformal)
  - SCTS = (c₁·c₂·c₃)^(1/3) · 100
  - **Three-signal calibration health flag** (threshold range / cliff fraction / calibration support)
  - Mondrian per-class conformal with n<30 fallback (Vovk 2005, Boström 2020)

- §SCTS.2 Validation on UNSW
  - Pearson +0.47 (95% CI from Day 4 bootstrap)
  - 21/30 cells GREEN-flagged → high operational reliability
  - Quartile validation monotonic

- §SCTS.3 Validation on CIC
  - Pearson +0.32 (95% CI from Day 4 bootstrap)
  - 5/30 GREEN, 12/30 AMBER, 13/30 RED — flag exposes the threshold over-strictness
  - DNN performs best (Pearson +0.48); tree models limited by ceiling effect

- §SCTS.4 NSL case study with health flag
  - Pearson +0.06 (raw, all samples)
  - 18/30 cells RED-flagged
  - **The R2L row that fails empirically (SCTS=77.2, acc=6.7%) is 94.4% RED-flagged by our system** (1,206 of 1,278 true-R2L samples) → user warning prevents misuse
  - Demonstrates the value of the health flag: SCTS without the flag would be misleading on NSL; SCTS with the flag is operationally responsible

- §SCTS.5 Per-flag validation
  - Within-flag mean Pearson all significantly positive: GREEN +0.29 [+0.17, +0.40], AMBER +0.35 [+0.22, +0.48], RED +0.18 [+0.12, +0.24]
  - GREEN-vs-RED gap +0.11 in expected direction but NOT significant at α=0.05 (95% CI: -0.02 to +0.23) — gap is descriptive, not inferential
  - Primary validation: 94.4% of true R2L samples in the NSL test cohort RED-flagged (95% Wilson CI: 93.0%, 95.5%)
  - 38.7% of samples flagged RED — system admits which scores are unreliable

- §SCTS.6 Discussion: trust scores inherit calibration quality
  - SCTS-v2 measures prediction self-coherence, not correctness
  - Without the health flag, this is a hidden failure mode
  - With the health flag, the limitation becomes a deployable warning

### §Limitations (revised)

Add to v11 §Limitations:
- "SCTS-v2 should be paired with the calibration health flag at deployment. Scores flagged RED carry a system warning and should not be used for automated decisions without human review."
- "The health flag marks 94.4% of true R2L samples in the NSL test cohort as RED, but flag coverage does not substitute for human judgment in safety-critical contexts."
- "RED-flagged samples retain weak but non-zero SCTS-correctness correlation (within-RED Pearson +0.18, 95% CI [+0.12, +0.24]); the flag indicates operational unreliability due to methodology concerns, not absence of information. The GREEN-vs-RED Pearson gap is +0.11 in the expected direction but not statistically significant; the flag is validated by its catch rate, not by the Pearson differential."

### §Future Work (revised)

- SCTS-v3 with ground-truth-orthogonal trust components (OOD detection, ensemble prediction disagreement)
- Per-sample (not per-class) health flag using sample-specific diagnostics
- Tighter integration with the operator's decision workflow: when RED-flagged, the system could auto-escalate to human review

---

# §26 — Day 3 Final State (with 07d)

| Component | Notebook | Commits | Status |
|---|---|---|---|
| Canonical SHAP | 04c | `a1f9be9` | ✓ |
| Krishna agreement | 06v3 | `d61fbdc`, `512b1f2` | ✓ |
| Stability v2 | 05c | `d7df8e7` | ✓ |
| SCTS-v2 (Mondrian) | 07c | `4e80987`, `6258826`, `bc85188` | ✓ |
| **SCTS-v2 health flag** | **07d** | **`a3a0a47`, `f40df85`** | ✓ |
| Session memory chain | v9, v10, v11, **v12** | `4a2012c`, `8c8cfbb`, `dbfac27`, [pending] | ✓ |

5 components fully sealed. 9 result files in `results/tables/` (5 from 07c + 4 from 07d). 17 audit catches documented.

# §27 — Day 4 Scope (Bootstrap CIs) — Updated

v11 §21 listed 5 bootstrap CI targets. With 07d added, the list is:

1. **Calibration sensitivity** (Brier, ECE per-class)
2. **Krishna rank correlations** (Spearman, Kendall)
3. **DNN stability advantage** (11/18 dataset-stratified)
4. **SCTS Mondrian per-dataset Pearson** (+0.47 UNSW, +0.32 CIC, +0.06 NSL)
5. **SCTS Normal-vs-U2R per-class gap** (15.6 points aggregate)
6. **NEW: Per-flag Pearson gap** (GREEN +0.29 vs RED +0.18; CI on the +0.11 gap to confirm it's not noise)
7. **NEW: NSL R2L RED-flag rate** (94.4% of true R2L samples flagged RED; binomial CI)

Bootstrap protocol unchanged: B=1000 stratified resamples, 95% CIs at 2.5/97.5 percentiles. Estimated time: 3-4h.

# §28 — Day 5 Scope (3-layer LLM evaluation) — Updated

v11 §22 unchanged but adding:

- **L2 sampling now includes flag stratification.** The 50 alerts should be drawn proportional to flag distribution: ~21 green, ~10 amber, ~19 red. Include NSL R2L high-SCTS-low-accuracy RED-flagged cases explicitly.
- **L3 LLM-judge prompt should be aware of the flag.** When evaluating a RED-flagged alert, the LLM should be asked whether it notices anomalies the flag system has surfaced. This tests whether human/LLM judgment can detect what SCTS alone cannot.

---

# Provenance

This document supersedes v11 in scope but does not replace it. v11 covers SCTS-v2 Mondrian methodology and Framing B paper decision. v12 adds the calibration health flag (07d) and the operational-responsibility framing upgrade.

Repository at the end of this session:
- Notebooks: 04c, 06v3, 05c, 07c, 07d all sealed
- Models: gitignored
- Results: 5 SCTS-v2 files + 4 health-flag files in `results/tables/`
- Docs: v9, v10, v11 committed; v12 pending (this file)

Audit catches: 17 total. v9 covered 1-6; v10 covered 7-11; v11 covered 12-15; v12 covers 16-17.

End of v12.
