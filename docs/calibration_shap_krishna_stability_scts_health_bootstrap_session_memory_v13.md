# Calibration → SHAP → Krishna → Stability → SCTS-v2 → Health Flag → Bootstrap CIs: Session Memory v13

**Status as of 2026-05-30 (Day 4 complete):** Bootstrap confidence intervals computed for all key Day 3 findings. 4 claims confirmed strong, 3 refined with CIs, 1 claim failed (GREEN-vs-RED Pearson gap not statistically significant). Multiple v9–v12 claims now require qualification or rewording.

**Repository:** github.com/anasbiswas1/xids-research

**Last commits (recent):**
- `dbfac27` Session memory v11
- `a3a0a47` Notebook 07d initial
- `f40df85` 07d JSON NaN fix
- `7a76deb` Session memory v12
- `d11c9bb` v12 r2 wording precision
- `10c1013` Notebook 08 initial bootstrap CIs
- **`8ce2a66` Notebook 08 corrections (Krishna fix + summary JSON updated)**

## What v13 adds beyond v12

v12 closed Day 3 with the calibration health flag sealed and Framing B locked. v13 documents Day 4 — bootstrap CIs across all 7 targets — and the resulting refinements to earlier-claimed numbers.

The main outcome: **some earlier point estimates remain robust; others tighten the story; one claim outright failed and must be reworded.** This is the audit value of CIs. The paper now rests on quantitatively defensible intervals rather than point estimates.

Five new audit catches added (#18–22). Audit total now 22.

This document supersedes v12 in scope but does not replace it. v12 §1–§28 remain authoritative for SCTS-v2 + health flag methodology. v13 adds §29–§32 covering Day 4 bootstrap results and required claim updates.

---

# §29 — Day 4 Bootstrap Confidence Intervals (Notebook 08)

## §29.1 — Protocol

| Parameter | Value |
|---|---|
| Bootstrap iterations B | 1000 |
| Confidence level | 95% (α = 0.05) |
| Method | Percentile bootstrap (Wilson score for binomial proportion) |
| Random seed | 42 (matches 03d) |
| RNG per iteration | `np.random.default_rng(SEED + b)` for parallel-friendly reproducibility |

**Outputs:** `notebooks/08_bootstrap_cis.ipynb` plus 12 result files in `results/tables/`:
- `bootstrap_cis_calibration.csv`
- `bootstrap_cis_krishna.csv` (per-dataset)
- `bootstrap_cis_krishna_per_pair.csv` (per dataset × pair)
- `bootstrap_cis_krishna_per_cell.csv` (per dataset × variant × pair)
- `bootstrap_cis_stability.csv`
- `bootstrap_cis_stability_advantage.csv`
- `bootstrap_cis_scts_pearson.csv`
- `bootstrap_cis_scts_class_gap.csv`
- `bootstrap_cis_flag_pearson.csv`
- `bootstrap_cis_flag_pearson_summary.csv`
- `bootstrap_cis_r2l_flag_rate.csv`
- `bootstrap_cis_summary.json` (master)

## §29.2 — Target 1: Calibration Brier per-class CIs

Per-class one-vs-rest Brier with bootstrap CI per (dataset, model, class). 108 rows total (18 models × 5 classes + 18 ECE values).

**Per-class Brier means across 6 models per dataset, all CIs tight (~5% relative width):**

| Class | CIC | NSL | UNSW |
|---|---:|---:|---:|
| Normal | 0.009 [0.009, 0.010] | **0.194** [0.189, 0.200] | 0.118 [0.116, 0.120] |
| DoS | 0.005 [0.005, 0.006] | 0.065 [0.062, 0.068] | 0.046 [0.045, 0.047] |
| Probe | 0.003 [0.003, 0.003] | 0.042 [0.040, 0.044] | 0.030 [0.029, 0.031] |
| R2L | 0.001 [0.001, 0.001] | **0.117** [0.113, 0.121] | 0.151 [0.149, 0.152] |
| U2R | 0.000 [0.000, 0.000] | 0.003 [0.003, 0.004] | 0.007 [0.007, 0.008] |

**Pattern:** NSL Normal/R2L and UNSW R2L are the worst-calibrated. CIC is uniformly excellent (10-100× tighter Brier than NSL/UNSW equivalents). This is the per-class footprint of the calibration cliff we documented in v11 §18.3.

**Caveat:** Per-class Brier is one-vs-rest — most samples are NOT the target class, so easy negatives dominate the average. The 0.117 Brier on NSL R2L is the test-pool-weighted view; the conditional-on-true-class failure rate (90.6% of R2L samples have p̂(R2L) = 0) is much more dramatic in absolute terms. See v11 §18.3 for the cliff diagnosis.

## §29.3 — Target 2: Krishna RC with three-layer CIs (corrected version)

**Methodological note:** Initial bootstrap (`10c1013`) used `krishna_agreement_canonical.csv` which is pre-aggregated by class. This conflated aggregate and per-class RC values, producing wrong CIs (audit catch #20). Corrected bootstrap (`8ce2a66`) uses `sample_disagreement_canonical.csv` which has per-sample RC values, allowing proper sample-level bootstrap across 1000 canonical samples per cell.

### Layer 3 — per-dataset RC (the dataset-level claim)

| Dataset | Mean sample RC | 95% CI | Significantly > 0? |
|---|---:|:---:|:---:|
| NSL | 0.157 | [0.147, 0.166] | YES |
| UNSW | 0.332 | [0.324, 0.340] | YES |
| CIC | 0.195 | [0.187, 0.204] | YES |

All three datasets show statistically significant positive RC. UNSW shows the strongest cross-architecture agreement; NSL and CIC weaker.

**These numbers contradict v9's "Krishna RC ~ 0.50" claim** (audit catch #21).

### Layer 2 — per (dataset, pair) RC

| Dataset | Pair | Mean RC | 95% CI | Significant direction |
|---|---|---:|:---:|:---:|
| NSL | rf-xgb | **+0.588** | [+0.578, +0.597] | strongly positive |
| NSL | rf-dnn | +0.013 | [+0.002, +0.024] | barely positive |
| NSL | xgb-dnn | **-0.131** | [-0.143, -0.118] | **significantly NEGATIVE** |
| UNSW | rf-xgb | **+0.556** | [+0.543, +0.568] | strongly positive |
| UNSW | rf-dnn | +0.179 | [+0.166, +0.192] | moderate positive |
| UNSW | xgb-dnn | +0.260 | [+0.249, +0.273] | moderate positive |
| CIC | rf-xgb | +0.315 | [+0.303, +0.328] | moderate positive |
| CIC | rf-dnn | -0.019 | [-0.033, -0.004] | barely negative |
| CIC | xgb-dnn | +0.290 | [+0.279, +0.300] | moderate positive |

**Key findings:**

1. **Tree-tree (RF-XGB) agreement is strong** on all three datasets (RC 0.32 to 0.59, all CIs significantly positive)
2. **Tree-vs-DNN agreement is weak-to-negative and dataset-dependent**
3. **NSL XGB-DNN shows significantly NEGATIVE RC** (-0.131, CI [-0.143, -0.118]) — a publishable finding about cross-architecture explanation disagreement that v9-v12 did not surface
4. v9's "RC ~ 0.50" applies specifically to RF-XGB pair on NSL/UNSW; the overall dataset-level RC is 0.16-0.33

## §29.4 — Target 3: DNN stability advantage (paired bootstrap CIs)

Paired bootstrap on per-sample Jaccard advantages (DNN minus tree). 12 comparisons total (3 datasets × 2 variants × 2 trees), each paired over 3000 samples (1000 canonical × 3 perturbations).

| Dataset | Variant | DNN vs RF | DNN vs XGB |
|---|---|---:|---:|
| NSL | cw | -0.100 [-0.107, -0.093] ❌ DNN worse | -0.092 [-0.099, -0.086] ❌ DNN worse |
| NSL | smote | +0.041 [+0.034, +0.048] ✓ | -0.020 [-0.026, -0.013] ❌ DNN worse |
| UNSW | cw | +0.090 [+0.082, +0.097] ✓ | +0.088 [+0.081, +0.094] ✓ |
| UNSW | smote | +0.041 [+0.034, +0.048] ✓ | +0.083 [+0.076, +0.089] ✓ |
| CIC | cw | +0.323 [+0.315, +0.332] ✓ | +0.073 [+0.066, +0.081] ✓ |
| CIC | smote | +0.129 [+0.121, +0.137] ✓ | +0.008 [+0.001, +0.015] ✓ |

**Summary:** 9/12 significant DNN advantage. **3/12 significant DNN disadvantage on NSL.** Advantage is dataset-dependent, not universal.

**Refinement vs v10/v11:** v10/v11 reported "DNN advantage 11/18 dataset-stratified" as a point count. The CI-aware version is 9/12 significant, with NSL CW showing significantly worse DNN performance. The "dataset-stratified" framing was correct; the precision was missing.

## §29.5 — Target 4: SCTS Mondrian Pearson per dataset (CRITICAL CI)

Per (dataset, model) bootstrap on Pearson(SCTS, correctness). Then bootstrap the mean across 6 models per dataset.

| Dataset | Mean Pearson | 95% CI | Significantly > 0? |
|---|---:|:---:|:---:|
| **NSL** | **+0.055** | [-0.031, +0.145] | **NO** |
| UNSW | +0.465 | [+0.438, +0.491] | YES |
| CIC | +0.318 | [+0.233, +0.408] | YES |

**The headline correction:** NSL Pearson is NOT statistically distinguishable from zero. v11 §18 / v12 §27 reported it as "+0.06 (weak)" — this CI version strengthens the case study framing: SCTS-Mondrian on NSL has no statistically detectable correlation with correctness.

UNSW and CIC remain confirmed at the v11 levels. CIs are wider on CIC (95% half-width ±0.09) than UNSW (±0.02) because CIC has more model-to-model heterogeneity (DNN +0.48 vs trees +0.20).

## §29.6 — Target 5: Normal-vs-U2R SCTS gap (CIs with aggregation clarification)

Stratified bootstrap by true class.

**Sample-pooled aggregate:** 12.03 [11.19, 13.03] — pools all 4890 Normal + 1644 U2R samples across 18 models.

**Per dataset:**
| Dataset | Gap | 95% CI |
|---|---:|:---:|
| NSL | 6.30 | [4.84, 7.92] |
| UNSW | 10.54 | [9.17, 11.92] |
| CIC | 29.72 | [17.39, 41.76] |

All three datasets show significantly positive gap.

**Clarification of v11/v12's "15.6":** v11 §18.5 reported aggregate Normal-vs-U2R gap = 15.6 points. The bootstrap aggregate is 12.03. The discrepancy is real:

- **Sample-pooled** (12.03): weights by sample count. UNSW dominates because it has 1200 U2R samples vs CIC's 42.
- **Dataset-averaged** (~15.5 = (6.30 + 10.54 + 29.72) / 3): weights each dataset equally.

Both are valid statistics; they answer different questions. The paper should pick one and state it explicitly. For the per-dataset story (which is the actual finding), the per-dataset CIs in the table above are the authoritative reporting (audit catch #22).

## §29.7 — Target 6: GREEN vs RED Pearson gap — FAILED CLAIM

Bootstrap on per-model Pearson within each flag group (14 GREEN-valid models, 15 RED-valid models). Then independent bootstrap of the gap of means.

```
GREEN: mean Pearson = +0.292 [95% CI: +0.172, +0.403], n_models=14
AMBER: mean Pearson = +0.351 [95% CI: +0.222, +0.484], n_models=6
RED:   mean Pearson = +0.184 [95% CI: +0.122, +0.242], n_models=15

GREEN minus RED Pearson gap: +0.108 [95% CI: -0.016, +0.230]
Significantly greater than zero: FALSE
```

**The gap CI crosses zero.** v12 §24.5 / §24.8 claim of "demonstrating that the flag separates operationally trustworthy from operationally untrustworthy scores" is empirically unsupported at α=0.05.

**Still true** (each flag's internal Pearson CI excludes zero):
- GREEN samples DO have significantly positive within-flag Pearson (+0.29, CI [+0.17, +0.40])
- AMBER samples have significantly positive within-flag Pearson (+0.35)
- RED samples have significantly positive within-flag Pearson (+0.18)

**No longer claimable as significant:**
- The differentiation between GREEN and RED groups at α=0.05

**Required v12 rewording:** the paper claim must pivot from "GREEN vs RED demonstrates separation" to descriptive only: "Within-flag mean Pearson is +0.29 for GREEN samples and +0.18 for RED samples; the +0.11 gap is in the expected direction but not statistically significant over the 14-vs-15 model bootstrap (95% CI: -0.02 to +0.23). The flag's primary empirical validation is the catch-rate finding in §29.8."

## §29.8 — Target 7: NSL R2L RED-flag rate (Wilson binomial CI)

```
1206 / 1278 NSL R2L true-class samples flagged RED = 94.4% [95% Wilson CI: 93.0%, 95.5%]
```

**Tight CI; lower bound 93.0% is well above any natural threshold of interest.** This is the strongest, most defensible empirical claim for the calibration health flag. It survives Day 4 CI analysis with the cleanest result.

The flag's value proposition is now anchored on this finding: when SCTS is unreliable due to calibration cliffs on rare classes, the flag identifies the affected cohort. The Pearson-gap claim (Target 6) was always secondary; with Target 6 failing, Target 7 carries the operational claim alone.

---

# §30 — Paper Claims Refinement (Required Updates)

The following v9-v12 claims need rewording to align with Day 4 CIs:

## v9 — Krishna RC

**Original (v9 §15):** "Mean Krishna rank correlation across pairs is ~0.50 on canonical samples"

**Required correction:** "Tree-pair (RF-XGB) rank correlation is strongly positive (RC = 0.32-0.59 across datasets, all 95% CIs above zero). Tree-vs-DNN agreement is weak (UNSW: RC 0.18-0.26) to negative (NSL XGB-DNN: RC = -0.131, CI [-0.143, -0.118]). Overall dataset-level RC is 0.16-0.33."

This is a stronger finding: instead of a single weakly-positive claim, we have a clean two-part story (trees agree with each other; DNN disagrees with trees, sometimes systematically).

## v10/v11 — DNN stability advantage

**Original:** "DNN advantage 11/18 dataset-stratified"

**Required correction:** "DNN shows significantly higher mean Jaccard stability than tree models on 9 of 12 paired comparisons (UNSW 4/4, CIC 4/4, NSL 1/4). On NSL CW, DNN is significantly *less* stable than both tree architectures (mean disadvantage ~0.09). The architecture advantage is dataset-dependent."

## v11 — NSL SCTS Pearson

**Original (v11 §18.5):** "NSL mean: +0.11 (range -0.06 to +0.26) — weak"

**Required correction:** "NSL SCTS-Mondrian mean Pearson +0.055 [95% CI: -0.031, +0.145] — not statistically distinguishable from zero. SCTS provides no reliable signal of correctness on NSL test samples."

This strengthens the case-study framing for NSL.

## v12 — GREEN vs RED Pearson gap

**Original (v12 §24.5/§24.8):** "Sample-level Pearson(SCTS, correctness) within green-flagged samples is +0.29 versus +0.18 within red-flagged samples, demonstrating that the flag separates operationally trustworthy from operationally untrustworthy scores."

**Required correction:** "Mean per-model Pearson is +0.29 within green-flagged samples and +0.18 within red-flagged samples; the gap (+0.11) is in the expected direction but not statistically significant at α=0.05 over the 14-vs-15 model bootstrap (95% CI: -0.02 to +0.23). The flag's primary empirical validation is its 94.4% RED-flag rate on true R2L samples in the NSL test cohort (95% Wilson CI: 93.0%, 95.5%)."

## v11/v12 — Normal-vs-U2R gap

**Original:** "Normal-vs-U2R SCTS gap = 15.6 points (aggregate)"

**Required clarification:** The 15.6 is the dataset-averaged version (model-weighted). The sample-pooled version is 12.03. Per-dataset gaps are NSL 6.30 [4.84, 7.92], UNSW 10.54 [9.17, 11.92], CIC 29.72 [17.39, 41.76]. All significantly positive. Paper should report per-dataset gaps individually rather than a single aggregate.

---

# §31 — Updated Paper Outline (CI-aware)

Replaces v12 §25.

### §SCTS section (revised again)

- **§SCTS.1 Methodology** — SCTS-v2 components, Mondrian conformal, health flag, bootstrap CI protocol (B=1000, percentile, Wilson for binomial)

- **§SCTS.2 Validation on UNSW** — SCTS Mondrian Pearson **+0.465 [+0.438, +0.491]** (95% CI). Per-class ordering tracks accuracy. 21/30 health-flag cells GREEN.

- **§SCTS.3 Validation on CIC** — SCTS Mondrian Pearson **+0.318 [+0.233, +0.408]**. DNN performs best (Pearson +0.48); tree models limited by 98-99% accuracy ceiling.

- **§SCTS.4 NSL case study** — SCTS Mondrian Pearson **+0.055 [-0.031, +0.145]**, not statistically distinguishable from zero. The R2L row (SCTS=77.2, accuracy=6.7%) is 1206/1278 = 94.4% [93.0%, 95.5%] RED-flagged. Demonstrates the value of the health flag: SCTS without the flag would be misleading; SCTS with the flag is operationally responsible.

- **§SCTS.5 Per-flag analysis** — Each flag's within-group Pearson is significantly positive: GREEN +0.29 [+0.17, +0.40], AMBER +0.35 [+0.22, +0.48], RED +0.18 [+0.12, +0.24]. The +0.11 GREEN-vs-RED gap is in the expected direction but not statistically significant. The primary flag validation is the catch-rate, not the gap.

- **§SCTS.6 Discussion** — Trust scores inherit calibration quality. The health flag transforms a hidden failure mode into a deployable warning. CIs across all targets establish which claims are reviewer-defensible.

### §Krishna section (revised)

- Tree-tree agreement (RF-XGB) is strong: 0.32-0.59 across datasets
- Tree-vs-DNN agreement is weak; NSL XGB-DNN actively disagrees (RC = -0.131, CI [-0.14, -0.12])
- This dual finding (trees consistent, DNN distinct) is itself a contribution

### §Stability section (revised)

- DNN advantage is dataset-dependent: significant on UNSW (4/4 vs trees), CIC (4/4 vs trees), NSL (1/4 — only SMOTE vs RF)
- NSL CW: DNN significantly *less* stable than both trees
- Frames the architecture choice as task-dependent, not universal

### §Calibration section (existing, expanded)

- Per-class Brier with CIs (Target 1 results)
- NSL Normal Brier 0.194, NSL R2L Brier 0.117 — substantially higher than UNSW/CIC equivalents
- Calibration quality directly feeds downstream trust score behavior (forward reference to §SCTS.4)

---

# §19 (Continued) — Audit Catches 18–22

## Catch #18 — Internal inconsistency in v12: UNSW 21/30 vs 22/30

**Context:** v12 §24.3 per-dataset table correctly stated UNSW = 21 GREEN + 3 AMBER + 6 RED = 30. But v12 §24.3 pattern description and §25 paper outline §SCTS.2 both said "22/30 cells GREEN-flagged."

**How caught:** User-flagged before commit during pre-Day-4 audit ("In §24.3, the dataset table says UNSW has 21/30 GREEN. But the paper outline later says 22/30. That should be corrected to 21/30").

**Fix:** Updated both references to 21/30 in v12 r2 (commit `d11c9bb`).

**Lesson:** When writing summary documents that reference the same number multiple times, single-source-of-truth tables should be the only place a count is stated; later references should say "(see Table X)" instead of restating numerically.

## Catch #19 — Imprecise denominator wording for 94.4%

**Context:** v12 phrased the flag-catch claim as "94.4% of catastrophic failure cases" in multiple places. The denominator is actually the entire NSL R2L true-class cohort (n=1278, including ~85 correctly-predicted samples), not a subset filtered to misclassifications. "Catastrophic failure cases" implied a filtered denominator.

**How caught:** User-flagged precision suggestion ("94.4% is calculated over the true-class R2L cohort, not necessarily over only the individually wrong predictions").

**Fix:** Reworded 7 instances in v12 r2 to "94.4% of true R2L samples in the NSL test cohort." Added over-conservative caveat documenting that 14 of 1206 RED-flagged R2L samples were actually correctly predicted (in dnn_5class_smote where the whole predicted-R2L group flagged RED due to threshold/cliff signals).

**Lesson:** When reporting a percentage, always state the denominator explicitly in the same sentence so readers can verify what's being measured.

## Catch #20 — Krishna CSV structure misread (Day 4)

**Context:** Notebook 08 cell 6 initially bootstrapped `krishna_agreement_canonical.csv` grouping by (dataset, pair, k). The CSV has 6 class-stratified rows per (dataset, variant, pair, k) cell (1 'aggregate' + 5 per-class). The bootstrap took mean across these 6 mixed rows, producing wrong CIs that conflated aggregate and per-class RC.

**Symptom:** Cell 6 output showed mean RC = 0.27 (NSL), 0.28 (UNSW), 0.19 (CIC) — contradicting v9's reported "RC ~ 0.50."

**Discovery:** Investigation revealed the right unit of analysis is per-sample RC values from `sample_disagreement_canonical.csv` (18,000 rows with `sample_RC` column), not the pre-aggregated table.

**Fix:** Patched bootstrap to use `sample_disagreement_canonical.csv` with three-layer analysis (per-cell, per-pair, per-dataset). Old `bootstrap_cis_krishna.csv` was overwritten with corrected Layer 3 data; per-pair and per-cell saved as new files. Commit `8ce2a66`.

**Lesson:** Before bootstrapping a CSV, inspect its structure: what does each row represent? Pre-aggregated tables and per-sample tables look similar but require different bootstrap units.

## Catch #21 — v9's "Krishna RC ~ 0.50" was per-pair, not overall

**Context:** v9 §15 reported Krishna RC ≈ 0.50 as the headline cross-architecture agreement number. Day 4 bootstrap revealed this applies specifically to the RF-XGB tree pair, not to all pairs averaged.

**Actual numbers:** Overall dataset-level mean RC is 0.157-0.332 (CI tight). RF-XGB specifically is 0.32-0.59. RF-DNN is -0.04 to +0.19. XGB-DNN is -0.19 to +0.30 with NSL XGB-DNN significantly negative.

**Implication:** Paper Krishna claim must be split into two: tree-pair agreement (strong) and tree-vs-DNN agreement (weak to negative). v9's single "0.50" framing was overgeneralized.

**Lesson:** Aggregate metrics across structured comparisons (here, 3 pairs × 2 variants × 3 K values per dataset) can hide important heterogeneity. Always report the structured breakdown alongside the aggregate.

## Catch #22 — Bootstrap unit choice matters

**Context:** Several Day 4 targets could be bootstrapped at different units:
- Per-sample within each (dataset, model) — measures sample variation
- Per-model within each dataset — measures model-to-model variation
- Per-dataset — measures dataset-level uncertainty

Each gives different CIs. Per-sample bootstrap typically gives the tightest CI but only addresses sampling variation, not model heterogeneity. Per-model bootstrap is wider but addresses "would this finding hold for a new model architecture?"

**Example:** GREEN-vs-RED gap (Target 6) was bootstrapped per-model (14 vs 15 valid models), giving CI [-0.016, +0.230]. If bootstrapped per-sample (7,618 vs 6,959 samples), the CI would be much tighter and likely exclude zero — but that wouldn't address the question "does the flag separate green from red across models?"

**Discipline:** For each CI in the paper, state explicitly what variation it addresses. Per-sample CIs are about sampling. Per-model CIs are about generalization across model architectures within the test set. Use the appropriate one for the claim being made.

**Lesson:** Bootstrap unit IS a methodology choice that determines what the CI means. Document the choice for each claim.

---

# §32 — Day 5 Preparation

Per v12 §28, Day 5 is the 3-layer LLM evaluation:
- L1: full-set LLM-judge on per-alert SCTS narratives (~3h compute)
- L2: stratified 50-alert human rating (Anas + supervisor + 1 peer rater) on Correctness / Actionability / Clarity 1–5 scale (~2h external + 1h aggregation)
- L3: LLM-judge on same 50 alerts, Cohen's kappa LLM-vs-human (~1h)

**Updated L2 sampling for Day 5 (incorporating Day 4 results):**

With the GREEN-vs-RED gap failing CI, the L2 sampling should explicitly stratify to test whether human judgment can detect what SCTS+flag cannot:

| Stratum | n | Rationale |
|---|---:|---|
| GREEN-flagged high SCTS (UNSW/CIC) | 12 | Verify flag's positive validation on the working cases |
| GREEN-flagged NSL R2L correctly predicted | 5 | Test if humans agree these few cases ARE trustworthy |
| RED-flagged NSL R2L misclassified-as-Normal | 15 | The catastrophic failure case — flag says don't trust |
| RED-flagged CIC tree models | 8 | Conformal-too-strict regime |
| AMBER-flagged samples | 10 | The 0.96 mean accuracy / +0.35 Pearson group |

Total = 50 alerts.

**Peer rater recruitment is still a priority blocker.** Document Anas's role + supervisor's role + need for one external rater for inter-rater reliability via Cohen's kappa.

---

# §33 — Day 4 Final State

| Component | Notebook | Commits | Status |
|---|---|---|---|
| Bootstrap CIs (all 7 targets) | 08 | `10c1013`, `8ce2a66` | ✓ |
| **v13 session memory** | — | [pending] | — |

Day 4 deliverables: 1 notebook + 12 result files. CIs available for every key paper claim. 5 new audit catches (#18–22).

| Total Day 1-4 commits | 28+ |
| Total notebooks sealed | 8 (04c, 06v3, 05c, 07c, 07d, 08, plus earlier 03/04 etc.) |
| Total session memory docs | v9-v13 (5 documents) |
| Total audit catches | 22 |
| Paper framing | Framing B + health flag + CIs |

---

# Provenance

v13 supersedes v12 in scope but does not replace it. v12 covers SCTS-v2 + health flag through `d11c9bb` (r2 wording fix). v13 adds Day 4 bootstrap CIs and the resulting refinements to v9–v12 claims.

Repository at the end of Day 4:
- Notebooks: 04c, 06v3, 05c, 07c, 07d, 08 all sealed
- Results: 23 files in `results/tables/` (9 pre-Day-3 + 14 from Days 3-4)
- Docs: v9, v10, v11, v12 (r2) committed; v13 pending (this file)

Audit catches: 22 total. v9 covered 1-6; v10 covered 7-11; v11 covered 12-15; v12 covered 16-17; v13 covers 18-22.

End of v13.
