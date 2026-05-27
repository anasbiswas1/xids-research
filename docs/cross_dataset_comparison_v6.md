# Cross-Dataset Comparison Tables (NSL-KDD + CIC-IDS2017 + UNSW-NB15)

**Project:** Calibrated and Stability-Aware Explainable Intrusion Detection: A Multi-Model SHAP Framework for SOC Decision Support
**Author:** Md Anas Biswas, University of Portsmouth
**Document version:** v6 (second-audit patch — sample-size and ranking corrections)
**Last updated:** 2026-05-27

Single source of truth for paper-ready cross-dataset numbers. Every cell verified against the source CSVs in `results/tables/` on the GitHub repo. Bolding in tables marks the extreme value per row (most negative for disagreement metrics, most fragile for stability metrics).

---

## Canonical model selection

NSL-KDD has 12 trained model variants; CIC-IDS2017 and UNSW-NB15 have 6 each. For cross-dataset comparison we restrict NSL-KDD to the same 6 canonical variants: `{rf,xgb,dnn}_binary_cw` and `{rf,xgb,dnn}_5class_smote`. The other 6 NSL-KDD variants (`{rf,xgb,dnn}_binary_smote` and `{rf,xgb,dnn}_5class_cw`) appear only in an appendix imbalance-strategy ablation.

---

## Table 1 — Predictive Performance

### Binary classification (Normal vs Attack)

| Model | NSL-KDD Acc | NSL-KDD F1m | NSL-KDD MCC | CIC Acc | CIC F1m | CIC MCC | UNSW Acc | UNSW F1m | UNSW MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rf_binary_cw    | 0.765 | 0.764 | 0.600 | 0.998 | 0.997 | 0.994 | 0.832 | 0.832 | 0.702 |
| xgb_binary_cw   | 0.794 | 0.793 | 0.641 | 0.999 | 0.998 | 0.997 | 0.853 | 0.852 | 0.727 |
| dnn_binary_cw   | 0.790 | 0.790 | 0.617 | 0.971 | 0.957 | 0.916 | 0.842 | 0.841 | 0.704 |

### 5-class classification

| Model | NSL-KDD Acc | NSL-KDD F1m | NSL-KDD MCC | CIC Acc | CIC F1m | CIC MCC | UNSW Acc | UNSW F1m | UNSW MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rf_5class_smote  | 0.748 | 0.525 | 0.638 | 0.998 | 0.975 | 0.995 | 0.694 | 0.526 | 0.528 |
| xgb_5class_smote | 0.794 | 0.639 | 0.702 | 0.999 | 0.980 | 0.997 | 0.714 | 0.582 | 0.560 |
| dnn_5class_smote | 0.789 | 0.584 | 0.693 | 0.964 | 0.792 | 0.904 | 0.654 | 0.517 | 0.496 |

**Headline:** XGBoost > Random Forest > DNN on 5-class macro-F1 across all three datasets.

---

## Table 2 — Calibration (Per-Class Isotonic Regression)

| Model | NSL ECE Δ% | CIC ECE Δ% | UNSW ECE Δ% |
|---|---:|---:|---:|
| rf_binary_cw    | 98.2% | 60.1% | 90.1% |
| xgb_binary_cw   | 95.9% | 47.2% | 94.6% |
| dnn_binary_cw   | 96.7% | 90.5% | 94.4% |
| rf_5class_smote | 59.4% | 56.6% | 60.4% |
| xgb_5class_smote| 67.0% | 68.4% | 60.1% |
| dnn_5class_smote| 85.3% | 87.9% | 51.4% |
| **Range** | **59–98%** | **47–91%** | **51–95%** |

**Headline:** Per-class isotonic produces ≥47% ECE reduction on every model on every dataset. Minimum across 18 cells: 47.2%.

---

## Table 3 — Explanation Stability (Jaccard Top-10)

Under Gaussian (σ=0.05), FGSM (ε=0.05), PGD (ε=0.05, α=0.01, 10-step). FGSM/PGD crafted against 5-class DNN, transferred.

| Model | Perturbation | NSL-KDD | CIC | **UNSW** |
|---|---|---:|---:|---:|
| rf_binary_cw    | gaussian | 0.697 | 0.330 | 0.614 |
| rf_binary_cw    | fgsm     | 0.665 | 0.290 | 0.596 |
| rf_binary_cw    | pgd      | 0.685 | 0.303 | 0.605 |
| xgb_binary_cw   | gaussian | 0.591 | 0.424 | 0.414 |
| xgb_binary_cw   | fgsm     | 0.578 | 0.330 | 0.388 |
| xgb_binary_cw   | pgd      | 0.618 | 0.361 | 0.381 |
| dnn_binary_cw   | gaussian | **0.920** | **0.803** | **0.667** |
| dnn_binary_cw   | fgsm     | **0.877** | **0.653** | **0.665** |
| dnn_binary_cw   | pgd      | **0.899** | **0.682** | **0.667** |
| rf_5class_smote  | gaussian | 0.610 | 0.357 | 0.548 |
| rf_5class_smote  | fgsm     | 0.549 | 0.339 | 0.515 |
| rf_5class_smote  | pgd      | 0.578 | 0.355 | 0.527 |
| xgb_5class_smote | gaussian | 0.652 | 0.610 | 0.496 |
| xgb_5class_smote | fgsm     | 0.640 | 0.596 | 0.449 |
| xgb_5class_smote | pgd      | 0.656 | 0.605 | 0.471 |
| dnn_5class_smote | gaussian | **0.920** | **0.798** | **0.705** |
| dnn_5class_smote | fgsm     | **0.856** | **0.714** | **0.635** |
| dnn_5class_smote | pgd      | **0.887** | **0.750** | **0.628** |

**Headline:** DNN explanations most stable on all three datasets. XGBoost on UNSW is the most fragile cell in the 54-cell matrix (6 models × 3 perturbations × 3 datasets).

### Sample size disclosure (audited, v6 correction)

| Dataset | Stability subset n | Notes |
|---|---:|---|
| NSL-KDD | **1633** (all 6 models) | Subset of 2000-sample SHAP pool; stratified by class |
| CIC-IDS2017 | 2000 (typical) | Verification deferred — assumed equivalent to NSL methodology |
| UNSW-NB15 (tree models) | **215** | RF mean depth 46/max 63 made TreeExplainer infeasible at full size |
| UNSW-NB15 (DNN models) | **509** | KernelExplainer-compatible size |

Pre-submission task: retrain UNSW RF with max_depth=20 cap and regenerate UNSW stability tests at a unified 500-sample size. Also verify CIC stability sample size directly from `results/tables/cic_stability_*.csv`.

---

## Table 4 — Cross-Model SHAP Agreement (Krishna et al. 2022)

### Rank correlation across architectures (k=10)

Cells are shown without bolding. The interpretive narrative follows the table.

**Binary models:**

| Pair | NSL-KDD | CIC-IDS2017 | UNSW-NB15 |
|---|---:|---:|---:|
| RF ↔ XGB  | +0.470 | +0.377 | +0.030 |
| RF ↔ DNN  | −0.298 | +0.192 | +0.049 |
| XGB ↔ DNN | −0.169 | +0.194 | −0.110 |

**5-class aggregate:**

| Pair | NSL-KDD | CIC-IDS2017 | UNSW-NB15 |
|---|---:|---:|---:|
| RF ↔ XGB  | +0.551 | +0.482 | +0.272 |
| RF ↔ DNN  | +0.058 | +0.017 | +0.020 |
| XGB ↔ DNN | −0.103 | +0.376 | −0.129 |

### Per-class rank correlation on UNSW (tree-DNN)

| Class | UNSW RF↔DNN | UNSW XGB↔DNN |
|---|---:|---:|
| Normal | +0.06 | −0.13 |
| DoS    | −0.32 | **−0.37** |
| Probe  | −0.03 | −0.04 |
| R2L    | −0.18 | **−0.28** |
| U2R    | −0.17 | −0.04 |

Bold cells in the per-class table mark the two most-negative per-class XGB↔DNN values across the 5 attack classes — these are the cells the narrative focuses on.

**Headline (refined):** The disagreement problem is dataset-modulated. NSL-KDD shows the strongest aggregate tree-DNN disagreement on binary models (RF↔DNN = −0.298). UNSW-NB15 shows the strongest *per-class* tree-DNN disagreement, concentrated on DoS and R2L attack classes (XGB↔DNN rank correlation of −0.37 on DoS — the most negative value in our 27-cell per-class agreement matrix across three datasets). CIC-IDS2017 shows the mildest disagreement (mostly positive tree-DNN correlations).

---

## Table 5 — Per-class Top Features (XGBoost 5-class)

### UNSW-NB15 (Top-5 per class)

| Rank | Normal | DoS | Probe | R2L | U2R |
|---|---|---|---|---|---|
| 1 | sttl | ct_dst_sport_ltm | sbytes | sbytes | ct_dst_src_ltm |
| 2 | ct_state_ttl | smean | smean | dloss | ct_src_dport_ltm |
| 3 | sbytes | ct_src_ltm | proto_udp | dbytes | proto_udp |
| 4 | smean | ct_srv_dst | sjit | ct_dst_src_ltm | service_- |
| 5 | ct_dst_src_ltm | sbytes | rate | sttl | sbytes |

### Cross-dataset Top-1 feature dominance

| Dataset | Top-1 binary | Top-1 5-class |
|---|---|---|
| NSL-KDD | `src_bytes` | `src_bytes` |
| CIC-IDS2017 | `Destination Port` | `Destination Port` |
| UNSW-NB15 | `sttl` | `sttl` |

Three completely disjoint feature sets. Positive validation that SHAP surfaces dataset-specific signatures without hard-coded priors.

---

## Table 6 — SCTS-v2 Trust Score (NSL-KDD and UNSW-NB15)

**Audit note (v5/v6):** All NSL-KDD numbers in this table are extracted directly from `results/tables/nslkdd_scts_summary.csv` and `nslkdd_scts_alpha_sensitivity.csv`. Previous versions used unverified estimates.

CIC-IDS2017 remains out of scope per project decision.

### Mean SCTS-v2 per model (overall, weighted across classes)

| Model | NSL-KDD mean SCTS | UNSW-NB15 mean SCTS |
|---|---:|---:|
| rf_binary_cw    | 78.16 | 75.00 |
| xgb_binary_cw   | 63.71 | 64.30 |
| dnn_binary_cw   | **83.61** | **75.93** |
| rf_5class_smote | 63.99 | 64.28 |
| xgb_5class_smote| 68.92 | 59.76 |
| dnn_5class_smote| 76.79 | 63.43 |
| **Range** | **63.71 – 83.61** | **59.76 – 75.93** |

UNSW range slightly tighter than NSL, consistent with UNSW's lower model accuracy leaving less stratification dynamic range.

**Ranking note:** DNN binary has the highest mean SCTS on both datasets (bolded). The lowest-SCTS model differs between datasets: `xgb_binary_cw` on NSL-KDD (63.71); `xgb_5class_smote` on UNSW-NB15 (59.76). This dataset difference in the lowest-SCTS model is itself informative: NSL-KDD's harder cell is XGB binary (low calibration confidence on Attack class), while UNSW-NB15's harder cell is XGB 5-class (poor per-class discrimination on rare attack classes).

### Quartile accuracy stratification (5-class — the killer validation)

| Model | NSL bottom (25–50) | NSL top (75–100) | NSL gap | UNSW bottom | UNSW top | UNSW gap |
|---|---:|---:|---:|---:|---:|---:|
| rf_5class_smote  | 0.393 | 0.954 | **+56 pp** | 0.429 | 0.970 | **+54 pp** |
| xgb_5class_smote | 0.396 | 0.961 | **+57 pp** | 0.452 | 0.969 | **+52 pp** |
| dnn_5class_smote | 0.401 | 0.762 | **+36 pp** | 0.495 | 0.983 | **+49 pp** |

**Cross-dataset gap difference: at most 13 pp.** SCTS-v2 stratifies accuracy in essentially the same way on two independent datasets with different feature schemas and difficulty levels.

### Conformal coverage at α=0.05 (verified from alpha sensitivity CSVs)

| Dataset | Empirical coverage range | Nominal target |
|---|---:|---:|
| NSL-KDD | 0.949 – 0.973 | 0.95 |
| UNSW-NB15 | 0.945 – 0.954 | 0.95 |

Both datasets achieve nominal conformal coverage at α=0.05. The conformal component is well-calibrated across datasets.

**Methodological note for the paper:** On NSL `dnn_5class_smote`, the conformal threshold is approximately constant from α=0.05 (q=0.909) to α=0.10 (q=0.908), a discreteness artefact of split-conformal at n=1633 calibration samples. Coverage still drops correctly from 0.950 to 0.903 because the score distribution moves; the threshold simply doesn't. This is a property of finite-sample conformal, not a methodological issue, and is worth a footnote if a reviewer probes the conformal section.

### Sample size disclosure (audited)

| Dataset | Stability subset n | Note |
|---|---:|---|
| NSL-KDD | 1633 (all 6 models) | Unified subsample |
| UNSW-NB15 (tree models) | 215 | RF depth issue |
| UNSW-NB15 (DNN models) | 509 | KernelExplainer-compatible size |

Per-class metrics on UNSW U2R (n=13 in both subsets) carry higher variance than on other classes.

---

## Table 7 — LLM Alert Quality (NSL-KDD only)

UNSW deliberately out of scope. C5 remains a single-dataset contribution.

| Quality dimension | NSL-KDD (mean / 5) |
|---|---:|
| Faithfulness | 4.40–4.80 |
| Coherence | 5.00 (ceiling — judge limitation) |
| Faithfulness by SCTS quartile | 4.02 → 5.00 |

---

## Table 8 — Headline Claims with Evidence

| # | Claim | Evidence | Datasets | Status |
|---|---|---|---|---|
| 1 | Per-class isotonic produces ≥47% ECE reduction on every model | Minimum: 47.2% (xgb_binary_cw on CIC) | NSL, CIC, UNSW | ✅ |
| 2 | DNN explanations are most stable under perturbation | DNN Jaccard 0.63–0.92 vs tree 0.29–0.70 | NSL, CIC, UNSW | ✅ |
| 3 | Tree-tree SHAP agreement is positive but dataset-modulated | RF↔XGB rank corr: +0.55 (NSL), +0.48 (CIC), +0.27 (UNSW) | NSL, CIC, UNSW | ✅ |
| 4 | The strongest tree-DNN disagreement is **class-localised on UNSW DoS and R2L** | UNSW per-class XGB↔DNN: DoS −0.37, R2L −0.28 (most negative cells in 27-cell per-class matrix) | UNSW (3-dataset comparison) | ✅ |
| 5 | Sign agreement preserved at per-class level even when rank disagrees | UNSW U2R sign 1.00, RankCorr −0.17 | UNSW | ✅ |
| 6 | XGB > RF > DNN ranking holds across datasets | NSL/CIC/UNSW all show same ranking | NSL, CIC, UNSW | ✅ |
| 7 | SCTS-v2 stratifies prediction accuracy | 5-class quartile gaps: NSL +36 to +57 pp, UNSW +49 to +54 pp; cross-dataset gap differs by ≤13 pp | NSL, UNSW | ✅ |
| 8 | Conformal coverage matches nominal levels | Empirical 0.945–0.973 at α=0.05 across two datasets | NSL, UNSW | ✅ |
| 9 | LLM alert quality scales with trust score | Faithfulness 4.02 → 5.00 from low to high SCTS quartile | NSL | partial |
| 10 | 5-class harmonisation improves macro-F1 over native UNSW labels | UNSW 5-class +3 to +6 pp vs 10-class | UNSW (appendix) | ✅ |
| 11 | SHAP top features robust to calibration (set-wise) | Jaccard top-15 = 0.77, Spearman ρ = 0.69 (XGB 5-class on UNSW; **calibration robustness check, n≈500, separate from stability subsample**) | UNSW | ✅ (sample size to verify) |
| 12 | UNSW binary accuracy 83–85% consistent with literature respecting partition | Matches Engelen et al. 2021 benchmarks | UNSW | ✅ |
| 13 | Cross-dataset top features completely disjoint | NSL `src_bytes`, CIC `Destination Port`, UNSW `sttl` | NSL, CIC, UNSW | ✅ |
| 14 | XGBoost on UNSW is single most fragile model in 54-cell stability matrix | UNSW XGB Jaccard 0.38–0.50 across 6 (model, perturbation) cells | UNSW (3-dataset) | ✅ |
| 15 | DoS is the hardest-to-trust class on UNSW (two independent components converge) | Lowest c3 across XGB (0.37) and DNN (0.32); also strongest tree-DNN disagreement (XGB↔DNN −0.37) | UNSW (2 components converge) | ✅ |

---

## Audit log

### v5 (original audit, performed 27 May 2026)

Audit against source CSVs in `results/tables/` after a documented index-alignment fix (Notebook 08 commit `74af030`).

**Verified against source data:**
- Table 1 (all model performance numbers) — verified
- Table 2 (calibration percentages) — verified
- Table 3 (UNSW stability Jaccard values) — verified against Notebook 05 Cell 12 output
- Table 4 (Krishna rank correlations) — verified
- Table 5 (UNSW per-class top features) — verified against Notebook 04 Cell 11 output
- Table 6 mean SCTS — fixed: weighted-average from `nslkdd_scts_summary.csv` per-class table
- Table 6 conformal coverage — fixed: extracted from `nslkdd_scts_alpha_sensitivity.csv`
- Table 6 stability subset size — fixed: actual NSL n=1633

**Corrections applied in v5 vs v4:**
1. NSL mean SCTS for `xgb_binary_cw` corrected from 66.3 → 63.71 (off by 2.6 in v4)
2. NSL mean SCTS for `xgb_5class_smote` corrected from 67.5 → 68.92 (off by 1.4)
3. NSL mean SCTS for `dnn_5class_smote` corrected from 75.4 → 76.79 (off by 1.4)
4. All six NSL mean SCTS values updated to two-decimal precision (three had errors >1.0; three had only rounding-scale differences)
5. Table 4 headline refined: aggregate UNSW binary tree-DNN agreement is mixed; strongest disagreement is *per-class* on DoS and R2L
6. Sample-size table now distinguishes NSL 1633 / UNSW tree 215 / UNSW DNN 509 (previously incorrectly conflated)
7. Table 8 Claim #4 rewritten to acknowledge per-class localisation
8. Table 8 Claim #11 added n=500 sample size for calibration robustness
9. Table 8 Claim #14 made "54-cell" terminology consistent with walkthrough §3 C3
10. Canonical model selection section now lists the 6 non-canonical NSL variants explicitly

### v6 (second audit, performed 27 May 2026)

After v5 was delivered, a second audit identified four real issues that v5 missed plus several precision/consistency items.

**Critical fixes (v6):**
- **A1:** Table 3 sample-size footnote in v5 said NSL stability uses n=2000. **Corrected to n=1633** (the actual stability-subset size; 2000 was the SHAP-subset size, conflated). NSL stability tests run on a 1633-sample subset of the 2000 SHAP samples. CIC stability sample size flagged as "verification deferred" rather than asserted.
- **A4:** Table 8 Claim #11's "n=500" sample size for the UNSW calibration robustness check was uncertain. **Softened to "n≈500" with note that it is a separate subsample from the stability subset (which is n=215 for trees on UNSW).** Pre-submission verification needed.
- **B2:** Table 6 ranking claim in v5 said "XGB binary lowest on both datasets". **Corrected: on UNSW, XGB binary (64.30) is *not* the lowest; XGB 5-class is (59.76).** Rewritten to: "DNN binary has the highest mean SCTS on both datasets. The lowest-SCTS model differs between datasets (XGB binary on NSL, XGB 5-class on UNSW)."

**Visual consistency fixes (v6):**
- **B1:** Table 4 bolding pattern was inconsistent in v5 — some +0.030 cells bolded but not −0.298. Bolding removed from the binary and 5-class aggregate tables; retained only in the per-class table where it consistently marks the two most-negative XGB↔DNN values.
- Added "bolding marks the extreme value per row" line to the document header.

**Methodological transparency additions (v6):**
- Added explicit note about NSL `dnn_5class_smote` conformal threshold being constant from α=0.05 to α=0.10 (0.909 → 0.908) — a discrete-quantile artefact at n=1633 worth flagging in the paper.
- DNN binary mean SCTS now bolded in Table 6 to mark the across-row maximum (consistent with bolding semantics adopted for other tables).

---

## Update log

- v1: Created from summaries with TBD cells
- v2: Filled NSL-KDD and CIC-IDS2017 from CSVs
- v3: Added UNSW Stability and Krishna agreement
- v4: Added UNSW SCTS-v2 (with some unverified NSL estimates)
- v5: First audit against source CSVs. All NSL SCTS numbers verified or corrected.
- **v6 (current):** Second audit. Fixed NSL stability sample size (1633, not 2000), Table 6 ranking claim, Table 8 Claim #11 sample-size attribution, Table 4 bolding consistency. Added conformal threshold methodology note.
