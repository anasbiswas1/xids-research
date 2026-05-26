# X-IDS Project — One-Page Summary

**Author:** Md Anas Biswas | **Date:** May 2026 | **Stage:** End of experimental phase

## What was delivered

An end-to-end explainable intrusion detection framework demonstrating five novel contributions, validated on two benchmark datasets (NSL-KDD and CIC-IDS2017), with reproducible code and paper-ready tables.

## Numerical headlines

| Claim | Evidence |
|---|---|
| Calibration reduces ECE by 47-98% across datasets | NSL-KDD: 67-98%; CIC-IDS2017: 47-91% |
| DNN explanations are most stable | Jaccard 0.86-0.92 vs trees 0.55-0.70 |
| Tree-DNN ranking disagreement is architectural | Trees converge on src_bytes (NSL) and Destination Port (CIC); DNN diverges |
| Rare classes amplify SHAP sign disagreement | Sign agreement drops from 0.96 → 0.78 for R2L/U2R |
| SCTS-v2 stratifies prediction accuracy | Top vs bottom quartile gap: 36-57 percentage points |
| Conformal coverage matches nominal levels | 94.9-97.3% at α=0.05 across all models |
| LLM alert quality tracks trust score | Faithfulness 4.02 → 5.00 from low to high SCTS quartile |

## Deliverable contents

- **15-16 Jupyter notebooks** covering preprocessing → training → calibration → SHAP → stability → SCTS-v2 → LLM alerts → final evaluation → baselines
- **8 paper-ready tables** + **1 summary figure**
- **600+ LLM-generated alerts** with judge scores
- **GitHub repository** with full version control

## What is NOT yet done

| Item | Plan |
|---|---|
| Engelen 2021 corrections on CIC-IDS2017 | Future iteration |
| GPT-4o validation | During paper writing |
| UNSW-NB15 + CICIoT-2023 | Future work |
| Full paper draft | Next phase (2-3 weeks) |

## Status

**Experimental foundation: publishable.** The bottleneck from here is writing.
