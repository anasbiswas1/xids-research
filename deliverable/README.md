# Calibrated and Stability-Aware Explainable Intrusion Detection

A Multi-Model SHAP Framework for SOC Decision Support

**Author:** Md Anas Biswas 
**Affiliation:** MSc Researcher, University of Portsmouth 
**Date:** May 2026

---

## Project overview

This research develops an explainable intrusion detection framework that combines model calibration, explanation stability, conformal prediction, and natural-language alerts into a unified trust score (SCTS-v2) for SOC analysts. The framework is evaluated on two benchmark datasets across three model architectures.

### Five contributions

1. **C1 — Calibrated predictions.** Per-class isotonic regression reduces Expected Calibration Error by 47-98% across all six canonical models, on both datasets.
2. **C2 — Cross-model SHAP agreement (Krishna metrics).** Six agreement metrics from Krishna et al. 2022 quantify how RF, XGBoost, and DNN models disagree on feature importance.
3. **C3 — Explanation stability.** SHAP top-10 rankings tested under Gaussian noise, FGSM, and PGD perturbations.
4. **C4 — SCTS-v2 trust score.** Geometric mean of calibration confidence × explanation stability × conformal coverage.
5. **C5 — LLM-augmented alerts.** Llama-3-8B generates SOC alerts; evaluated via LLM-as-judge.

## Folder structure (this deliverable)

```
deliverable/
├── notebooks/          # All 15-16 Jupyter notebooks (run in order)
├── paper_tables/       # 8 paper-ready CSVs + headline figure
├── figures/            # All supporting figures (PNG)
├── raw_tables/         # Intermediate CSVs (input to paper tables)
├── llm_alerts/         # Sample alerts and judge scores
├── README.md           # This file
└── PROJECT_SUMMARY.md  # One-page summary for review
```

## Datasets

- **NSL-KDD:** Official UNB download; 148,517 samples; 122 features (after one-hot); 5-class labels
- **CIC-IDS2017:** Kaggle mirror; 2.83M flows → 200K stratified subsample; 78 features; 5-class mapped

## Headline findings

See `PROJECT_SUMMARY.md` for the one-page narrative.

## Reproducibility

- Environment: Google Colab Pro (T4 GPU)
- Random seed: 42 throughout
- Code available at github.com/anasbiswas1/xids-research

## Limitations

1. Engelen WTMC-2021 corrections not applied to CIC-IDS2017 (future work)
2. GPT-4o validation deferred to paper writing phase
3. UNSW-NB15 and CICIoT-2023 extensions planned for future work
4. Baseline reproductions (Notebook 10) are representative, not exact
5. U2R class on CIC-IDS2017 has only 36 samples — high-variance F1
