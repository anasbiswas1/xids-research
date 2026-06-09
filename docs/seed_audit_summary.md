# Seed Audit Summary

**Date**: 2026-06-09
**Purpose**: pre-flight checklist for the multi-seed runner

## Scope

Notebooks audited: 14
Missing notebooks: ['02_data_pipeline.ipynb', '02b_unsw_pipeline.ipynb', '02c_cic_pipeline.ipynb', '05_stability_analysis.ipynb']

## Findings

- Seed/RNG references: 39
- File path references: 60

## What the multi-seed runner needs to do

1. **Parameterize all SEED references**: replace every literal `random_state=42`, `seed=42`, `np.random.seed(42)` with the loop variable.
2. **Seed-suffix all writable output paths**: model files, calibrators, SHAP arrays, results tables/JSONs.
3. **Reuse canonical 1000 across seeds**: do NOT re-derive `canonical_eval_idx` per seed.
4. **Phase B not rerun**: 08* notebooks are seed-independent at the threshold-search level.

## Detailed CSVs

- `docs/seed_audit_seed_findings.csv` — every seed/RNG mention with notebook + cell + line context
- `docs/seed_audit_path_findings.csv` — every file-path reference with read/write classification

## Next step

Use these findings to design `00_multi_seed_runner.ipynb`. The runner will:
- Define each pipeline stage as an inline function with explicit seed parameter
- Save outputs to `models/{ds}/seed{N}/` instead of `models/{ds}/`
- Skip stages whose outputs already exist (resumability)
- Run all 9 stages per seed for SEEDS = [123, 456, 789]
