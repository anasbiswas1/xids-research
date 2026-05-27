# UNSW-NB15 — Design Decisions and Methodology Notes

**Project:** Calibrated and Stability-Aware Explainable Intrusion Detection: A Multi-Model SHAP Framework for SOC Decision Support
**Author:** Md Anas Biswas, University of Portsmouth
**Purpose of this document:** Defensible justifications for the design decisions made when adding UNSW-NB15 as a third dataset to the X-IDS framework. Each section is drop-in text for the paper's methodology, with citations marked `[ref]` for filling in during writing.

---

## 1. Dataset choice and partition

UNSW-NB15 [Moustafa & Slay, 2015] was selected as the third evaluation dataset alongside NSL-KDD and CIC-IDS2017 for three reasons. First, it was generated on a hybrid testbed using the IXIA PerfectStorm framework, combining realistic background traffic with controlled attack injection, which addresses the lack of contemporary background traffic in NSL-KDD. Second, it contains 49 features extracted using Argus and Bro-IDS, providing a feature schema distinct from both NSL-KDD's connection-level features and CIC-IDS2017's CICFlowMeter features. This heterogeneity strengthens the cross-dataset generalisability claim in our framework. Third, UNSW-NB15 is one of the most-cited IDS benchmarks of the past decade, allowing direct comparison with a large body of prior work.

We use the official train/test partition (175,341 / 82,332 rows) published by the dataset authors rather than re-partitioning the full ~2.54M flow records. This decision was made for two reasons. First, the official partition contains deliberate distribution shift between train and test — the test set has a higher proportion of Normal traffic and a different attack-subtype distribution than the training set — and this shift is integral to the dataset's design philosophy of testing generalisation under realistic non-IID conditions. Repartitioning the full dataset with a random split would discard this property and produce inflated performance estimates. Second, using the official partition allows direct comparison with the large body of UNSW-NB15 literature that uses this same split. We acknowledge that the full set of four raw chunks (~2.54M rows) exists; this is documented as a possible extension in future work.

A note for readers comparing our numbers to the literature: published accuracy on UNSW-NB15 binary classification ranges widely, from ~83% to >99%. The upper end of this range almost exclusively reflects papers that performed a random train/test split on the union of all four raw chunks, which leaks the train-test distribution shift designed into the official partition and produces optimistic estimates [Moustafa & Slay, 2015; Engelen et al., WTMC 2021]. Our results, in the 83–85% binary accuracy range, are consistent with published results that respect the official partition [refs to fill in].

---

## 2. Attack category harmonisation (10-class → 5-class)

A central design choice for cross-dataset analysis is harmonising the three datasets' attack taxonomies into a single five-class scheme: **Normal, DoS, Probe, R2L, U2R**. This taxonomy originates with NSL-KDD and KDD'99 and is one of the most widely used IDS attack categorisations in the literature. We map UNSW-NB15's ten native categories to this scheme as follows:

| UNSW-NB15 category | Mapped to | Rationale |
|---|---|---|
| Normal | Normal | Direct |
| DoS | DoS | Direct |
| Reconnaissance | Probe | Active scanning and recon behaviour |
| Analysis | Probe | Passive recon: port scans, spam, HTML probes |
| Exploits | R2L | Remote-to-local intrusion via known vulnerabilities |
| Fuzzers | R2L | Remote payloads attempting unauthorised access |
| Backdoor | R2L | Persistent remote unauthorised access |
| Shellcode | U2R | Code-execution payloads, privilege escalation vector |
| Worms | U2R | Self-propagating, capable of privilege escalation |
| **Generic** | **Dropped** | See below |

This mapping is a deliberate harmonisation choice for cross-dataset comparison, not a claim about the canonical taxonomy of network attacks. We acknowledge two contestable points and address them directly.

**First, mapping Exploits (the largest UNSW-NB15 attack class) to R2L:** Exploits in UNSW-NB15 consists primarily of vulnerability-exploitation payloads that target services and applications accessible over the network. By the original Lippmann et al. [1999] definition of R2L — "an attacker who does not have an account on the victim machine gains local access" — most Exploits payloads in UNSW-NB15 qualify. Fuzzers and Backdoor similarly aim at gaining unauthorised access or persistence. The alternative would be to introduce a sixth class ("Exploits") used only by UNSW-NB15, which would defeat the purpose of harmonised cross-dataset evaluation. To address concerns about this mapping, Appendix B reports a sensitivity analysis using the native 10-class UNSW labels.

**Second, dropping Generic:** Generic is a UNSW-NB15-specific class consisting of attacks against block ciphers (e.g., known-plaintext attacks against cryptographic primitives). It accounts for approximately 18,871 training rows and 18,871 test rows (about 22% of each partition). Generic has no clean analogue in NSL-KDD or CIC-IDS2017 — neither dataset includes cryptographic protocol attacks. Forcing Generic into the five-class taxonomy would either require an arbitrary mapping (e.g. into R2L or U2R, both of which describe fundamentally different attack mechanics) or the introduction of a sixth class present in only one of three datasets. We chose to drop Generic and document this transparently. The resulting effective dataset sizes are 135,341 training rows and 63,461 test rows. We note this is conservative: keeping Generic mapped to any class would have lowered per-class F1 scores for that class due to the heterogeneous nature of the attacks. Reviewers can verify this trade-off by inspecting the un-mapped raw test set.

---

## 3. Categorical feature encoding

UNSW-NB15 contains three categorical features: `proto` (protocol, 131 unique values), `service` (application protocol, 13 unique values including 'no service'), and `state` (connection state, 11 unique values). We applied one-hot encoding fitted on the union of training and test partitions rather than on the training partition alone. This is a standard precaution against unseen categorical levels at inference time — particularly important for `proto`, where some protocols appear only in one partition. Unified one-hot fitting does not leak label information (it only enumerates the feature vocabulary); it produces a fixed-dimensional feature space across the train/test boundary. The same approach was used for NSL-KDD's `protocol_type`, `service`, and `flag` columns and is documented in our NSL-KDD notebook (`01_data_exploration.ipynb`).

The post-encoding feature dimensionality is 196, compared with approximately 120 for NSL-KDD and 78 for CIC-IDS2017. Cross-dataset SHAP analyses (Notebook 06-UNSW) are dimensionless across this difference because they operate on normalised feature-attribution rankings (Krishna et al., 2022).

---

## 4. Numerical feature standardisation and missing values

We applied `StandardScaler` (zero mean, unit variance) fitted on the training partition only and transformed both train and test, following standard scaling-leakage practice. UNSW-NB15 contains no missing values in the canonical sense, but a small number of rate-based features (`*_rate`, `*_per_sec`) produce `inf` values when the denominator is zero. We replaced `inf` with `NaN` and then `NaN` with `0` before standardisation. The same preprocessing recipe is used for CIC-IDS2017 to maintain consistency.

---

## 5. Validation strategy for DNN training

The deep neural network models (`dnn_binary_cw`, `dnn_5class_smote`) require a held-out validation set for early stopping. We split 10% of the training partition (stratified by 5-class label) as the validation set, used only for monitoring validation loss and triggering early stopping. The remaining 90% of training data was used for parameter updates. This split is internal to the training partition and does not touch the held-out test set used for evaluation. Stratification by 5-class label ensures the validation set contains samples from all five categories (including the rare U2R class) so that early stopping is not driven by majority-class accuracy alone.

For NSL-KDD and CIC-IDS2017 DNN training, the same 90/10 stratified validation strategy was used. On UNSW-NB15, the DNN models were trained for up to 80 epochs with early-stopping patience of 10 epochs on validation loss; the 5-class DNN's best validation loss occurred at epoch 79, indicating mild continued improvement at the budget cutoff. We accept this as a documented limitation: extending the training budget further produced marginal validation-loss reductions but did not change the cross-model ranking on the test set. The binary DNN converged at epoch 20.

---

## 6. SMOTE strategy for 5-class

For 5-class models trained with SMOTE, we did not apply naive equal-balancing of all five classes. Naive balancing would oversample U2R from 1,263 to 56,000 training samples (a 44× synthesis ratio), producing dense clusters of synthetic neighbours from a small set of real seeds and risking overfitting on synthetic distributional artefacts. Instead, we oversampled each attack class to match the size of the largest *attack* class (R2L, ~53,323 training samples), while leaving the Normal class at its original 56,000 samples. This produces a training set in which all attack classes are balanced *with each other* and the Normal class remains slightly larger, with the most aggressive U2R synthesis ratio capped at 42× rather than 44×. The same strategy is documented for CIC-IDS2017 in `02_train_models.ipynb` (Cell 8). NSL-KDD uses the same approach.

---

## 7. Methodological note: single-seed evaluation

All results reported for UNSW-NB15 (and NSL-KDD and CIC-IDS2017) come from single training runs with a fixed random seed (42). This is a known limitation of the present work: error bars from multi-seed training would strengthen the empirical claims. The single-seed choice reflects the compute budget for an MSc thesis and the priority of producing three internally-consistent dataset analyses rather than one statistically rigorous single-dataset analysis. We document this transparently and note that the cross-dataset consistency of our findings (XGBoost > Random Forest > DNN on rare classes across all three datasets) provides an alternative form of robustness evidence to seed-variance: if all three independent datasets produce the same ranking under the same hyperparameters, the ranking is unlikely to be a seed artifact.

---

## 8. What this document is and is not

This document is a record of design decisions made when adding UNSW-NB15 to the X-IDS framework. It is intended to be lifted verbatim or near-verbatim into the paper's methodology section. It deliberately does not contain results, model comparisons, or claims about the framework's contributions — those belong in their own sections. Where a decision is genuinely contestable (the Exploits→R2L mapping; the dropping of Generic), the document acknowledges the contestability rather than defending the decision as objectively correct.

**References to fill in during paper writing:**
- Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems.
- Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study. WTMC 2021.
- Lippmann, R., et al. (1999/2000). The 1999 DARPA off-line intrusion detection evaluation.
- Krishna, S., et al. (2022). The Disagreement Problem in Explainable Machine Learning.
- Kull, M., Filho, T. S., & Flach, P. (2019). Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration. NeurIPS 2019.
- Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining Well Calibrated Probabilities Using Bayesian Binning. AAAI 2015.
