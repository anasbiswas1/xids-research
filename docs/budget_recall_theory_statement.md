# Fixed-budget recall certificate: statement and assumptions


Setting. A detector scores each network flow. An analyst can inspect exactly k flows per deployment window,
where k is fixed by staffing and is an input, not something the method may choose. Write R_k for the fraction
of the attacks present in that window which fall inside the k highest-scoring flows.

Assumption. The labelled certification sample and the deployment window are exchangeable, and the score is
continuous (no exact ties), which is arranged by breaking ties uniformly at random inside the gap to the next
distinct value, so that no two genuinely different scores are ever reordered.

Construction. Fix delta and split it as delta = d1 + d2 + d3.
  Stage 1 (count).  Choose the largest calibration rank r whose implied number of deployment flows above the
                    corresponding threshold tau is at most k, using an upper Clopper-Pearson bound on the
                    score exceedance rate followed by an upper binomial quantile on the deployment count.
                    With probability at least 1 - d1 the set {score >= tau} in the window has at most k
                    members, hence is contained in the inspected top-k.
  Stage 2 (rate).   Let q = P(score >= tau | attack). With probability at least 1 - d2 the lower
                    Clopper-Pearson bound q_lo computed from the certification sample satisfies q_lo <= q.
  Stage 3 (count of attacks). Given the number m of attacks realised in the window, the number of them above
                    tau is Binomial(m, q). With probability at least 1 - d3 its fraction is at least the
                    lower binomial quantile evaluated at q_lo, which is monotone in q.

Claim. With probability at least 1 - delta, R_k is at least the returned bound. The three failure events are
unioned, so no independence between stages is required.

Why fixing k is not the usual setting. Conformal risk control and Learn-then-Test fix a risk level and search
for a threshold, so the size of the selected set is an output and may grow until the risk is met. Fixing the
count removes that freedom and introduces the stage-1 quantity, the number of flows above the threshold, which
those methods never have to control. Conformal selection controls the false-discovery side and likewise lets
the selected set float. Dropping stage 1 or stage 3 from the construction is measured as an ablation in
notebook 18 and loses validity on real data.

Limits. Validity is conditional on exchangeability; the bound is expected to fail where a benchmark's
partitions are constructed to differ, which is measured on NSL-KDD. The bound is conservative by construction
and becomes vacuous when attacks are very rare, the certification sample is small and the budget is tight; the
operating region where it is non-vacuous is mapped below.

