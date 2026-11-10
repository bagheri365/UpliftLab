# M8 — Validation Review and Pre-Test Gate

## Research objective

Before a final report or use of the held-out test split, audit how the six models compare under hypothetical costs and capacity constraints, and clearly document unresolved causal assumptions.

## Inputs and assumptions

- Input: `models/m7_policy_analysis/policy_analysis.json` generated in M7.
- Data source: X5 RetailHero validation results. No raw test data is accessed or required.
- The assumed treatment propensity comes from the observed training assignment frequency, **not a verified randomization design probability**.
- All results are exploratory. Feature availability before intervention is not independently established.
- Cost values are *purchase-probability equivalents* and are not actual marketing costs.

## What the review fixes

M7 printed an unconstrained `treat_all` comparison even for budgets below 100%. That policy is **not feasible** at a 10%, 30%, or 50% cap. M8 labels this explicitly and supplies an expected-value **random-at-budget** reference:

\[
V_{\mathrm{random}(b)} = b V_{\mathrm{all}} + (1-b)V_{\mathrm{none}}.
\]

The baseline is feasible **in expectation**, with random targeting independent of user features. It uses each reference policy's *already cost-adjusted* value. The result is a point estimate; **no uncertainty interval is computed for its paired difference**. No significance claim is allowed on this new comparison.

For any budget, `treat_none` is feasible. `treat_all` is feasible only for budget = 1.0.

## Run

```bash
python -m upliftlab.evaluation.m8_review \
  --analysis models/m7_policy_analysis/policy_analysis.json \
  --output models/m8_validation_review

pytest
git status
```

Outputs are `review_summary.json`, `feasible_policy_comparison.csv`, and `validation_research_review.md` under `models/m8_validation_review/` (ignored as generated artifacts).

## Required evidence before final held-out evaluation

1. Document treatment assignment mechanism using the original experimental protocol. Equal numbers in each group do not prove randomization.
2. Verify that all features precede assignment and that dataset-relative windows do not contain post-treatment purchases. If not verifiable, omit affected features, retrain, and document the sensitivity analysis.
3. Choose the **single primary** research claim, metric, learner selection rule, budget/cost scenario, and feasible comparator **before** looking at the test set.
4. Obtain paired uncertainty for the chosen comparison and account for selection/multiple comparisons in any claims.
5. Audit dataset license, file lineage, and reproducibility requirements.
6. Only after those gates are met, run a final test once and write a report separating exploratory from confirmatory evidence.

This milestone does **not** certify the causal assumptions or unlock the test set. If original documentation is unavailable, the final report must remain a qualified observational/assumption-dependent analysis.
