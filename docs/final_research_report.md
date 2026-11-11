# UpliftLab: Final Exploratory Research Report

*Status: validation-stage publication; no confirmatory causal claim and no held-out test evaluation.*

## Abstract

This project compares six heterogeneous treatment-effect estimators for personalized promotional targeting using the X5 RetailHero data: S-, T-, X-, R-, and DR-Learners and an honest Causal Forest. We analyze a fixed train/validation split, uplift ranking, offline policy values, hypothetical treatment costs and maximum treatment budgets. A second model run excludes three historical temporal features whose timing relative to intervention has not been verified. The original specification gives X-Learner the largest zero-threshold policy-value point estimate, but the interval for its improvement over treat-all includes zero. Rankings are sensitive to policy constraints and feature specification. Importantly, the treatment allocation mechanism, design propensity and all engineered feature timestamps were not independently established. The findings must therefore be interpreted as **exploratory and conditional on causal assumptions**, not verified intervention effects.

## Question and scope

**Can individualized treatment targeting improve expected conversion outcomes compared with simple treatment policies, and are the results robust to model choice, treatment cost, budget and feature engineering?**

The study uses X5 RetailHero only. ACIC 2016 was considered but **no ACIC experiments or PEHE results were produced**. No production deployment, cost-in-dollars model, or prospective intervention trial was performed.

## Data and experimental design

- Labeled population: **200,039** customers; **139,868 train**, **30,109 validation**, **30,062 held-out test (unopened)**.
- Binary communication/treatment indicator and later purchase response; historical customer demographics and purchase-derived behavior supply predictors.
- Historical transaction reference date used for dataset-relative features: **2019-03-18** (latest observed transaction, **not** a verified treatment cutoff).
- Original learners fitted on train and compared on validation; nuisance functions for R-, DR-Learners and Causal Forest use cross-fitting; honest Causal Forest uses EconML. This is a comparison of the implemented configurations, **not** a tuned contest across all possible hyperparameters.
- Treatment fraction in training: **0.5004933222752881**. Analyses use this empirical fraction as a constant propensity; it is **not verified as the treatment design probability**.
- Original and conservative feature specifications are compared on the **same** validation customers, which supports a sensitivity contrast but cannot resolve model-selection uncertainty.

## Original validation results

Values below are provisional offline estimates; policy value is a response-probability scale. Qini and AUUC have the project implementation's normalization, so comparisons should use the code-defined metrics, not assume interchangeable conventions from other libraries.

| Model | Qini | AUUC | Zero-threshold policy value | Treated fraction |
|---|---:|---:|---:|---:|
| S-Learner | 0.006999 | 0.026779 | 0.640067 | 0.9793 |
| T-Learner | 0.004666 | 0.024447 | 0.632975 | 0.8241 |
| X-Learner | 0.007042 | 0.026822 | 0.642019 | 0.9579 |
| R-Learner | 0.005837 | 0.025618 | 0.637354 | 0.9191 |
| DR-Learner | 0.006245 | 0.026026 | 0.638152 | 0.9192 |
| Causal Forest | 0.003575 | 0.023355 | 0.637075 | 0.8882 |

Baseline IPW estimates: **treat-all 0.640041**, **treat-none 0.600480**. X-Learner's fixed-policy difference versus treat-all is **+0.001979** with a paired 95% bootstrap interval **[-0.001944, +0.005835]**; zero is inside the interval. The T-Learner's original zero-threshold policy was below treat-all in this analysis, but comparisons of many candidates involve multiple testing and selection effects. None of the learned zero-threshold policies demonstrated a statistically clear superiority over treat-all in the supplied comparison.

Bootstrap uncertainty here is **conditional on trained scores/policies** and does not include nuisance-fit variability, model refitting, choosing models or tuning policies on validation data. The intervals are not a substitute for a prospectively specified trial or confirmed identification assumptions.

## Budgets and hypothetical intervention costs

We evaluated **12 scenarios**: maximum treatment fractions 10%, 30%, 50% and 100%; hypothetical per-treated-customer costs 0, 0.01 and 0.03 **in purchase-probability-equivalent units** (not dollars). At constrained budgets, treat-all can be shown as an *unconstrained reference* but is not a feasible alternative; equal-budget random selection is a suitable feasibility-aligned baseline.

The model with the largest scenario net-policy-value point estimate changes by feature specification:

| Cost | Maximum budget | Original model leader | Conservative model leader |
|---:|---:|---|---|
| 0.00 | 10% | S-Learner | R-Learner |
| 0.00 | 30% | S-Learner | Causal Forest |
| 0.00 | 50% | DR-Learner | T-Learner |
| 0.00 | 100% | X-Learner | Causal Forest |
| 0.01 | 10% | S-Learner | R-Learner |
| 0.01 | 30% | S-Learner | Causal Forest |
| 0.01 | 50% | DR-Learner | T-Learner |
| 0.01 | 100% | X-Learner | S-Learner |
| 0.03 | 10% | S-Learner | R-Learner |
| 0.03 | 30% | S-Learner | Causal Forest |
| 0.03 | 50% | DR-Learner | T-Learner |
| 0.03 | 100% | DR-Learner | T-Learner |

All 12 point-estimate leaders change. This is **descriptive sensitivity**, not 12 independent hypothesis tests. Some differences are tiny (for example, X- and R-Learner at the conservative 10% budget), and no claim of a statistically meaningful difference follows from a change of nominal winner.

For the earlier *draft*, validation-selected **S-Learner / 30% budget / cost 0 / equal-budget random comparator**, the original S-Learner net-value point estimate was **0.622765**; under conservative features it is **0.617183** (a decrease of **0.005581**). The conservative **Causal Forest** at this scenario has an estimated **0.623950**. These comparisons have **not** been certified with paired refitting or model-selection-adjusted intervals, and the candidate was selected with knowledge of validation results.

At cost 0.03 and maximum budget 100%, original DR-Learner leads the six models at about **0.616577** versus conservative T-Learner at about **0.619533**. The cost units are hypothetical. The comparison should not be described as a dollar ROI or evidence of actual commercial benefit.

## Feature-specification sensitivity

Three temporal variables were omitted from all models for the sensitivity run:

1. `historical_recency_days_dataset_relative`
2. `historical_lines_last_30d_dataset_relative`
3. `historical_lines_last_90d_dataset_relative`

The M8.3 runner verified the source feature-table hash against M8.1 and used only **169,977 training and validation rows**, leaving 30,062 test rows excluded. It refitted six estimators and recomputed 12 policy scenarios on validation. The resulting sensitivity shows materially unstable point-estimate rankings. This **does not prove leakage** in the omitted columns, nor establish correct timing for the remaining purchase-derived fields.

## What the evidence supports

1. All six methods and the evaluation pipeline ran in the local project environment; the maintainer reported **85 passing tests** at commit `74d7b74`.
2. Offline rankings and policy values are contingent on feature choice and decision constraints; there is no single robust point-estimate winner.
3. The simple treat-all reference remains competitive when intervention is assumed free; the reported original policy-value differences generally do not establish gains over it.
4. Stronger identification evidence would be required before claiming that targeting *causes* increased conversion, or that one method has a statistically verified policy advantage.

## Limitations and research integrity

- **Unknown assignment design:** The original competition defines communication and purchase outcomes but independently verified randomized allocation, eligibility rules and design propensity were not available to this audit.
- **Unknown timing:** The original data description calls purchases historical, yet an exact campaign cutoff and per-customer feature availability have not been verified.
- **Hypothetical costs:** 0, 0.01 and 0.03 are response-equivalent scenarios; no measured promotion cost or margin was provided.
- **Validation reuse and multiplicity:** Choices of learners, budgets and specifications were informed by the same validation outcomes. Point-estimate winners are not post-selection-confirmed findings.
- **Conditional inference:** Bootstrap intervals were calculated on fixed scores/policies; they do not capture refitting/selection uncertainty or unverified propensity.
- **Dataset rights:** Repository code may be Apache-2.0, but the external dataset's license and redistribution rights remain separately unverified.
- **Incomplete planned benchmark:** ACIC 2016 and semi-synthetic known-effect benchmarks were not executed.
- **Held-out test not opened:** The protocol-readiness gate was blocked, so no test-based result is reported and none should be implied.

## Conclusion

The central outcome is a careful **exploratory, reproducible comparison**, not a proof that sophisticated uplift modeling generates incremental revenue or better interventions. Model choice, costs, maximum budgets and feature definitions change estimated best practices, while unresolved assignment and timing assumptions prevent confirmatory causal interpretation. These negative and robustness findings are useful for responsible treatment-policy modeling.

## Reproduction, inputs and report basis

- Code and milestone docs: [repository README](../README.md), [reproducibility guide](final_reproducibility.md).
- Local inputs: `models/m6_comparison/evaluation_summary.json`, `models/m7_policy_analysis/policy_analysis.json`, `models/m8_1_audit/audit_summary.json`, `models/m8_2_protocol/readiness.json`, `models/m8_3_sensitivity/sensitivity_comparison.csv`.
- The scenario table and conservative model values above were checked against the **user-provided M8.3 `sensitivity_comparison.csv`**. The original learner values and uncertainty intervals are taken from the user's M6/M7 outputs. The table is a documentation snapshot, not new independent evaluation or a newly run randomized trial.
- Study artifacts may be excluded from Git; reproducibility requires separately acquiring licensed X5 data, following the dataset instructions, and regenerating local model files.
