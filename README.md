# UpliftLab

**When does personalized treatment targeting outperform treating everyone—or treating no one?**

UpliftLab is a reproducible, **exploratory** study of six uplift-learning methods for personalized promotional targeting using the X5 RetailHero dataset. It compares uplift rankings, estimated policy values, treatment-budget constraints, hypothetical intervention costs, and feature-specification sensitivity.

The project rule is simple:

> Compare models on the same customers, preserve the uncertainty and negative results, and never turn unverified causal assumptions into causal claims.

## At a Glance

- **Research question:** can individualized treatment decisions improve estimated purchase response relative to simpler treatment policies?
- **Study stage:** final validation-stage research report; held-out test not opened
- **Dataset:** X5 RetailHero, **200,039** labeled customers
- **Splits:** 139,868 training / 30,109 validation / **30,062 untouched test**
- **Methods:** S-, T-, X-, R-, DR-Learner and honest Causal Forest
- **Original validation leader:** X-Learner on Qini and zero-threshold policy-value point estimates, **not** a statistically established winner
- **Robustness finding:** the point-estimate leader changed in **all 12** cost–budget scenarios after removing three temporal predictors
- **Important limitation:** randomization, design propensity and complete pre-treatment feature timing remain unverified

**Read the evidence:** [Final research report](docs/final_research_report.md) · [Reproducibility guide](docs/final_reproducibility.md) · [Research audit](docs/m8_1_research_audit.md)

## Why This Project Exists

Predicting who will buy is not the same as predicting **whose behavior changes because of an intervention**. A high-propensity customer may purchase anyway; a lower-propensity customer may be persuadable.

Uplift modeling tries to rank customers by the difference between their potential outcomes under treatment and no treatment. UpliftLab asks:

> Do more sophisticated heterogeneous-effect estimators improve practical targeting decisions—and do their apparent advantages survive changes in budgets, costs and features?

The study compares six approaches under a shared data split:

| Approach | Basic idea |
|---|---|
| S-Learner | One outcome model with treatment as a predictor |
| T-Learner | Separate outcome models by treatment arm |
| X-Learner | Imputed treatment effects combined across arms |
| R-Learner | Residualized outcome and treatment to estimate heterogeneity |
| DR-Learner | Doubly robust pseudo-outcomes with nuisance cross-fitting |
| Causal Forest | Honest tree-based heterogeneous-effect estimation (EconML) |

## Key Findings

### 1. No Clear Gain Over Treat-All in the Original Zero-Threshold Comparison

All values are **exploratory validation estimates**, using the training treatment fraction as an unverified constant propensity. Qini and AUUC follow this repository's implementations.

| Model | Qini | AUUC | Estimated policy value | Treated |
|---|---:|---:|---:|---:|
| S-Learner | 0.00700 | 0.02678 | 64.007% | 97.93% |
| T-Learner | 0.00467 | 0.02445 | 63.297% | 82.41% |
| **X-Learner** | **0.00704** | **0.02682** | **64.202%** | 95.79% |
| R-Learner | 0.00584 | 0.02562 | 63.735% | 91.91% |
| DR-Learner | 0.00625 | 0.02603 | 63.815% | 91.92% |
| Causal Forest | 0.00357 | 0.02336 | 63.708% | 88.82% |

**Treat-all:** 64.004% · **Treat-none:** 60.048%.

X-Learner's policy-value difference against treat-all was **+0.198 percentage points**, with a paired fixed-policy 95% bootstrap interval of **[−0.194, +0.584] percentage points**. The interval contains zero, so the apparent gain is **not statistically conclusive**. These intervals do not include training or model-selection uncertainty.

### 2. Costs and Budgets Change Which Model Looks Best

UpliftLab compares maximum treatment budgets of **10%, 30%, 50% and 100%**, and hypothetical costs of **0, 0.01 and 0.03** per treated customer in *purchase-probability-equivalent units*—**not dollars**.

- At a 10% budget, S-Learner led the original feature-specification point estimates.
- At a 30% budget, S-Learner also led the original specification.
- At a 50% budget, DR-Learner led the original specification.
- At a 100% maximum budget, the original winner depended on intervention cost.

**Fair-comparison rule:** treat-all is an unconstrained reference, **not a feasible policy** when the budget is below 100%. Equal-budget random targeting is a feasible comparator. A budget is a maximum, not an instruction to spend it all.

### 3. Removing Three Temporal Features Changes All 12 Point-Estimate Winners

The conservative-feature sensitivity run omitted:

- `historical_recency_days_dataset_relative`
- `historical_lines_last_30d_dataset_relative`
- `historical_lines_last_90d_dataset_relative`

These features used the last observed historical transaction date (**2019-03-18**) as a reference, **not** a verified campaign date. All six methods were refit on training data and re-evaluated on the same validation split. The test split was excluded.

| Maximum budget | Original winner (cost 0) | Conservative winner (cost 0) |
|---:|---|---|
| 10% | S-Learner | R-Learner* |
| 30% | S-Learner | Causal Forest |
| 50% | DR-Learner | T-Learner |
| 100% | X-Learner | Causal Forest |

*At the conservative 10% budget, R-Learner and X-Learner are effectively tied by point estimate.* Across all three costs, the nominal winning learner changed in each of the **12** scenarios. This is **feature-specification sensitivity**, not proof of temporal leakage or of statistically significant model superiority.

### 4. The Causal Assumptions Audit Remains Open

The original dataset documentation describes purchases as historical, but this study did **not** independently establish the treatment allocation procedure, design probability, exact intervention cutoff or complete per-customer feature availability.

Consequently, policy-value calculations rely on assumptions that have not been confirmed. **The reported values are not established causal effects, and no confirmatory test-set result is claimed.**

## Dataset and Evaluation Design

| Component | Details |
|---|---|
| Data | X5 RetailHero promotional communication and subsequent purchase response |
| Labeled customers | 200,039 |
| Train / validation / test | 139,868 / 30,109 / 30,062 |
| Treatment fraction in training | 0.5004933; empirical frequency, **not** verified design propensity |
| Features | Demographics and historical retail-behavior aggregates |
| Metrics | Qini, AUUC, top-K uplift, provisional IPW policy values |
| Uncertainty | Paired bootstrap conditional on fixed scores and policies |
| Scenarios | 3 treatment costs × 4 maximum budgets |
| Robustness | Six-model retraining after excluding three temporal variables |
| Test status | **Untouched; pre-test audit gate closed** |

Data must be acquired separately. See [dataset documentation](docs/datasets.md), [data instructions](data/README.md), and the [feature dictionary](docs/data_dictionary.md). The repository's code license does **not** establish permission to redistribute the external dataset.

## Research Evolution

> validate data → engineer features → compare learners → evaluate policies → audit assumptions → test robustness → publish exploratory findings

| Phase | Main question | Result |
|---|---|---|
| M0–M2.2 | Can data, features and documentation be reproduced? | Validated data, enriched features and documented splits |
| M3–M3.1 | How do simple meta-learners perform? | S-/T-Learners and a shared uplift evaluation framework |
| M4–M6 | Do more advanced learners improve performance? | X-, R-, DR-Learners and Causal Forest implemented |
| M7 | Do costs and budgets change policy rankings? | 12 cost–budget scenarios evaluated |
| M8–M8.2 | Are the causal assumptions and test protocol ready? | Important evidence gaps found; gate remains closed |
| M8.3 | Are results robust to uncertain temporal features? | Six models refit; winners changed across 12 scenarios |
| M9 | What can be published responsibly? | Final exploratory report and reproducibility guidance |

## Negative and Unresolved Results

UpliftLab preserves findings that do **not** support a simple success story:

- No learned original zero-threshold policy established a statistically clear advantage over treat-all.
- No model consistently leads across the original and conservative feature specifications.
- Temporal-feature sensitivity is substantial, but does not prove leakage.
- The treatment assignment mechanism and true treatment propensity remain unverified.
- Validation-guided model and policy selection is **not** prospective pre-registration.
- ACIC 2016 was contemplated but **not run**; there are no ACIC/PEHE benchmark results.
- Held-out test outcomes were intentionally not evaluated.

See the [final report](docs/final_research_report.md) and [research audit](docs/m8_1_research_audit.md) for details.

## Reproducibility

The maintainer reported **85 passing tests** at commit `74d7b74` before the M9 documentation update. This number is a historical observation, not a claim about a new checkout.

```bash
python -m pip install -e '.[dev]'
python -m pip install -r requirements-m6.txt  # Causal Forest / EconML
pytest
python scripts/check_docs.py
```

Dataset downloads, feature preparation, training, comparison and report-generation commands are in the [final reproducibility guide](docs/final_reproducibility.md). Training and evaluation outputs under `models/` and external data under `data/` are local artifacts, not bundled research evidence in this repository.

**Evidence-integrity note:** M8.1 recorded the source feature-table SHA-256 as:

`84218869a26677d701a2682c84bef5e89d45b1e6f473bcebf628a93bd8625295`

This hash identifies the locally audited enriched table; it does not verify feature timing or treatment assignment.

## Repository Structure

```text
UpliftLab/
├── configs/                  # draft evaluation protocol
├── data/                     # local inputs and data instructions
├── docs/                     # research reports, audits and guides
├── models/                   # local fitted models and evaluation outputs (ignored)
├── scripts/                  # documentation and data utilities
├── src/upliftlab/
│   ├── data/                  # validation and feature engineering
│   ├── models/                # six uplift learners
│   └── evaluation/            # metrics, policy analysis, audits, sensitivity
├── tests/                    # automated tests
├── pyproject.toml
└── README.md
```

## Running Locally

After installing dependencies and preparing the dataset, start with `pytest` and follow the [reproduction guide](docs/final_reproducibility.md) for the milestone-by-milestone pipeline. The project does **not** auto-download third-party data or automatically unlock the test set.

## Experimental Discipline

- Maintain a fixed train/validation/test split.
- Fit models on training data and compare on the same validation customers.
- Keep the final test split sealed while the research gate is closed.
- Compare feasible policies at equal budgets; show treat-all only as an unconstrained reference when appropriate.
- State whether treatment costs are observed or hypothetical.
- Report intervals and identify what uncertainty they omit.
- Preserve unfavorable results and unresolved assumptions rather than removing them from the narrative.

## Limitations

- Assignment randomization, eligibility and design treatment propensity have not been independently verified.
- Exact campaign timing and per-customer pre-treatment availability of all historical features are unknown.
- Policy-value inference assumes a constant propensity estimated from training frequency.
- Bootstrap intervals condition on fixed fitted models/policies; they do not capture model selection or refitting.
- Feature-specification and model comparisons reuse validation data; no multiplicity-adjusted confirmatory claim is made.
- The intervention costs are hypothetical conversion-equivalent values, not monetary ROI.
- Dataset redistribution terms require independent review.
- The ACIC known-effect experiment and held-out test evaluation were not completed.

## What the Experiments Suggest

The evidence favors a **decision-focused** conclusion rather than a universal model ranking:

> Which uplift learner looks best depends on treatment budget, intervention cost and which features are available before the decision.

That result is useful, but it does not prove a model would improve outcomes in deployment. Verifying assignment and feature timing—and running a separately pre-specified evaluation—would be necessary for stronger causal conclusions.

## Future Research

- Recover and independently cite the original treatment-assignment protocol and campaign cutoff.
- Repeat the benchmark with verified pre-treatment covariates and known assignment probabilities.
- Add a semi-synthetic dataset with known treatment effects (for example, ACIC 2016).
- Compare practical monetary utility once actual treatment cost and margin are available.
- Evaluate a genuinely frozen policy on an untouched holdout **only after** the evidence gate can be justified.

## Milestones

| Milestone | Outcome |
|---|---|
| M0–M2.2 | Repository, data, features and documentation |
| M3–M6 | Six uplift learning methods and shared evaluation |
| M7 | Policy budgets and intervention costs |
| M8–M8.2 | Research review, assumptions audit and draft protocol |
| M8.3 | Conservative-feature sensitivity experiment |
| M9 | Exploratory publication and release preparation |

## Current Conclusion

UpliftLab asks:

> **When is individualized treatment targeting worth it?**

The current validation evidence says there is **no universally best learner** and **no confirmed advantage over treating everyone**. Cost, capacity and feature choices substantially change estimated policy rankings. The project's main contribution is a reproducible six-method comparison that also documents the uncertainty and limits of its causal interpretation.

---

[Final research report](docs/final_research_report.md) · [Documentation index](docs/index.md) · [Reproducibility guide](docs/final_reproducibility.md) · [Release checklist](docs/release_checklist.md)
