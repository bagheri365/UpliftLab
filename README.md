# UpliftLab

**A reproducible, exploratory benchmark of six uplift learning methods for personalized treatment targeting.**

UpliftLab compares S-, T-, X-, R-, and DR-Learners and an honest Causal Forest on the X5 RetailHero promotional-treatment dataset. It evaluates uplift ranking, estimated offline policy value, targeting budgets, hypothetical treatment costs, and sensitivity to feature design. The emphasis is on reproducible analysis and honest limits of causal identification—not claiming that an algorithm reliably improves outcomes without sufficient evidence.

## Main findings

- **No learned zero-threshold policy showed a statistically clear gain over treat-all** on the original validation specification. X-Learner had the highest policy-value point estimate (64.202% versus 64.004% for treat-all), but its 95% conditional bootstrap interval for the difference included zero.
- **There is no stable winning learner.** The best estimated policy depends on the budget, treatment cost and feature set.
- **Feature choice matters.** Removing three dataset-relative temporal predictors changed the point-estimate leader in every one of the 12 evaluated cost–budget scenarios.
- **Causal inference remains assumption-dependent.** The assignment protocol, design treatment probability and all per-feature pre-treatment timestamps were not independently verified. Results are exploratory, not established causal effects.

See the [final research report](docs/final_research_report.md) for numerical results, methods and limitations. **The held-out test split (30,062 rows) has not been evaluated.**

## Dataset and methods

| Item | Description |
|---|---|
| Dataset | X5 RetailHero; download separately (no data redistribution in this repository) |
| Labeled customers | 200,039 |
| Split | 139,868 train / 30,109 validation / 30,062 untouched test |
| Methods | S-, T-, X-, R-, DR-Learner; Causal Forest (EconML) |
| Evaluation | Qini, AUUC, top-K uplift, exploratory IPW policy value, paired conditional bootstrap |
| Robustness | 3 temporal predictors omitted; six models refit on train only; validation re-evaluated |
| Status | **Completed exploratory study; confirmatory causal evaluation not claimed** |

The original feature specification used dataset-relative recency/activity features based on **2019-03-18**, the last observed historical transaction date—not a verified campaign date. The conservative specification omits `historical_recency_days_dataset_relative`, `historical_lines_last_30d_dataset_relative`, and `historical_lines_last_90d_dataset_relative`. Remaining features are not proven pre-treatment for every customer.

## Reproduce

Tested by the project maintainer with Python 3.12 (85 tests passing at commit `74d7b74`). From a clean checkout, create a virtual environment and run:

```bash
python -m pip install -e '.[dev]'
python -m pip install -r requirements-m6.txt  # EconML Causal Forest backend
pytest
python scripts/check_docs.py
```

Download and prepare X5 data following [the data instructions](data/README.md) and [feature pipeline guide](docs/reproducibility.md); datasets and trained outputs are intentionally ignored by Git. For exact experiment commands and artifact paths, see the [reproduction guide](docs/final_reproducibility.md). The command sequence assumes the original dataset and the milestone outputs have already been prepared; it does not auto-download licensed data.

## Results and documentation

- [Final research report](docs/final_research_report.md) — methodology, results and limitations
- [Final reproducibility guide](docs/final_reproducibility.md) — commands, inputs, artifacts and integrity checks
- [Release checklist](docs/release_checklist.md) — how to tag a publication release without claiming confirmatory evidence
- [Documentation index](docs/index.md) — detailed milestone-specific research and implementation notes
- [Dataset provenance](docs/datasets.md), [feature dictionary](docs/data_dictionary.md), [evaluation protocol](docs/evaluation_protocol.md), [limitations](docs/limitations.md)

**Scope note:** ACIC 2016 was considered as a potential future semi-synthetic benchmark but is **not included in the reported experiments**. The project-level Apache-2.0 license does **not** imply a right to redistribute X5 data; the dataset's license and redistribution terms require separate review.

## Repository layout

```text
src/upliftlab/data/          Validation and feature engineering
src/upliftlab/models/        Six uplift learner implementations
src/upliftlab/evaluation/    Ranking, policy, audits, sensitivity and reporting
docs/                       Research report and methodology
configs/                    Draft protocol (not prospectively registered)
tests/                      Automated tests
data/                       Local input data (not checked in)
models/                     Local experiment outputs (not checked in)
```

**Research question:** When, if ever, can a personalized intervention policy improve estimated outcomes relative to simpler treatment strategies under realistic budgets, costs and uncertain causal assumptions?
