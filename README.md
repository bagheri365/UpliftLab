# UpliftLab

A reproducible research project for comparing heterogeneous treatment-effect estimators and evaluating personalized intervention policies.

## Research question

**Can an individualized treatment policy improve expected outcomes compared with treating everyone or treating no one?**

We will compare S-, T-, X-, R-, and DR-Learners and Causal Forests using consistent features, splits, and evaluation procedures. The main evaluation targets are uplift ranking, estimated policy value, uncertainty, and runtime—not ordinary predictive accuracy alone.

## Documentation

Start with the [documentation index](docs/index.md), then explore:

- [Research questions and hypotheses](docs/research_questions.md)
- [Datasets and provenance](docs/datasets.md)
- [Feature dictionary](docs/data_dictionary.md)
- [Reproduce data preparation](docs/reproducibility.md)
- [Model comparison and evaluation protocol](docs/evaluation_protocol.md)
- [Causal assumptions and limitations](docs/limitations.md)

**Current research status:** M0–M2.1 code and validation are implemented. Documentation checks were introduced in M2.2. The treatment assignment mechanism and feature timing still require confirmation before causal claims are made.

Validate documentation links with `python scripts/check_docs.py`.

## Datasets

- **X5 RetailHero**: primary real-world retail uplift dataset. Verify treatment assignment and pre-treatment feature timing before causal analysis.
- **ACIC 2016**: supplementary semi-synthetic causal benchmark with known simulation effects, where available.

Datasets are **not** checked into Git. See `data/README.md` for handling rules.

## Milestones

- **M0:** Bootstrap repository, dependency management, tests, and data contracts.
- **M1:** Verify dataset provenance, treatment assignment, outcomes, and timing.
- **M2:** Build reproducible customer feature pipeline and splits.
- **M2.1:** Enrich historical customer behavior features (timing verification pending).
- **M2.2:** Establish research documentation and reproducibility guides.
- **M3:** Implement simple baselines and S-/T-Learners.
- **M4:** Implement X-/R-/DR-Learners and Causal Forest.
- **M5:** Compare models using held-out uplift ranking and policy value estimates.
- **M6:** Test sensitivity, uncertainty, subgroup performance, and robustness.
- **M7:** Benchmark against ACIC simulations with treatment-effect ground truth.
- **M8:** Package a reproducible report and documented conclusions.

## Local setup

With Python 3.12 and an activated virtual environment:

```bash
python -m pip install -e '.[dev]'
pytest
```

Data validation, reproducible splits, and behavioral feature engineering are implemented (M1–M2.1). Uplift estimators have not yet been implemented. Download datasets separately; raw and derived data must remain outside version control.

## Project layout

```text
src/upliftlab/     Reusable Python package
src/upliftlab/data/  Validation and feature engineering modules
src/upliftlab/models/  Uplift estimators (future)
src/upliftlab/evaluation/  Metrics and policy evaluation (future)
docs/              Research documentation and methodology
scripts/           Documentation validation tools
tests/             Automated tests
configs/           Experiment configurations (future)
notebooks/         Exploratory analyses (future)
data/              Local datasets, excluded from version control
```
