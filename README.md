# UpliftLab

A reproducible research project for comparing heterogeneous treatment-effect estimators and evaluating personalized intervention policies.

## Research question

**Can an individualized treatment policy improve expected outcomes compared with treating everyone or treating no one?**

We will compare S-, T-, X-, R-, and DR-Learners and Causal Forests using consistent features, splits, and evaluation procedures. The main evaluation targets are uplift ranking, estimated policy value, uncertainty, and runtime—not ordinary predictive accuracy alone.

## Datasets

- **X5 RetailHero**: primary real-world retail uplift dataset. Verify treatment assignment and pre-treatment feature timing before causal analysis.
- **ACIC 2016**: supplementary semi-synthetic causal benchmark with known simulation effects, where available.

Datasets are **not** checked into Git. See `data/README.md` for handling rules.

## Milestones

- **M0:** Bootstrap repository, dependency management, tests, and data contracts.
- **M1:** Verify dataset provenance, treatment assignment, outcomes, and timing.
- **M2:** Build reproducible, leakage-safe pre-treatment feature pipeline and splits.
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

The first patch contains only a package smoke test. It does **not** yet implement uplift estimators or download datasets.

## Project layout

```text
src/upliftlab/     Reusable Python package
src/upliftlab/data/  Data contracts and loaders (future)
src/upliftlab/models/  Uplift estimators (future)
src/upliftlab/evaluation/  Metrics and policy evaluation (future)
tests/             Automated tests
configs/           Experiment configurations (future)
notebooks/         Exploratory analyses (future)
data/              Local datasets, excluded from version control
```
