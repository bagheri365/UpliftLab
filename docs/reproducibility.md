# Reproducibility guide

## Environment

Python 3.12 with a virtual environment. From the repository root:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Obtain data

Download the X5 files from the [dataset mirror](https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift/tree/main/data) to `data/raw/x5/`. Do not commit downloaded data or generated outputs.

## Run M1 validation

```bash
python -m upliftlab.data.x5_validate \
  data/raw/x5/uplift_train.csv \
  --output data/interim/x5_validation.json
```

## Build baseline M2 features and deterministic splits

```bash
python -m upliftlab.data.x5_features \
  --train data/raw/x5/uplift_train.csv \
  --clients data/raw/x5/clients.csv.gz \
  --purchases data/raw/x5/purchases.csv.gz \
  --output data/interim/x5_features.csv
```

Expected local split counts with seed 42: train 139,868; validation 30,109; test 30,062. Preserve this split for method comparisons.

## Enrich with M2.1 historical features

```bash
python -m upliftlab.data.x5_enrich \
  --features data/interim/x5_features.csv \
  --purchases data/raw/x5/purchases.csv.gz \
  --output data/interim/x5_features_enriched.csv
```

The local enrichment run produced 200,039 records and nine additional features. This may require several minutes and substantial I/O due to the large compressed purchase file.

## Documentation quality

```bash
python scripts/check_docs.py
pytest
```

For formal reproducibility, future milestones should log package versions, data hashes, code commit, seeds, treatment-propensity assumptions, and experiment configurations. Test split must not be used to choose models or thresholds.
