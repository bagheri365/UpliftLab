# M2: X5 model-ready features

## Scope

This is a **conservative baseline** that keeps raw files out of Git and uses only customer demographics (`age`, `gender`) and an optional count of historical purchase line items. The dataset documentation describes `purchases.csv` as purchase history *before communication*. Source: https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html

We deliberately exclude `first_issue_date` and `first_redeem_date` until their relation to the treatment date has been verified. Age values outside 0–110 are marked missing. No treatment or outcome is used to create features or determine splits. Purchase line counts are **not** distinct transactions.

## Download

The full purchase file is ~609 MB compressed; start with clients only.

```bash
mkdir -p data/raw/x5
curl -L --fail -o data/raw/x5/clients.csv.gz 'https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift/resolve/main/data/clients.csv.gz'
# Optional, much larger:
curl -L --fail -o data/raw/x5/purchases.csv.gz 'https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift/resolve/main/data/purchases.csv.gz'
```

## Build

```bash
python -m upliftlab.data.x5_features --train data/raw/x5/uplift_train.csv --clients data/raw/x5/clients.csv.gz --output data/interim/x5_features.csv
# Include purchase history if downloaded:
python -m upliftlab.data.x5_features --purchases data/raw/x5/purchases.csv.gz
pytest
```

`split` is a deterministic SHA-256 customer-level 70/15/15 assignment (`train`, `validation`, `test`) using seed 42. Use the same split across all uplift estimators. Because splits are not stratified, inspect treatment/outcome support by split before fitting learners. The output includes `client_id`, feature columns, `treatment_flg`, and `target`; never include identifier, split, treatment, or target in the covariate matrix.

## M2 acceptance checklist

- [ ] Validate the exact columns and provenance for clients and purchases.
- [ ] Confirm cohort coverage and treatment/control/response counts within each split.
- [ ] Verify all chosen features predate communication; documentation alone does not establish an exact event cutoff.
- [ ] Audit for missing or implausible demographic fields.
- [ ] Freeze the feature schema and split seed for all downstream estimators.

Avoid claims of unbiased causal effects until randomization or conditional ignorability assumptions are documented. No datasets are committed to Git.
