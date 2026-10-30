# M2.1 — Historical behavior enrichment

This additive module extends the existing M2 customer features with spending,
quantity, approximate product diversity, active purchase days and dataset-relative
recency and recent-window purchase counts. It does **not** change `x5_features.py`.

Run after generating `data/interim/x5_features.csv`:

```bash
python -m upliftlab.data.x5_enrich \
  --features data/interim/x5_features.csv \
  --purchases data/raw/x5/purchases.csv.gz \
  --output data/interim/x5_features_enriched.csv
```

The purchase file is read in streaming passes, keeping memory roughly proportional
to the number of labeled customers; the existing M2 features CSV is not overwritten.
The feature join is by `client_id`, preserving split, treatment and outcome columns.

**Timing limitations:** Purchase history is described as pre-communication in the
published dataset, but per-customer assignment timestamps have not been verified.
The newest recorded purchase date *among labeled customers* anchors recency and
30/90-day windows; this is **not** a validated campaign cutoff. Do not interpret
these fields as campaign-relative until timing is confirmed. Monetary semantics
of `purchase_sum` also require audit. Product diversity is approximated with a
1024-bit bitmap, not an exact unique-product count. Missing optional source
columns yield blank feature values rather than fabricated zeroes.
