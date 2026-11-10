# Data dictionary

## Label and identifiers

| Column | Meaning | Model input? |
| --- | --- | --- |
| `client_id` | Customer join key; use for splitting/deduplication | No |
| `treatment_flg` | Promotional communication indicator (0/1) | Treatment only |
| `target` | Binary observed purchase response (0/1) | Outcome only |
| `split` | Deterministic train/validation/test assignment | No |

The exact source label names should be confirmed against the local CSV and validation code before introducing a new loader.

## Baseline profile and history features

| Feature | Interpretation | Caveat |
| --- | --- | --- |
| `age` | Reported age | Check missingness and outliers |
| `gender` | Reported gender encoding | Check value semantics |
| `has_profile` | Profile matched in join | Constant in current labeled data |
| `historical_purchase_lines` | Number of historical purchase lines | Counts rows, not necessarily unique transactions |

## M2.1 enriched behavior

| Feature | Interpretation | Caveat |
| --- | --- | --- |
| `historical_active_days` | Distinct historical activity dates | Verify event timestamp provenance |
| `historical_spend` | Sum of purchase-line monetary values | Confirm monetary units and refunds |
| `historical_avg_line_spend` | Average purchase-line spending | Denominator is lines |
| `historical_quantity` | Sum of purchased quantities | Confirm quantity semantics |
| `historical_unique_products_approx` | Approximate product variety | Approximation must be documented in implementation |
| `historical_recency_days_dataset_relative` | Days since last purchase relative to a global dataset date | Not confirmed treatment-relative |
| `historical_span_days` | Duration between first/last recorded activity | May reflect data-collection window |
| `historical_lines_last_30d_dataset_relative` | Lines in last 30 days of global purchase coverage | Not confirmed treatment-relative |
| `historical_lines_last_90d_dataset_relative` | Lines in last 90 days of global purchase coverage | Not confirmed treatment-relative |

The enrichment run reported a dataset-relative date of **2019-03-18** and processed **22,882,690** purchase lines for the labeled population. These do not establish a campaign date. Exclude features with unresolved post-treatment timing from any causal claims.
