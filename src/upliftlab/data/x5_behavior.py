"""Streaming, bounded-memory behavioral aggregates from pre-communication line items.

Absent optional source columns produce blank (not invented zero) feature values.
Date features use the latest date in the source as a *dataset-relative* anchor,
not an unverified campaign date.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from math import isfinite

from .x5_features import as_number, rows

FEATURES = (
    'historical_active_days', 'historical_spend', 'historical_avg_line_spend',
    'historical_quantity', 'historical_unique_products_approx',
    'historical_recency_days_dataset_relative', 'historical_span_days',
    'historical_lines_last_30d_dataset_relative',
    'historical_lines_last_90d_dataset_relative',
)


def _day(value):
    try:
        return date.fromisoformat(value[:10]) if value else None
    except ValueError:
        return None


def aggregate(purchases, eligible_ids):
    """Return per-customer features and observed optional schema capabilities.

    Two streaming passes keep memory proportional to number of labeled customers.
    The approximate product diversity uses a compact 1024-bit bitmap per user.
    """
    eligible_ids = set(eligible_ids)
    sums = defaultdict(float)
    quantities = defaultdict(float)
    priced = defaultdict(int)
    masks = defaultdict(int)
    first = {}
    last = {}
    day_masks = defaultdict(set)
    n30 = defaultdict(int)
    n90 = defaultdict(int)
    maximum = None
    cols = None
    for row in rows(purchases, ('client_id',)):
        if cols is None:
            cols = set(row)
        cid = row['client_id']
        if cid not in eligible_ids:
            continue
        if 'transaction_datetime' in row:
            d = _day(row['transaction_datetime'])
            if d:
                maximum = max(maximum, d) if maximum else d
                first[cid] = min(first[cid], d) if cid in first else d
                last[cid] = max(last[cid], d) if cid in last else d
                day_masks[cid].add(d.toordinal())
        if 'purchase_sum' in row:
            value = as_number(row['purchase_sum'])
            if value is not None and isfinite(value):
                sums[cid] += value
                priced[cid] += 1
        if 'product_quantity' in row:
            value = as_number(row['product_quantity'])
            if value is not None and isfinite(value):
                quantities[cid] += value
        product = row.get('product_id', '')
        if product:
            import hashlib
            bit = int.from_bytes(hashlib.blake2b(product.encode(), digest_size=4).digest(), 'big') % 1024
            masks[cid] |= 1 << bit
    if maximum and 'transaction_datetime' in (cols or set()):
        for row in rows(purchases, ('client_id', 'transaction_datetime')):
            cid = row['client_id']
            if cid not in eligible_ids:
                continue
            d = _day(row['transaction_datetime'])
            if d is not None:
                delta = (maximum - d).days
                if 0 <= delta < 30:
                    n30[cid] += 1
                if 0 <= delta < 90:
                    n90[cid] += 1
    available = cols or set()
    result = {}
    for cid in eligible_ids:
        result[cid] = (
            len(day_masks[cid]) if 'transaction_datetime' in available else '',
            round(sums[cid], 6) if 'purchase_sum' in available and priced[cid] else '',
            round(sums[cid] / priced[cid], 6) if 'purchase_sum' in available and priced[cid] else '',
            round(quantities[cid], 6) if 'product_quantity' in available else '',
            masks[cid].bit_count() if 'product_id' in available else '',
            (maximum - last[cid]).days if maximum and cid in last else '',
            (last[cid] - first[cid]).days if cid in last else '',
            n30[cid] if maximum and cid in last else '',
            n90[cid] if maximum and cid in last else '',
        )
    return result, sorted(available), maximum.isoformat() if maximum else None
