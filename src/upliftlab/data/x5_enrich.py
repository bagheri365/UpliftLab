"""Enrich M2 customer-level CSV with pre-communication purchase aggregates.

This module is additive: M2 x5_features.py is not modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .x5_behavior import FEATURES, aggregate


def enrich(features, purchases, output):
    features, output = Path(features), Path(output)
    if features.resolve() == output.resolve():
        raise ValueError('Output must differ from input to prevent data loss')
    with features.open('r', newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if 'client_id' not in fields:
            raise ValueError('M2 features file requires client_id')
        if any(col in fields for col in FEATURES):
            raise ValueError('Input already contains M2.1 features')
        ids = set()
        for row in reader:
            cid = row['client_id']
            if not cid or cid in ids:
                raise ValueError('Duplicate or missing client_id in feature CSV')
            ids.add(cid)
    behavior, columns, reference_date = aggregate(purchases, ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    with features.open('r', newline='', encoding='utf-8') as source, output.open('w', newline='', encoding='utf-8') as target:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        insert_at = fieldnames.index('treatment_flg') if 'treatment_flg' in fieldnames else len(fieldnames)
        fieldnames[insert_at:insert_at] = list(FEATURES)
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            row.update(dict(zip(FEATURES, behavior.get(row['client_id'], ('',) * len(FEATURES)))))
            writer.writerow(row)
    return {'rows': len(ids), 'output': str(output), 'feature_columns_added': list(FEATURES), 'purchase_columns': columns, 'reference_date': reference_date}


def main(argv=None):
    p = argparse.ArgumentParser(description='Add historical behavioral aggregates to M2 feature CSV')
    p.add_argument('--features', default='data/interim/x5_features.csv')
    p.add_argument('--purchases', default='data/raw/x5/purchases.csv.gz')
    p.add_argument('--output', default='data/interim/x5_features_enriched.csv')
    args = p.parse_args(argv)
    print(json.dumps(enrich(args.features, args.purchases, args.output), indent=2))


if __name__ == '__main__':
    main()
