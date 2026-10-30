"""M2: reproducible customer-level X5 features, using only documented pre-campaign history.

Standard library only; purchases are processed in a streaming fashion.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

DATA_URL = 'https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift/resolve/main/data/'
FILES = ('uplift_train.csv.gz', 'clients.csv.gz', 'purchases.csv.gz')


def open_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing dataset file: {path}')
    return gzip.open(path, 'rt', newline='', encoding='utf-8') if path.suffix == '.gz' else path.open('r', newline='', encoding='utf-8')


def rows(path, required):
    with open_csv(path) as stream:
        reader = csv.DictReader(stream)
        absent = set(required) - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f'{path}: missing columns {sorted(absent)}')
        yield from reader


def split_for(client_id, seed=42):
    """Stable 70/15/15 split at customer level, independent of outcome."""
    value = int.from_bytes(hashlib.sha256(f'{seed}:{client_id}'.encode()).digest()[:8], 'big') / 2**64
    return 'train' if value < .70 else ('validation' if value < .85 else 'test')


def as_number(value):
    try:
        return float(value) if value not in ('', None) else None
    except (ValueError, TypeError):
        return None


def build(train, clients, output, purchases=None, seed=42):
    """Build a model-ready CSV without including outcomes in engineered features."""
    labels = {}
    for row in rows(train, ('client_id', 'treatment_flg', 'target')):
        cid = row['client_id']
        if not cid or cid in labels:
            raise ValueError('Blank or duplicate client_id in uplift training labels')
        if row['treatment_flg'] not in ('0', '1') or row['target'] not in ('0', '1'):
            raise ValueError(f'Nonbinary treatment/target for {cid}')
        labels[cid] = (row['treatment_flg'], row['target'])
    if not labels:
        raise ValueError('Training labels are empty')

    # The public dataset does not guarantee first_redeem_date predates the campaign.
    # Exclude both dates until a campaign cutoff is explicitly confirmed.
    profiles = {}
    for row in rows(clients, ('client_id', 'age', 'gender')):
        cid = row['client_id']
        if cid in labels:
            if cid in profiles:
                raise ValueError(f'Duplicate client profile: {cid}')
            age = as_number(row['age'])
            profiles[cid] = (age if age is not None and 0 <= age <= 110 else '', row['gender'] if row['gender'] in ('M', 'F', 'U') else 'U')

    counts = defaultdict(int)
    # Source documentation describes purchases as *prior to communication*.
    # Each row is a line item, not necessarily a distinct shopping trip.
    if purchases:
        for row in rows(purchases, ('client_id',)):
            cid = row['client_id']
            if cid in labels:
                counts[cid] += 1

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ('client_id', 'split', 'age', 'gender', 'has_profile', 'historical_purchase_lines', 'treatment_flg', 'target')
    split_counts = defaultdict(int)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for cid in sorted(labels):
            age, gender = profiles.get(cid, ('', 'U'))
            split = split_for(cid, seed)
            split_counts[split] += 1
            writer.writerow((cid, split, age, gender, int(cid in profiles), counts[cid], *labels[cid]))
    return {'rows': len(labels), 'profiles_matched': len(profiles), 'splits': dict(split_counts), 'historical_purchase_lines': sum(counts.values()), 'output': str(output), 'seed': seed, 'feature_columns': list(header[2:-2])}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build model-ready X5 features from pre-campaign data')
    parser.add_argument('--train', default='data/raw/x5/uplift_train.csv')
    parser.add_argument('--clients', default='data/raw/x5/clients.csv.gz')
    parser.add_argument('--purchases', default=None, help='Optional purchases.csv.gz (large: ~609 MB compressed)')
    parser.add_argument('--output', default='data/interim/x5_features.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.train, args.clients, args.output, args.purchases, args.seed), indent=2))


if __name__ == '__main__':
    main()
