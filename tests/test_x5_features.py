import csv
import gzip

import pytest

from upliftlab.data.x5_features import build, split_for


def write(path, header, data):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(data)


def test_stable_nonoverlapping_splits():
    assert split_for('customer_1') == split_for('customer_1')
    assert len({split_for(f'c{i}') for i in range(200)}) == 3
    assert split_for('customer_1', seed=43) in ('train', 'validation', 'test')


def test_features_prevent_label_leakage_and_handle_gzip(tmp_path):
    train = tmp_path / 'train.csv'
    clients = tmp_path / 'clients.csv.gz'
    purchases = tmp_path / 'purchases.csv.gz'
    output = tmp_path / 'features.csv'
    write(train, ['client_id', 'treatment_flg', 'target'], [('a', 0, 1), ('b', 1, 0), ('c', 1, 1)])
    write(clients, ['client_id', 'age', 'gender', 'first_redeem_date'], [('a', 30, 'F', '2099-01-01'), ('b', -5, 'M', '2099-01-01'), ('z', 35, 'M', '')])
    write(purchases, ['client_id', 'other'], [('a', 4), ('a', 5), ('z', 7)])
    result = build(train, clients, output, purchases)
    with output.open() as stream:
        reader = csv.DictReader(stream)
        assert 'first_redeem_date' not in reader.fieldnames
        assert reader.fieldnames[-2:] == ['treatment_flg', 'target']
        data = {r['client_id']: r for r in reader}
    assert result['rows'] == 3
    assert data['a']['historical_purchase_lines'] == '2'
    assert data['b']['age'] == ''
    assert data['c']['has_profile'] == '0'
    assert data['a']['target'] == '1'
    assert data['c']['historical_purchase_lines'] == '0'
    assert all(v['split'] == split_for(cid) for cid, v in data.items())
    first = output.read_bytes()
    build(train, clients, output, purchases)
    assert output.read_bytes() == first


def test_duplicate_ids_rejected(tmp_path):
    train = tmp_path / 'train.csv'
    clients = tmp_path / 'clients.csv'
    write(train, ['client_id', 'treatment_flg', 'target'], [('a', 0, 0), ('a', 1, 1)])
    write(clients, ['client_id', 'age', 'gender'], [('a', 20, 'M')])
    with pytest.raises(ValueError, match='duplicate'):
        build(train, clients, tmp_path / 'out.csv')


def test_missing_columns_rejected(tmp_path):
    train = tmp_path / 'train.csv'
    clients = tmp_path / 'clients.csv'
    write(train, ['client_id', 'treatment_flg', 'target'], [('a', 0, 0)])
    write(clients, ['client_id'], [('a',)])
    with pytest.raises(ValueError, match='missing columns'):
        build(train, clients, tmp_path / 'out.csv')
