import csv
import gzip

import pytest

from upliftlab.data.x5_enrich import enrich


def make_csv(path, columns, rows):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        writer.writerows(rows)


def test_enrich_additive_and_treatment_untouched(tmp_path):
    base = tmp_path / 'features.csv'
    purchases = tmp_path / 'purchases.csv.gz'
    output = tmp_path / 'enriched.csv'
    make_csv(base, ['client_id','split','age','historical_purchase_lines','treatment_flg','target'],
             [('a','train',30,2,1,1),('b','test',45,0,0,0)])
    make_csv(purchases, ['client_id','transaction_datetime','purchase_sum','product_quantity','product_id'],
             [('a','2020-01-01',10,1,'p1'),('a','2020-01-03',20,3,'p2'),('z','2023-01-01',50,2,'p4')])
    result = enrich(base, purchases, output)
    with output.open() as f:
        items = {r['client_id']:r for r in csv.DictReader(f)}
    assert result['rows'] == 2
    assert items['a']['historical_spend'] == '30.0'
    assert items['a']['historical_quantity'] == '4.0'
    assert items['a']['historical_recency_days_dataset_relative'] == '0'
    assert items['a']['treatment_flg'] == '1' and items['a']['target'] == '1'
    assert items['b']['historical_spend'] == ''
    assert items['b']['split'] == 'test'


def test_reject_overwriting_input(tmp_path):
    base = tmp_path / 'features.csv'
    make_csv(base, ['client_id'], [('a',)])
    with pytest.raises(ValueError, match='Output must differ'):
        enrich(base, tmp_path/'purchases.csv.gz', base)


def test_reject_duplicates(tmp_path):
    base = tmp_path / 'features.csv'
    make_csv(base, ['client_id'], [('a',),('a',)])
    with pytest.raises(ValueError, match='Duplicate'):
        enrich(base, tmp_path/'purchases.csv.gz', tmp_path/'out.csv')
