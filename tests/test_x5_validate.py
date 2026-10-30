"""Fixtures deliberately synthetic: no restricted data in the repository."""

import csv

import pytest

from upliftlab.data.x5_validate import validate_x5


def write_csv(tmp_path, rows, columns=("client_id", "treatment_flg", "target")):
    path = tmp_path / "uplift_train.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_valid_data(tmp_path):
    path = write_csv(tmp_path, [
        {"client_id": "a", "treatment_flg": "0", "target": "0"},
        {"client_id": "b", "treatment_flg": "0", "target": "1"},
        {"client_id": "c", "treatment_flg": "1", "target": "1"},
        {"client_id": "d", "treatment_flg": "1", "target": "1"},
    ])
    report = validate_x5(path)
    assert report["rows"] == 4
    assert report["treatment_counts"] == {"0": 2, "1": 2}
    assert report["response_rates"] == {"0": 0.5, "1": 1.0}
    assert report["observed_response_difference"] == 0.5


@pytest.mark.parametrize("rows,reason", [
    ([{"client_id": "a", "treatment_flg": "1", "target": "1"},
      {"client_id": "a", "treatment_flg": "0", "target": "1"}], "Duplicate"),
    ([{"client_id": "a", "treatment_flg": "x", "target": "1"}], "Nonbinary"),
    ([{"client_id": "a", "treatment_flg": "1", "target": ""}], "Nonbinary"),
    ([{"client_id": "a", "treatment_flg": "1", "target": "1"}], "Both treatment"),
    ([], "no observations"),
])
def test_bad_inputs(tmp_path, rows, reason):
    with pytest.raises(ValueError, match=reason):
        validate_x5(write_csv(tmp_path, rows))


def test_missing_columns(tmp_path):
    path = write_csv(tmp_path, [{"client_id": "a", "treatment_flg": "1"}],
                     columns=("client_id", "treatment_flg"))
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_x5(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_x5(tmp_path / "missing.csv")
