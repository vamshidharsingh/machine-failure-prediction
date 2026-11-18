"""Loading and validating the NASA C-MAPSS files."""

import pytest

from mfp.data import DataError, load_cmapss, validate


def test_loads_three_files(cmapss_dir):
    train, test, test_rul = load_cmapss(cmapss_dir)
    assert train["unit"].nunique() == 30
    assert test["unit"].nunique() == 12
    assert len(test_rul) == 12
    assert list(test_rul.columns) == ["unit", "true_rul"]


def test_missing_file_has_clear_message(tmp_path):
    with pytest.raises(DataError, match="Missing file"):
        load_cmapss(tmp_path)


def test_rejects_gaps_in_cycles(cmapss_dir):
    train, _, _ = load_cmapss(cmapss_dir)
    broken = train.drop(index=train.index[5])
    with pytest.raises(DataError, match="not consecutive"):
        validate(broken, "train")


def test_rejects_rul_count_mismatch(cmapss_dir):
    (cmapss_dir / "RUL_FD001.txt").write_text("10\n20\n")
    with pytest.raises(DataError, match="test engines"):
        load_cmapss(cmapss_dir)
