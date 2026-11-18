"""NASA labels, past-only features and engine splits."""

import numpy as np
import pandas as pd
import pytest

from mfp.data import load_cmapss
from mfp.features import (
    add_label,
    add_test_rul,
    add_train_rul,
    build_features,
    feature_columns,
    informative_sensors,
    last_cycle_rows,
)
from mfp.split import engine_split, three_way_split


def test_labels_on_hand_checked_example():
    # One engine that fails after cycle 35, window of 30 cycles.
    df = pd.DataFrame({"unit": 1, "cycle": range(1, 36)})
    out = add_label(add_train_rul(df), window=30)
    assert out.loc[out["cycle"] == 1, "rul"].item() == 34
    assert out.loc[out["cycle"] == 4, "label"].item() == 0  # RUL 31
    assert out.loc[out["cycle"] == 5, "label"].item() == 1  # RUL 30
    assert out.loc[out["cycle"] == 35, "rul"].item() == 0
    assert out["label"].sum() == 31


def test_test_rul_counts_back_from_given_value():
    df = pd.DataFrame({"unit": [7, 7, 7], "cycle": [1, 2, 3]})
    out = add_test_rul(df, pd.DataFrame({"unit": [7], "true_rul": [50]}))
    assert out["rul"].tolist() == [52, 51, 50]


def test_constant_sensors_are_dropped(cmapss_dir):
    train, _, _ = load_cmapss(cmapss_dir)
    used = informative_sensors(train)
    for s in ["s1", "s5", "s6", "s10", "s16", "s18", "s19", "op3"]:
        assert s not in used
    assert "s2" in used


def test_features_use_no_future_cycles(cmapss_dir):
    """Deleting an engine's later cycles must not change its earlier features."""
    train, _, _ = load_cmapss(cmapss_dir)
    train = add_label(add_train_rul(train), 30)
    sensors = informative_sensors(train)
    full = build_features(train, sensors, window=10)

    cut = 60
    truncated = train[~((train["unit"] == 1) & (train["cycle"] > cut))]
    partial = build_features(truncated, sensors, window=10)

    cols = feature_columns(full)
    a = full[(full["unit"] == 1) & (full["cycle_id"] <= cut)][cols].to_numpy()
    b = partial[(partial["unit"] == 1) & (partial["cycle_id"] <= cut)][cols].to_numpy()
    np.testing.assert_allclose(a, b)


def test_no_missing_feature_values(cmapss_dir):
    train, _, _ = load_cmapss(cmapss_dir)
    feats = build_features(add_label(add_train_rul(train), 30), informative_sensors(train), 10)
    assert not feats[feature_columns(feats)].isna().any().any()


def test_trend_matches_manual_calculation():
    df = pd.DataFrame({"unit": 1, "cycle": range(1, 7), "s2": [0.0, 1, 3, 6, 10, 15]})
    feats = build_features(df, ["s2"], window=3)
    # cycle 2: (1 - 0) / 1; cycle 6: (15 - 6) / 2
    assert feats["s2_trend"].tolist() == pytest.approx([0.0, 1.0, 1.5, 2.5, 3.5, 4.5])


def test_engine_split_has_no_shared_engines(cmapss_dir):
    train, _, _ = load_cmapss(cmapss_dir)
    fit, val = engine_split(train, val_fraction=0.2, seed=42)
    assert set(fit["unit"]).isdisjoint(set(val["unit"]))
    assert set(fit["unit"]) | set(val["unit"]) == set(train["unit"])
    assert val["unit"].nunique() == 6


def test_engine_split_is_reproducible(cmapss_dir):
    train, _, _ = load_cmapss(cmapss_dir)
    _, val_a = engine_split(train, 0.2, seed=1)
    _, val_b = engine_split(train, 0.2, seed=1)
    assert set(val_a["unit"]) == set(val_b["unit"])


def test_three_way_split_has_no_shared_engines(cmapss_dir):
    train, _, _ = load_cmapss(cmapss_dir)
    fit, val, life = three_way_split(train, 0.2, 0.2, seed=42)
    f, v, lt = set(fit["unit"]), set(val["unit"]), set(life["unit"])
    assert f.isdisjoint(v) and f.isdisjoint(lt) and v.isdisjoint(lt)
    assert f | v | lt == set(train["unit"])
    assert (len(f), len(v), len(lt)) == (18, 6, 6)
    assert len(fit) + len(val) + len(life) == len(train)


def test_three_way_split_is_reproducible(cmapss_dir):
    train, _, _ = load_cmapss(cmapss_dir)
    a = three_way_split(train, 0.2, 0.2, seed=42)
    b = three_way_split(train, 0.2, 0.2, seed=42)
    c = three_way_split(train, 0.2, 0.2, seed=7)
    for x, y in zip(a, b, strict=True):
        assert set(x["unit"]) == set(y["unit"])
    assert set(a[2]["unit"]) != set(c[2]["unit"])


def test_last_cycle_rows_one_per_engine(cmapss_dir):
    _, test, test_rul = load_cmapss(cmapss_dir)
    labelled = add_label(add_test_rul(test, test_rul), 30)
    feats = build_features(labelled, informative_sensors(test), 10)
    last = last_cycle_rows(feats)
    assert len(last) == 12
    assert (
        last.set_index("unit")["rul"].to_dict() == test_rul.set_index("unit")["true_rul"].to_dict()
    )
