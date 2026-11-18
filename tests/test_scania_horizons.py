"""Horizon labels, class conversion, weights, calibration and selection.

Horizon labels, class-probability conversion, vehicle weights, calibration and the simulated
validation set.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mfp.config import ScaniaConfig
from mfp.scania.data import load_split
from mfp.scania.features import FleetRelative, readout_features
from mfp.scania.horizons import (
    AtRiskCalibrator,
    balanced_rows,
    class_shares,
    cumulative_to_classes,
    fixed_prior_weights,
    horizon_labels,
    time_left,
    vehicle_weights,
)


def test_horizon_labels_keep_censored_readouts_where_known():
    #             repaired vehicle            not repaired (censored)
    left = np.array([100, 30, 10, 3, -2, 100, 30, 10, 3])
    repaired = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    nan = np.nan
    assert np.array_equal(
        horizon_labels(left, repaired, 6), [0, 0, 0, 1, nan, 0, 0, 0, nan], equal_nan=True
    )
    assert np.array_equal(
        horizon_labels(left, repaired, 12), [0, 0, 1, 1, nan, 0, 0, nan, nan], equal_nan=True
    )
    assert np.array_equal(
        horizon_labels(left, repaired, 48), [0, 1, 1, 1, nan, 0, nan, nan, nan], equal_nan=True
    )
    # Edge: exactly h steps left counts as failing within h.
    assert horizon_labels(np.array([6.0]), np.array([1]), 6)[0] == 1
    assert horizon_labels(np.array([6.0]), np.array([0]), 6)[0] != 0


def test_cumulative_to_class_probabilities():
    cum = np.array(
        [
            [0.1, 0.2, 0.3, 0.5],  # well ordered
            [0.3, 0.2, 0.4, 0.35],  # horizons disagree: running max makes them ordered
            [0.0, 0.0, 0.0, 0.0],
            [1.2, 1.0, 1.0, 1.0],  # out of range is clipped
        ]
    )
    p = cumulative_to_classes(cum)
    assert np.allclose(p[0], [0.5, 0.2, 0.1, 0.1, 0.1])
    assert np.allclose(p[1], [0.6, 0.0, 0.1, 0.0, 0.3])
    assert np.allclose(p[2], [1, 0, 0, 0, 0])
    assert np.allclose(p[3], [0, 0, 0, 0, 1])
    assert (p >= 0).all() and np.allclose(p.sum(axis=1), 1)


def test_vehicle_weights_give_each_vehicle_equal_total():
    ids = np.array([1, 1, 1, 1, 2, 3, 3])
    w = vehicle_weights(ids)
    totals = pd.Series(w).groupby(ids).sum()
    assert np.allclose(totals, totals.iloc[0])
    assert w.mean() == pytest.approx(1.0)
    assert w[0] == pytest.approx(w[4] / 4)


def test_balanced_rows_give_each_vehicle_the_same_rows():
    vehicles = np.array([1] * 40 + [2] * 3 + [3] * 10 + [4] * 5)
    mask = vehicles != 4
    rows = balanced_rows(vehicles, mask, seed=1)
    counts = pd.Series(vehicles[rows]).value_counts()
    assert set(counts.index) == {1, 2, 3}  # never outside the mask
    assert (counts == 10).all()  # median rows per vehicle
    assert np.array_equal(rows, balanced_rows(vehicles, mask, seed=1))  # seeded
    assert len(balanced_rows(vehicles, mask, per_vehicle=2, seed=1)) == 6


def test_fixed_prior_weights_have_no_search():
    train = class_shares([0] * 90 + [1] * 8 + [4] * 2)
    target = class_shares([0] * 95 + [1] * 1 + [4] * 4)
    w = fixed_prior_weights(target, train)
    assert np.allclose(w, [0.95 / 0.9, 0.01 / 0.08, 1.0, 1.0, 0.04 / 0.02])
    weighted = class_shares([0, 0, 1], weights=[1, 1, 2])
    assert np.allclose(weighted, [0.5, 0.5, 0, 0, 0])


def test_at_risk_calibrator_keeps_class_mix_and_sums_to_one():
    probs = np.array(
        [
            [0.9, 0.05, 0.0, 0.0, 0.05],
            [0.5, 0.25, 0.0, 0.0, 0.25],
            [0.2, 0.4, 0.0, 0.0, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    at_risk = np.array([0, 0, 1, 0])
    cal = AtRiskCalibrator().fit(probs, at_risk)
    out = cal.transform(probs)
    assert np.allclose(out.sum(axis=1), 1) and (out >= 0).all()
    # Classes 1 and 4 keep their 1:1 ratio wherever there is risk mass.
    assert np.allclose(out[:3, 1], out[:3, 4])
    # Monotone in the original at-risk probability.
    assert (np.diff(1 - out[:3, 0]) >= 0).all()


def test_simulated_validation_rows_are_past_only_and_seeded(scania_dir):
    from mfp.scania.selection import simulated_validation

    train = load_split(scania_dir, "train", use_cache=False)
    feats = readout_features(train["readouts"], extras=("intensity", "rate10"))
    ordered = train["readouts"].loc[feats.index]
    left, repaired = time_left(ordered, train["tte"])
    feats = feats.reset_index(drop=True)
    sim = simulated_validation(feats, left, repaired, n_folds=5, seed=7)
    again = simulated_validation(feats, left, repaired, n_folds=5, seed=7)
    pd.testing.assert_frame_equal(sim, again)
    assert sim["vehicle_id"].is_unique and set(sim["fold"]) == set(range(5))
    # Censored vehicles cut within 48 steps of the study end are skipped.
    assert not ((repaired[sim["row"]] == 0) & (left[sim["row"]] <= 48)).any()

    # Features at the chosen readout equal features computed from the
    # readouts truncated at that point.
    ro = ordered.reset_index(drop=True)
    for r in sim.itertuples():
        cut = ro[
            (ro["vehicle_id"] == r.vehicle_id) & (ro["time_step"] <= ro.loc[r.row, "time_step"])
        ]
        last = readout_features(cut, extras=("intensity", "rate10")).iloc[-1]
        pd.testing.assert_series_equal(last, feats.iloc[r.row], check_names=False)
        if r.Index >= 5:
            break


def test_fleet_relative_uses_training_medians_only():
    feats = pd.DataFrame(
        {
            "vehicle_id": [1, 1, 2, 3],
            **{
                f"{c}_intensity": [1.0, 3.0, 2.0, 10.0]
                for c in ("171", "666", "427", "837", "309", "835", "370", "100")
            },
        }
    )
    specs = pd.DataFrame({"vehicle_id": [1, 2, 3], "Spec_0": ["Cat0", "Cat0", "Cat1"]})
    fleet = FleetRelative().fit(feats.iloc[:3], specs)  # vehicle 3 not in training
    out = fleet.transform(feats, specs)
    assert out["171_vs_fleet"].tolist()[:3] == [0.5, 1.5, 1.0]  # median of Cat0 = 2
    assert out["171_vs_fleet"].iloc[3] == pytest.approx(5.0)  # unseen Cat1: overall median 2


def test_selection_runs_without_test_files(tmp_path, scania_dir):
    from mfp.scania.selection import run_selection

    for name in ("test_operational_readouts.csv", "test_specifications.csv", "test_labels.csv"):
        (scania_dir / name).unlink()
    cfg = replace(
        ScaniaConfig(),
        data_dir=scania_dir,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        selection_folds=3,
        boosting_params={"learning_rate": 0.1, "max_leaf_nodes": 7, "max_iter": 20},
        prior_weight_grid=(0.5, 1.0, 2.0),
    )
    out = run_selection(cfg, log=lambda *a, **k: None)
    table, choice = out["table"], out["choice"]
    assert (cfg.reports_dir / "model_selection.csv").exists()
    assert (cfg.reports_dir / "selection.json").exists()
    assert set(table["calibration"]) == {
        "none",
        "prior_shift",
        "prior_shift_isotonic",
        "searched_on_validation",
    }
    assert {"five_class", "horizons"} <= set(table["target"])
    assert choice["calibration"] != "searched_on_validation"
    eligible = table[
        (table["calibration"] != "searched_on_validation")
        & (table["learner"] == "gradient_boosting")
    ]
    assert choice["learner"] == "gradient_boosting"
    assert choice["sim_cost"] == eligible["sim_cost"].min()
    assert choice["baseline"]["learner"] == "logistic_regression"
    for col in (
        "sim_pr_auc",
        "sim_roc_auc",
        "sim_brier",
        "sim_healthy_flagged_share",
        "sim_recall",
        "val_cost",
        "val_pr_auc",
    ):
        assert table[col].notna().all(), col
