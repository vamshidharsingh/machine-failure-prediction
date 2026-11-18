"""Scania Component X: labels, features, decisions and the pipeline.

Scania Component X: labels, features, decisions and the pipeline, on small synthetic data in the
real file format (see conftest.write_scania).
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mfp.config import ScaniaConfig
from mfp.scania.data import COUNTERS, HIST_COLS, ScaniaDataError, load_split
from mfp.scania.evaluate import (
    COST_MATRIX,
    bootstrap_costs,
    capacity_decisions,
    capacity_rules,
    cost_from_confusion,
    decide,
    fit_prior_weights,
    reweight,
    rule_decisions,
    summarize,
    total_cost,
)
from mfp.scania.features import SpecEncoder, last_readout_rows, readout_features
from mfp.scania.labels import label_training, time_left_to_class


def test_class_boundaries_on_edge_values():
    t = [0, 0.5, 6, 6.01, 12, 12.01, 24, 24.01, 48, 48.01, 300]
    assert time_left_to_class(t).tolist() == [4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0]


def _readouts(rows):
    df = pd.DataFrame(rows, columns=["vehicle_id", "time_step"])
    values = COUNTERS + [c for cols in HIST_COLS.values() for c in cols]
    return pd.concat([df, pd.DataFrame(1.0, index=df.index, columns=values)], axis=1)


def test_repaired_and_not_repaired_labelling():
    # Vehicle 1 repaired at 100; vehicle 2 not repaired, study ended at 100.
    readouts = _readouts(
        [(1, 10), (1, 52), (1, 60), (1, 90), (1, 100), (2, 10), (2, 51), (2, 52), (2, 80), (2, 100)]
    )
    tte = pd.DataFrame(
        {
            "vehicle_id": [1, 2],
            "length_of_study_time_step": [100.0, 100.0],
            "in_study_repair": [1, 0],
        }
    )
    labels, report = label_training(readouts, tte)

    vehicle_1 = labels[labels["vehicle_id"] == 1]
    # time left 90, 48, 40, 10, 0 -> classes 0, 1, 1, 3, 4 (exact, so all kept)
    assert vehicle_1["class_label"].tolist() == [0, 1, 1, 3, 4]

    vehicle_2 = labels[labels["vehicle_id"] == 2]
    # Not repaired: time left 90 and 49 are certainly > 48 -> class 0.
    # Time left 48, 20 and 0 could be any class -> dropped.
    assert vehicle_2["time_step"].tolist() == [10, 51]
    assert vehicle_2["class_label"].tolist() == [0, 0]
    assert report["dropped_censored_unknown_class"] == 3
    assert report["labelled_readouts"] == 7
    assert report["dropped_after_repair"] == 0


def test_readouts_after_repair_are_dropped():
    readouts = _readouts([(1, 10), (1, 50), (1, 60)])
    tte = pd.DataFrame(
        {"vehicle_id": [1], "length_of_study_time_step": [50.0], "in_study_repair": [1]}
    )
    labels, report = label_training(readouts, tte)
    assert labels["time_step"].tolist() == [10, 50]
    assert report["dropped_after_repair"] == 1


def test_loader_errors_are_clear(scania_dir):
    (scania_dir / "test_labels.csv").unlink()
    with pytest.raises(ScaniaDataError, match="Missing file"):
        load_split(scania_dir, "test")

    labels = pd.read_csv(scania_dir / "validation_labels.csv")
    labels.iloc[1:].to_csv(scania_dir / "validation_labels.csv", index=False)
    with pytest.raises(ScaniaDataError, match="vehicle IDs differ"):
        load_split(scania_dir, "validation")

    ro = pd.read_csv(scania_dir / "train_operational_readouts.csv")
    pd.concat([ro, ro.iloc[:1]]).to_csv(scania_dir / "train_operational_readouts.csv", index=False)
    with pytest.raises(ScaniaDataError, match="duplicate"):
        load_split(scania_dir, "train")


def test_loader_accepts_other_spec_spelling_and_uses_cache(scania_dir):
    (scania_dir / "validation_specifications.csv").rename(
        scania_dir / "validation_specification.csv"
    )
    first = load_split(scania_dir, "validation")
    assert (scania_dir / "cache" / "validation_operational_readouts.parquet").exists()
    second = load_split(scania_dir, "validation")
    pd.testing.assert_frame_equal(first["readouts"], second["readouts"])
    assert first["readouts"]["171_0"].dtype == np.float32


def test_features_do_not_use_later_readouts(scania_dir):
    ro = load_split(scania_dir, "train", use_cache=False)["readouts"]
    full = readout_features(ro, window=5)
    vid = ro["vehicle_id"].iloc[0]
    cut = ro.loc[ro["vehicle_id"] == vid, "time_step"].sort_values().iloc[6]
    trimmed = ro[~((ro["vehicle_id"] == vid) & (ro["time_step"] > cut))]
    part = readout_features(trimmed, window=5)
    a = full[(full["vehicle_id"] == vid) & (full["time_step"] <= cut)]
    b = part[part["vehicle_id"] == vid]
    pd.testing.assert_frame_equal(a, b)
    # The rate uses the window: at the 7th readout it spans readouts 3..7.
    g = ro[ro["vehicle_id"] == vid].sort_values("time_step")
    v, t = g["171_0"].to_numpy(float), g["time_step"].to_numpy(float)
    assert b["171_rate"].iloc[6] == pytest.approx((v[6] - v[2]) / (t[6] - t[2]), rel=1e-5)
    assert np.isnan(b["171_rate"].iloc[0])


def test_histogram_shares_sum_to_one_and_last_rows():
    ro = _readouts([(1, 1.0), (1, 2.5), (2, 1.0)])
    ro.loc[0, HIST_COLS["167"]] = np.arange(10)
    feats = readout_features(ro)
    shares = feats[[f"167_bin{i}_share" for i in range(10)]].to_numpy()
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert feats.loc[0, "167_total"] == 45
    assert last_readout_rows(feats)["time_step"].tolist() == [2.5, 1.0]
    assert feats["n_readouts"].tolist() == [1, 2, 1]


def test_one_hot_categories_learned_from_training_only():
    cols = [f"Spec_{i}" for i in range(8)]
    train = pd.DataFrame({"vehicle_id": [1, 2], **{c: ["Cat0", "Cat1"] for c in cols}})
    test = pd.DataFrame({"vehicle_id": [3], **{c: ["Cat5"] for c in cols}})
    enc = SpecEncoder().fit(train)
    assert len(enc.columns_) == 16 and "Spec_0=Cat5" not in enc.columns_
    out = enc.transform(test)
    assert out.shape == (1, 16) and out.to_numpy().sum() == 0  # unseen -> all zeros
    assert enc.transform(train).loc[1, "Spec_3=Cat0"] == 1.0


def test_cost_matrix_total_on_hand_made_confusion_matrix():
    cm = np.zeros((5, 5), int)
    cm[0, 0], cm[0, 1], cm[0, 4] = 100, 3, 2  # 3*7 + 2*10 = 41
    cm[1, 0], cm[1, 1] = 1, 4  # 200
    cm[3, 2], cm[4, 4] = 2, 5  # 2*200 = 400
    cm[2, 3] = 1  # 7
    assert cost_from_confusion(cm) == 41 + 200 + 400 + 7
    y_true = np.repeat(np.arange(5), cm.sum(axis=1))
    y_pred = np.concatenate([np.repeat(np.arange(5), row) for row in cm])
    assert total_cost(y_true, y_pred) == 648
    s = summarize(y_true, y_pred)
    assert s["confusion_matrix"] == cm.tolist()
    assert s["always_class_0_cost"] == 200 * 5 + 300 * 1 + 400 * 2 + 500 * 5
    ar = s["at_risk"]
    assert (ar["true_at_risk"], ar["false_alarms"], ar["missed"]) == (13, 5, 1)


def test_capacity_decisions_flag_only_the_riskiest_share():
    p = np.array(
        [
            [0.99, 0.01, 0.0, 0.0, 0.0],  # at risk 0.01
            [0.50, 0.10, 0.10, 0.10, 0.20],  # at risk 0.50, expected cost lowest at class 4
            [0.90, 0.10, 0.0, 0.0, 0.0],  # at risk 0.10, unconstrained says class 1
            [0.70, 0.00, 0.0, 0.0, 0.30],  # at risk 0.30
        ]
    )
    # Unlimited capacity: lowest expected cost for every vehicle.
    assert rule_decisions(p, "unconstrained").tolist() == [0, 4, 1, 4]
    # Room for 2 of 4: the two riskiest (rows 1 and 3) keep their class 1-4 with
    # the lowest expected cost; everyone else gets class 0.
    assert capacity_decisions(p, 0.5).tolist() == [0, 4, 0, 4]
    assert rule_decisions(p, "capacity_50").tolist() == [0, 4, 0, 4]
    # Room for 1: only row 1.
    assert capacity_decisions(p, 0.25).tolist() == [0, 4, 0, 0]
    # A flagged vehicle never gets class 0, even if class 0 is cheaper for it:
    # row 0 gets class 1 (expected cost 0.99 x 7 = 6.93 beats 9.99 for class 4).
    assert capacity_decisions(p, 1.0).tolist() == [1, 4, 1, 4]
    assert capacity_decisions(p, 0.0).tolist() == [0, 0, 0, 0]
    # Ties in risk keep the input order.
    same = np.tile([[0.8, 0.0, 0.0, 0.0, 0.2]], (4, 1))
    assert capacity_decisions(same, 0.5).tolist() == [4, 4, 0, 0]
    assert capacity_rules((0.02, 0.1)) == ["unconstrained", "capacity_2", "capacity_10"]


def test_capacity_ties_are_broken_by_the_uncalibrated_score():
    # Four vehicles with the same calibrated risk (as isotonic calibration
    # produces) and room for two: the two with the highest uncalibrated score
    # are flagged, whatever their order in the input.
    same = np.tile([[0.8, 0.0, 0.0, 0.0, 0.2]], (4, 1))
    score = np.array([0.10, 0.40, 0.05, 0.30])
    assert capacity_decisions(same, 0.5, tiebreak=score).tolist() == [0, 4, 0, 4]
    order = np.array([3, 1, 0, 2])
    shuffled = capacity_decisions(same[order], 0.5, tiebreak=score[order])
    assert (shuffled > 0).tolist() == [True, True, False, False]
    # The tie-break never overrides a difference in calibrated risk.
    p = np.vstack([[0.7, 0.0, 0.0, 0.0, 0.3], same[:3]])
    assert capacity_decisions(p, 0.25, tiebreak=[0.0, 0.9, 0.9, 0.9]).tolist() == [4, 0, 0, 0]
    assert rule_decisions(same, "capacity_50", tiebreak=score).tolist() == [0, 4, 0, 4]


def test_lowest_expected_cost_decision_on_hand_made_probabilities():
    p = np.array(
        [
            [0.90, 0.10, 0, 0, 0],  # exp. cost: m0 = 20, m1 = 6.3 -> 1 (argmax says 0)
            [0.99, 0.01, 0, 0, 0],  # m0 = 2, m1 = 6.93 -> 0
            [0.50, 0.00, 0, 0, 0.50],  # m0 = 250, m3 = 104.5, m4 = 5 -> 4
        ]
    )
    assert decide(p, "argmax").tolist() == [0, 0, 0]
    assert decide(p, "lowest_expected_cost").tolist() == [1, 0, 4]
    expected = p @ COST_MATRIX
    assert expected[0, 1] == pytest.approx(6.3) and expected[0, 0] == pytest.approx(20)


def test_prior_correction_reweights_and_picks_cheapest_weights():
    p = np.array([[0.75, 0.25, 0, 0, 0], [0.7, 0.3, 0, 0, 0], [0.95, 0.05, 0, 0, 0]])
    q = reweight(p, [1, 2, 1, 1, 1])
    assert np.allclose(q.sum(axis=1), 1) and q[0, 1] == pytest.approx(0.5 / 1.25)
    y = np.array([1, 1, 0])
    w, cost, n = fit_prior_weights(p, y, "argmax", grid=(0.5, 1.0, 4.0))
    assert n == 3**4
    # Raw argmax predicts 0 for all (cost 400); weight 4 on class 1 catches
    # the two class-1 vehicles and leaves the third at class 0 (cost 0).
    assert cost == 0 and w[1] == 4.0
    # Ties go to the weights nearest no correction: classes 2-4 stay at 1.
    assert w[0] == 1.0 and list(w[2:]) == [1.0, 1.0, 1.0]


def test_bootstrap_intervals_are_seeded():
    y = np.array([0] * 90 + [4] * 10)
    good = y.copy()
    a = bootstrap_costs(y, {"model": good}, n=200, seed=1)
    b = bootstrap_costs(y, {"model": good}, n=200, seed=1)
    assert a == b
    assert a["models"]["model"]["total_cost_95"] == [0.0, 0.0]
    assert a["models"]["model"]["saving_vs_always_class_0_95"][0] > 0


def _cfg(tmp_path, data_dir, **kw):
    return replace(
        ScaniaConfig(),
        data_dir=data_dir,
        reports_dir=tmp_path / "reports" / "scania",
        models_dir=tmp_path / "models" / "scania",
        bootstrap_samples=100,
        tuning_grid={"learning_rate": [0.1], "max_leaf_nodes": [15], "max_iter": [30, 60]},
        prior_weight_grid=(0.5, 1.0, 2.0),
        selection_folds=3,
        boosting_params={"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 50},
        **kw,
    )


def test_scania_pipeline_end_to_end(tmp_path, scania_dir):
    from mfp.scania.pipeline import run

    cfg = _cfg(tmp_path, scania_dir)
    m = run(cfg, tune=True, verbose=False)

    for name in [
        "metrics.json",
        "model_selection.csv",
        "selection.json",
        "capacity.csv",
        "test_predictions.csv",
        "confusion_matrix.png",
        "results.png",
        "tuning_results.csv",
    ]:
        assert (cfg.reports_dir / name).exists(), name
    assert not (cfg.reports_dir / "validation_decision_search.csv").exists()
    assert (cfg.models_dir / "model.joblib").exists()

    # The headline is the candidate with the lowest simulated-validation cost,
    # never a calibration searched on the official validation set.
    table = pd.read_csv(cfg.reports_dir / "model_selection.csv")
    eligible = table[
        (table["calibration"] != "searched_on_validation")
        & (table["learner"] == "gradient_boosting")
    ]
    assert m["selection"]["sim_cost"] == eligible["sim_cost"].min()
    assert m["final_choice"]["calibration"] != "searched_on_validation"
    assert m["final_choice"]["validation_cost"] == m["validation"]["total_cost"]
    assert m["calibration"]["searched_correction"]["kind"] == "searched_on_validation"
    assert len(m["tuning"]["results"]) == 2
    rules = ["unconstrained", "capacity_2", "capacity_5", "capacity_10", "capacity_20"]
    assert m["capacity"]["rules"] == rules
    assert m["capacity"]["recommended"] in rules[1:] + [None]

    preds = pd.read_csv(cfg.reports_dir / "test_predictions.csv")
    assert list(preds.columns) == [
        "vehicle_id",
        "true_class",
        "predicted_class",
        *[f"p_class_{k}" for k in range(5)],
        "at_risk_probability",
        "model_score",
        *[f"predicted_class_{r}" for r in rules[1:]],
    ]
    for r in rules[1:]:
        flagged = (preds[f"predicted_class_{r}"] > 0).sum()
        assert flagged == round(int(r.split("_")[1]) / 100 * 40)
        assert (
            total_cost(preds["true_class"], preds[f"predicted_class_{r}"])
            == m["test"]["by_rule"]["gradient_boosting"][r]["total_cost"]
        )
    assert len(preds) == 40
    assert np.allclose(preds[[f"p_class_{k}" for k in range(5)]].sum(axis=1), 1, atol=1e-4)
    final = m["test"]["final"]
    assert total_cost(preds["true_class"], preds["predicted_class"]) == final["total_cost"]
    assert final["always_class_0_cost"] == total_cost(preds["true_class"], np.zeros(40, int))
    with open(cfg.reports_dir / "metrics.json") as f:
        saved = json.load(f)
    assert saved["test"]["final"]["total_cost"] == final["total_cost"]
    assert saved["bootstrap_test"]["samples"] == 100
    assert saved["dataset"]["labels"]["dropped_censored_unknown_class"] > 0

    # SHAP: every test vehicle explained, with readable names and exact sums.
    names = pd.read_csv(cfg.reports_dir / "feature_names.csv")
    assert len(names) == saved["dataset"]["n_features"]
    assert not (names["feature"] == names["readable_name"]).any()
    top = pd.read_csv(cfg.reports_dir / "shap_top_features.csv")
    assert top["vehicle_id"].nunique() == 40 == saved["shap"]["explained_rows"]
    assert (top.groupby(["vehicle_id", "direction"]).size() <= 5).all()
    assert set(top["readable_name"]) <= set(names["readable_name"])
    assert len(pd.read_csv(cfg.reports_dir / "shap_global.csv")) == 20
    assert (cfg.reports_dir / "shap_summary.png").exists()
    flagged = pd.read_csv(cfg.reports_dir / "shap_global_flagged.csv")
    rule = m["capacity"]["recommended"] or "unconstrained"
    column = "predicted_class" if rule == "unconstrained" else f"predicted_class_{rule}"
    assert saved["shap"]["flagged_rows"] == int((preds[column] > 0).sum())
    assert 0 < len(flagged) <= 20 and flagged["mean_abs_shap"].is_monotonic_decreasing
    assert saved["shap"]["additivity_max_error"] < 1e-6

    # README sections and comparison.csv are generated from the saved metrics.
    from mfp.report import write_summary

    readme = tmp_path / "README.md"
    sections = "".join(
        f"<!-- {n}:START -->\nPLACEHOLDER\n<!-- {n}:END -->\n"
        for n in ("SCANIA", "SCANIA_SHAP", "COMPARISON")
    )
    readme.write_text(f"# Title\n{sections}\nFooter\n")
    write_summary(cfg.reports_dir.parent, readme)
    text = readme.read_text()
    assert "PLACEHOLDER" not in text and text.endswith("Footer\n")
    assert f"| Gradient boosting | unlimited | {final['total_cost']:,.0f} " in text
    comparison = pd.read_csv(cfg.reports_dir.parent / "comparison.csv")
    models = ["gradient_boosting", "logistic_regression", "searched_correction"]
    rec = m["capacity"]["recommended"]
    if rec:
        # Recommended capacity first (with always class 0), then unlimited.
        label = f"Scania ({rec.split('_')[1]}% capacity)"
        assert comparison["dataset"].tolist() == [label] * 4 + ["Scania (unlimited capacity)"] * 3
        assert comparison["model"].tolist() == [*models, "always_class_0", *models]
        assert (
            comparison.loc[0, "official_cost"]
            == (m["test"]["by_rule"]["gradient_boosting"][rec]["total_cost"])
        )
        assert comparison.loc[4, "official_cost"] == final["total_cost"]
    else:
        assert comparison["model"].tolist() == [*models, "always_class_0"]
    assert comparison.loc[3, "official_cost"] == final["always_class_0_cost"]
    assert comparison["lifecycle_cost"].isna().all()


def test_choices_do_not_depend_on_test_data(tmp_path, scania_dir):
    """Choices do not depend on test data.

    Model, calibration, capacity and the searched-correction weights are chosen on training and
    validation data only: scrambling the test readouts and labels must not change them.
    """
    from mfp.scania.pipeline import run

    a = run(_cfg(tmp_path / "a", scania_dir), tune=False, verbose=False)

    ro = pd.read_csv(scania_dir / "test_operational_readouts.csv")
    rng = np.random.default_rng(3)
    value_cols = [c for c in ro.columns if c not in ("vehicle_id", "time_step")]
    ro[value_cols] = ro[value_cols] * rng.uniform(0.5, 2.0, ro[value_cols].shape)
    ro.to_csv(scania_dir / "test_operational_readouts.csv", index=False)
    labels = pd.read_csv(scania_dir / "test_labels.csv")
    labels["class_label"] = rng.permutation(labels["class_label"].to_numpy())
    labels.to_csv(scania_dir / "test_labels.csv", index=False)

    b = run(_cfg(tmp_path / "b", scania_dir), tune=False, verbose=False)
    assert a["final_choice"] == b["final_choice"]

    def without_timing(selection):
        return {k: v for k, v in selection.items() if k != "run_time_seconds"}

    assert without_timing(a["selection"]) == without_timing(b["selection"])
    assert a["calibration"] == b["calibration"]
    assert a["simulated"] == b["simulated"]
    assert a["validation_by_model"] == b["validation_by_model"]


def test_max_rows_per_vehicle_caps_training_rows(tmp_path, scania_dir):
    from mfp.scania.pipeline import run

    m = run(_cfg(tmp_path, scania_dir, max_rows_per_vehicle=3), tune=False, verbose=False)
    assert m["shap"]["explained_rows"] == 40  # every test vehicle
    lab = m["dataset"]["labels"]
    assert lab["rows_used_for_training"] <= 3 * lab["labelled_vehicles"]
    assert lab["rows_used_for_training"] < lab["labelled_readouts"]
