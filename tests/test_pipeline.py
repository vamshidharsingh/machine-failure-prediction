"""End-to-end smoke test on synthetic data in the real file format."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from mfp.config import Config
from mfp.pipeline import run
from mfp.report import END, START, update_readme


def _cfg(tmp_path, data_dir):
    return replace(
        Config(),
        data_dir=data_dir,
        reports_dir=tmp_path / "out" / "reports",
        models_dir=tmp_path / "out" / "models",
        tuning_grid={"learning_rate": [0.1], "max_leaf_nodes": [15], "max_iter": [50, 100]},
    )


def test_pipeline_runs_and_saves_reports(tmp_path, cmapss_dir):
    cfg = _cfg(tmp_path, cmapss_dir)
    m = run(cfg, tune=True, update_docs=False)

    for name in [
        "metrics.json",
        "validation_lifecycle_sweep.csv",
        "test_predictions.csv",
        "results.png",
        "tuning_results.csv",
    ]:
        assert (cfg.reports_dir / name).exists(), name
    # Only the lifecycle sweep is saved, not a per-cycle one.
    assert not (cfg.reports_dir / "validation_sweep.csv").exists()
    assert (cfg.models_dir / "model.joblib").exists()

    assert m["model_selection"]["chosen_model"] in {"logistic_regression", "gradient_boosting"}
    assert 0 < m["alert_threshold"] < 1
    # Training engines are split three ways (fit / validation / lifecycle
    # test); together they account for all 30 training engines.
    d = m["dataset"]
    assert d["fit_engines"] + d["validation_engines"] + d["lifecycle_test_engines"] == 30
    assert len(m["tuning"]["results"]) == 2
    # The saved file matches what run() returned.
    with open(cfg.reports_dir / "metrics.json") as f:
        assert json.load(f)["alert_threshold"] == m["alert_threshold"]

    preds = pd.read_csv(cfg.reports_dir / "test_predictions.csv")
    sweep = pd.read_csv(cfg.reports_dir / "validation_lifecycle_sweep.csv")
    assert len(preds) == 12
    assert sweep["threshold"].between(0, 1).all()


def test_pipeline_saves_lifecycle_outputs(tmp_path, cmapss_dir):
    cfg = _cfg(tmp_path, cmapss_dir)
    m = run(cfg, tune=False, update_docs=False)

    life_sweep = pd.read_csv(cfg.reports_dir / "validation_lifecycle_sweep.csv")
    assert list(life_sweep.columns) == [
        "threshold",
        "confirm_cycles",
        "good_catches",
        "early_alarms",
        "missed",
        "median_lead_time",
    ]
    assert sorted(life_sweep["confirm_cycles"].unique()) == [1, 2, 3]
    # Every validation engine gets exactly one outcome at every policy.
    n_val = m["dataset"]["validation_engines"]
    totals = life_sweep[["good_catches", "early_alarms", "missed"]].sum(axis=1)
    assert (totals == n_val).all()

    timelines = pd.read_csv(cfg.reports_dir / "engine_timelines.csv")
    assert list(timelines.columns) == ["set", "unit", "cycle", "failure_probability", "rul"]
    assert set(timelines["set"]) == {"lifecycle_test", "nasa_test"}
    life_units = timelines.loc[timelines["set"] == "lifecycle_test", "unit"].nunique()
    assert life_units == m["dataset"]["lifecycle_test_engines"] == 6
    assert timelines.loc[timelines["set"] == "nasa_test", "unit"].nunique() == 12
    # Lifecycle test engines run to failure.
    last = timelines[timelines["set"] == "lifecycle_test"].groupby("unit")["rul"].min()
    assert (last == 0).all()

    lt = m["lifecycle_test"]
    for block in (lt["chosen"], lt["fixed_0.5"]):
        assert block["good_catches"] + block["early_alarms"] + block["missed"] == 6
    assert (lt["fixed_0.5"]["threshold"], lt["fixed_0.5"]["confirm_cycles"]) == (0.5, 1)
    assert (lt["chosen"]["threshold"], lt["chosen"]["confirm_cycles"]) == (
        m["alert_threshold"],
        m["confirm_cycles"],
    )
    assert lt["cost_saving_vs_0.5"] == lt["fixed_0.5"]["cost"] - lt["chosen"]["cost"]
    assert [r["cost_ratio"] for r in m["lifecycle_sensitivity"]] == [5.0, 10.0, 20.0]


def test_policy_does_not_depend_on_lifecycle_test_engines(tmp_path, cmapss_dir):
    """The alarm policy does not depend on lifecycle test engines.

    The policy is chosen on validation engines only: scrambling the sensor readings of lifecycle
    test engines must not change it.
    """
    from mfp.data import SENSOR_COLS, load_cmapss
    from mfp.split import three_way_split

    a = run(_cfg(tmp_path / "a", cmapss_dir), tune=False, update_docs=False)

    train, _, _ = load_cmapss(cmapss_dir)
    _, _, life = three_way_split(train, 0.2, 0.2, seed=42)
    life_units = set(life["unit"])
    rows = train["unit"].isin(life_units)
    rng = np.random.default_rng(1)
    train.loc[rows, SENSOR_COLS] = train.loc[rows, SENSOR_COLS].to_numpy() + rng.normal(
        0, 5, (rows.sum(), len(SENSOR_COLS))
    )
    fmt = ["%d", "%d"] + ["%.6f"] * (train.shape[1] - 2)
    np.savetxt(cmapss_dir / "train_FD001.txt", train.to_numpy(), fmt=fmt)

    b = run(_cfg(tmp_path / "b", cmapss_dir), tune=False, update_docs=False)
    assert (a["alert_threshold"], a["confirm_cycles"]) == (
        b["alert_threshold"],
        b["confirm_cycles"],
    )
    assert a["model_selection"] == b["model_selection"]
    assert a["validation_lifecycle"] == b["validation_lifecycle"]


def test_pipeline_is_reproducible(tmp_path, cmapss_dir):
    a = run(_cfg(tmp_path / "a", cmapss_dir), tune=False, update_docs=False)
    b = run(_cfg(tmp_path / "b", cmapss_dir), tune=False, update_docs=False)
    assert a["test_last_cycle"] == b["test_last_cycle"]
    assert a["rul"] == b["rul"]


def test_readme_results_come_from_metrics(tmp_path, cmapss_dir):
    m = run(_cfg(tmp_path, cmapss_dir), tune=False, update_docs=False)
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START}\nPLACEHOLDER_TEXT\n{END}\n\nFooter\n")
    assert update_readme(readme, m)
    text = readme.read_text()
    assert "PLACEHOLDER_TEXT" not in text
    assert f"| Threshold | {m['alert_threshold']:.2f}" in text
    assert text.endswith("Footer\n")


def test_multiregime_subset_runs_end_to_end(tmp_path, multiregime_dir):
    from mfp.pipeline import run_all

    base = replace(
        Config(),
        data_dir=multiregime_dir,
        reports_dir=tmp_path / "out" / "reports",
        models_dir=tmp_path / "out" / "models",
        tuning_grid={"learning_rate": [0.1], "max_leaf_nodes": [15], "max_iter": [50]},
    )
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START}\nPLACEHOLDER_TEXT\n{END}\n\nFooter\n")

    results = run_all(base, ["FD002"], tune=True, readme=readme)
    m = results["FD002"]

    out = base.reports_dir / "FD002"
    for name in [
        "metrics.json",
        "validation_lifecycle_sweep.csv",
        "engine_timelines.csv",
        "test_predictions.csv",
        "results.png",
        "tuning_results.csv",
    ]:
        assert (out / name).exists(), name
    assert (base.models_dir / "FD002" / "model.joblib").exists()

    assert m["dataset"]["subset"] == "FD002"
    assert m["dataset"]["n_regimes"] == 6
    # Sensors constant within each regime are dropped after normalisation.
    for s in ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]:
        assert s not in m["dataset"]["sensors_used"]
    # With regimes removed, the wear signal is learnable.
    assert m["model_selection"]["validation_pr_auc"]["logistic_regression"] > 0.5

    with open(base.reports_dir / "summary.json") as f:
        summary = json.load(f)
    assert list(summary["subsets"]) == ["FD002"]
    s = summary["subsets"]["FD002"]
    assert s["cost_saving_vs_0.5"] == m["lifecycle_test"]["cost_saving_vs_0.5"]
    assert s["cost_based_policy_wins"] in {"Yes", "No", "Tie"}

    text = readme.read_text()
    assert "PLACEHOLDER_TEXT" not in text
    assert "| FD002 | 6 |" in text
    assert f"| Threshold | {m['alert_threshold']:.2f}" in text
    assert text.endswith("Footer\n")
