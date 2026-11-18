"""NASA reporting: baselines, metrics, policies and warning time.

NASA reporting: gradient boosting with a logistic baseline, precision and false alarms next to
recall, the NASA scoring function, the three lifecycle policies and the required warning time.
"""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import write_cmapss

from mfp.config import Config
from mfp.lifecycle import best_policy, simulate, summarize
from mfp.models import nasa_score


def test_nasa_score_on_hand_calculated_values():
    # d = predicted - true: -13 (early) -> e^1 - 1; 0 -> 0; +10 (late) -> e^1 - 1
    assert nasa_score([50, 50, 50], [37, 50, 60]) == pytest.approx(2 * (math.e - 1))
    # Same size of error, late costs more than early.
    early = nasa_score([100], [80])  # d = -20 -> e^(20/13) - 1 = 3.6577
    late = nasa_score([100], [120])  # d = +20 -> e^2 - 1 = 6.3891
    assert early == pytest.approx(math.exp(20 / 13) - 1)
    assert late == pytest.approx(math.exp(2) - 1)
    assert late > early
    assert nasa_score([10, 20], [10, 20]) == 0


def test_lifecycle_precision_counts_alarms_that_rang():
    life = 100
    cycles = np.arange(1, life + 1)
    rul = life - cycles

    def engine(unit, start):
        p = np.full(life, 0.1)
        if start is not None:
            p[life - start - 1 :] = 0.9  # rings with `start` cycles left
        return pd.DataFrame({"unit": unit, "cycle": cycles, "failure_probability": p, "rul": rul})

    # good catch (30 left), early alarm (70), never rang, too late (5)
    tl = pd.concat([engine(1, 30), engine(2, 70), engine(3, None), engine(4, 5)])
    s = summarize(simulate(tl, 0.5, 1, 10, 50), 10, 1)
    assert s["alarms_rang"] == 3
    assert s["precision"] == pytest.approx(1 / 3)
    assert s["early_alarms"] == 1  # the false alarm count


@pytest.fixture(scope="module")
def nasa_run(tmp_path_factory):
    from mfp.pipeline import run

    root = tmp_path_factory.mktemp("nasa")
    data = write_cmapss(root)
    cfg = replace(Config(), data_dir=data, reports_dir=root / "reports", models_dir=root / "models")
    return cfg, run(cfg, tune=False, update_docs=False)


def test_gradient_boosting_is_headline_with_logistic_baseline(nasa_run):
    _, m = nasa_run
    assert m["model_selection"]["chosen_model"] == "gradient_boosting"
    assert "margin_required" not in m["model_selection"]
    assert set(m["model_selection"]["validation_pr_auc"]) == {
        "gradient_boosting",
        "logistic_regression",
    }
    lr = m["baseline_logistic_regression"]
    for key in [
        "alert_threshold",
        "confirm_cycles",
        "lifecycle_test",
        "test_last_cycle",
        "test_all_cycles_pr_auc",
        "validation_lifecycle",
    ]:
        assert key in lr, key


def test_benchmark_reports_precision_recall_and_false_alarms(nasa_run):
    cfg, m = nasa_run
    preds = pd.read_csv(cfg.reports_dir / "test_predictions.csv")
    for block, col in [
        (m["test_last_cycle"], "failure_probability"),
        (m["baseline_logistic_regression"]["test_last_cycle"], "failure_probability_logistic"),
    ]:
        for key, t in [("at_0.5", 0.5), ("at_chosen_threshold", None)]:
            a = block[key]
            t = a["threshold"] if t is None else t
            alert = preds[col] >= t
            pos = preds["at_risk_true"] == 1
            assert a["false_alarms"] == int((alert & ~pos).sum())
            assert a["true_alerts"] == int((alert & pos).sum())
            if alert.any():
                assert a["precision"] == pytest.approx((alert & pos).sum() / alert.sum())
            assert a["recall"] == pytest.approx((alert & pos).sum() / pos.sum())
    assert m["test_last_cycle"]["at_chosen_threshold"]["threshold"] == m["alert_threshold"]


def test_rul_reports_rmse_and_nasa_score_for_both_models(nasa_run):
    cfg, m = nasa_run
    preds = pd.read_csv(cfg.reports_dir / "test_predictions.csv")
    r = m["rul"]
    # Predictions are saved rounded to 0.1 cycle, so compare loosely.
    assert r["nasa_score_model"] == pytest.approx(
        nasa_score(preds["true_rul"], preds["predicted_rul"]), rel=0.02
    )
    assert r["nasa_score_ridge"] == pytest.approx(
        nasa_score(preds["true_rul"], preds["predicted_rul_ridge"]), rel=0.02
    )
    for key in ["rmse_model", "rmse_ridge", "rmse_naive_mean", "nasa_score_naive_mean"]:
        assert r[key] >= 0


def test_third_policy_is_fixed_half_with_chosen_confirm_cycles(nasa_run):
    cfg, m = nasa_run
    lt = m["lifecycle_test"]
    third = lt["fixed_0.5_same_confirm"]
    assert (third["threshold"], third["confirm_cycles"]) == (0.5, m["confirm_cycles"])
    tl = pd.read_csv(cfg.reports_dir / "engine_timelines.csv")
    tl = tl[tl["set"] == "lifecycle_test"]
    s = summarize(simulate(tl, 0.5, m["confirm_cycles"], 10, 50), 10, 1)
    assert third["cost"] == s["cost"]
    assert lt["cost_saving_vs_0.5_same_confirm"] == third["cost"] - lt["chosen"]["cost"]


def test_warning_time_sweeps_saved_and_policies_chosen_on_validation(nasa_run):
    cfg, m = nasa_run
    sweeps = pd.read_csv(cfg.reports_dir / "validation_lifecycle_sweeps.csv")
    assert sorted(sweeps["alert_min_lead"].unique()) == [10, 20, 30]
    assert len(sweeps) == 3 * 3 * 99
    default = pd.read_csv(cfg.reports_dir / "validation_lifecycle_sweep.csv")
    min10 = sweeps[sweeps["alert_min_lead"] == 10].drop(columns="alert_min_lead")
    pd.testing.assert_frame_equal(min10.reset_index(drop=True), default)
    assert [w["alert_min_lead"] for w in m["warning_time"]] == [10, 20, 30]
    for w in m["warning_time"]:
        sweep = sweeps[sweeps["alert_min_lead"] == w["alert_min_lead"]]
        assert best_policy(sweep, 10, 1) == (w["alert_threshold"], w["confirm_cycles"])
        n = m["dataset"]["lifecycle_test_engines"]
        c = w["chosen"]
        assert c["good_catches"] + c["early_alarms"] + c["missed"] == n
    # The default warning time reproduces the main policy.
    assert (m["warning_time"][0]["alert_threshold"], m["warning_time"][0]["confirm_cycles"]) == (
        m["alert_threshold"],
        m["confirm_cycles"],
    )


def test_shap_explains_each_nasa_test_engine_with_readable_names(nasa_run):
    cfg, m = nasa_run
    d = cfg.reports_dir
    names = pd.read_csv(d / "feature_names.csv")
    assert len(names) == m["dataset"]["n_features"]
    assert names["readable_name"].str.startswith(("Sensor", "Cycle")).all()
    glob = pd.read_csv(d / "shap_global.csv")
    assert 0 < len(glob) <= 20 and glob["mean_abs_shap"].is_monotonic_decreasing
    top = pd.read_csv(d / "shap_top_features.csv")
    assert top["unit"].nunique() == m["dataset"]["test_engines"] == m["shap"]["explained_rows"]
    assert (top.groupby(["unit", "direction"]).size() <= 5).all()
    assert set(top["readable_name"]) <= set(names["readable_name"])
    assert (d / "shap_summary.png").exists()
    assert m["shap"]["method"] == "shap.TreeExplainer"
    assert m["shap"]["additivity_max_error"] < 1e-4
