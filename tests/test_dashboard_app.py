"""Dashboard logic and the Streamlit app run from reports only.

Dashboard logic for Scania, warning time, SHAP and the comparison, and the real Streamlit app run
against a folder that holds only reports/ (no data/, no models/).
"""

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mfp.dashboard import (
    SCANIA,
    available_datasets,
    best_cells,
    comparison_display,
    confusion_long,
    explanation,
    lowest_cost_rows,
    nasa_takeaway,
    policy_comparison,
    scania_takeaway,
    scania_vehicle_table,
    sweep_for_min_lead,
    warning_time_options,
)

ROOT = Path(__file__).resolve().parent.parent


def test_warning_time_sweeps_are_selected_by_min_lead():
    sweeps = pd.DataFrame(
        {
            "alert_min_lead": [10, 10, 20],
            "threshold": [0.1, 0.2, 0.1],
            "confirm_cycles": [1, 1, 1],
            "good_catches": [3, 2, 1],
            "early_alarms": [0, 1, 2],
            "missed": [0, 0, 0],
            "median_lead_time": [20.0, 30.0, 40.0],
        }
    )
    assert warning_time_options(sweeps) == [10, 20]
    assert warning_time_options(None) == []
    s20 = sweep_for_min_lead(sweeps, 20)
    assert "alert_min_lead" not in s20 and len(s20) == 1
    with pytest.raises(ValueError):
        sweep_for_min_lead(sweeps, 30)


def _engine(unit, ring_at_rul, life=100):
    probs = np.full(life, 0.1)
    if ring_at_rul is not None:
        probs[life - 1 - ring_at_rul :] = 0.9
    return pd.DataFrame(
        {
            "set": "lifecycle_test",
            "unit": unit,
            "cycle": np.arange(1, life + 1),
            "failure_probability": probs,
            "rul": np.arange(life - 1, -1, -1),
        }
    )


def test_policy_comparison_has_three_policies():
    tl = pd.concat([_engine(1, 30), _engine(2, 80), _engine(3, None)])
    table = policy_comparison(tl, 0.3, 2, 10, 50, cost_ratio=10)
    assert table["Policy"].tolist() == [
        "Cost-based policy",
        "Fixed 0.5",
        "Fixed 0.5, same confirm cycles",
    ]
    assert table.loc[2, "Threshold"] == 0.5 and table.loc[2, "Confirm cycles"] == 2
    row = table.loc[0]
    assert (row["Good catches"], row["Early alarms"], row["Missed"]) == (1, 1, 1)
    assert row["Cost"] == 11 and row["Precision"] == 0.5


def test_explanation_in_plain_words():
    top = pd.DataFrame(
        {
            "vehicle_id": [7, 7, 7],
            "direction": ["up", "up", "down"],
            "rank": [1, 2, 1],
            "feature": ["459_total", "171_rate", "Spec_1=Cat2"],
            "readable_name": [
                "Histogram 459: total count",
                "Counter 171: rate of change",
                "Specification 1 is Cat2",
            ],
            "shap_value": [0.25, 0.01, -0.05],
            "feature_value": [123456.0, np.nan, 1.0],
        }
    )
    e = explanation(top, "vehicle_id", 7, "probability")
    assert e["explained"]
    assert e["up"] == [
        "Histogram 459: total count = 123,456 raised the risk by 0.250 (probability)",
        "Counter 171: rate of change = missing raised the risk by 0.010 (probability)",
    ]
    assert e["down"] == ["Specification 1 is Cat2 = 1 lowered the risk by 0.050 (probability)"]
    assert explanation(top, "vehicle_id", 8, "probability")["explained"] is False
    assert explanation(None, "unit", 1, "log-odds") == {"explained": False, "up": [], "down": []}


def test_scania_table_and_confusion_matrix():
    preds = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3],
            "true_class": [0, 4, 0],
            "predicted_class": [0, 4, 4],
            "at_risk_probability": [0.1, 0.9, 0.5],
        }
    )
    top = pd.DataFrame({"vehicle_id": [2]})
    table = scania_vehicle_table(preds, top)
    assert table["Vehicle"].tolist() == [2, 3, 1]
    assert table["Explained"].tolist() == [True, False, False]
    cm = confusion_long([[3, 0, 0, 0, 2], [1, 0, 0, 0, 0], [0] * 5, [0] * 5, [0, 0, 0, 0, 1]])
    assert len(cm) == 25
    assert cm.loc[(cm["True class"] == 0) & (cm["Predicted class"] == 4), "Cost"].item() == 20
    assert cm.loc[(cm["True class"] == 1) & (cm["Predicted class"] == 0), "Cost"].item() == 200


def test_capacity_view_matches_saved_metrics(tmp_path):
    from mfp.dashboard import capacity_label, capacity_options, prediction_column, scania_summary

    preds = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3, 4],
            "true_class": [0, 4, 0, 1],
            "predicted_class": [4, 4, 4, 1],
            "predicted_class_capacity_50": [0, 4, 0, 1],
        }
    )
    metrics = {
        "final_choice": {
            "model": "gradient_boosting",
            "decision": "lowest_expected_cost",
            "prior_correction": True,
        },
        "capacity": {"rules": ["unconstrained", "capacity_50"], "recommended": "capacity_50"},
        "bootstrap_test": {
            "by_rule": {
                "capacity_50": {
                    "models": {"gradient_boosting": {"saving_vs_always_class_0_95": [1.0, 2.0]}}
                }
            }
        },
    }
    assert capacity_options(metrics) == ["capacity_50", "unconstrained"]
    assert capacity_label("capacity_50") == "50%" and capacity_label("unconstrained") == "unlimited"
    assert prediction_column("unconstrained") == "predicted_class"
    s = scania_summary(metrics, preds, "capacity_50")
    assert (s["test_cost"], s["false_alarms"], s["recall"]) == (0, 0, 1.0)
    assert s["saving"] == 700 and s["saving_95"] == [1.0, 2.0]
    u = scania_summary(metrics, preds, "unconstrained")
    assert (u["test_cost"], u["false_alarms"]) == (20, 2)  # two healthy vehicles flagged as 4
    assert u["saving_95"] is None


def test_default_vehicle_and_risk_score_wording():
    from mfp.dashboard import default_vehicle

    table = pd.DataFrame(
        {"Vehicle": [5, 6, 7, 8], "True class": [0, 0, 3, 2], "Predicted class": [4, 4, 4, 0]}
    )
    assert default_vehicle(table) == 7  # first correct alarm, not the false alarms
    assert default_vehicle(table.iloc[:2]) == 5  # no correct alarm: the riskiest vehicle
    top = pd.DataFrame(
        {
            "vehicle_id": [7],
            "direction": ["up"],
            "rank": [1],
            "feature": ["t"],
            "readable_name": ["Time step"],
            "shap_value": [1.25],
            "feature_value": [120.0],
        }
    )
    assert explanation(top, "vehicle_id", 7)["up"] == [
        "Time step = 120 raised the risk score by 1.250"
    ]


def test_comparison_display_blanks_missing_cells():
    df = pd.DataFrame(
        {
            "Dataset": ["Scania", "FD001"],
            "Model": ["Always class 0", "Gradient boosting"],
            "PR-AUC": [np.nan, 0.93681],
            "False alarms": [0, 2359],
            "Official cost (Scania)": [56100.0, np.nan],
        }
    )
    out = comparison_display(df)
    assert out["PR-AUC"].tolist() == ["", "0.937"]
    assert out["False alarms"].tolist() == ["0", "2,359"]
    assert out["Official cost (Scania)"].tolist() == ["56,100", ""]


def test_takeaways_state_the_result_plainly():
    s = {
        "model": "Gradient boosting",
        "saving": 13506.0,
        "always_class_0_cost": 56100.0,
        "saving_95": [6169.0, 21329.0],
        "alarms": 2454,
        "true_at_risk": 142,
    }
    text = scania_takeaway(s)
    assert text.startswith("Gradient boosting with unlimited workshop capacity saves 13,506 (24%)")
    assert scania_takeaway({**s, "capacity": "20%"}).startswith(
        "Gradient boosting at 20% workshop capacity saves"
    )
    assert "(6,169 to 21,329) is above zero" in text and "2,454 alarms for 142 vehicles" in text
    assert "includes zero" in scania_takeaway({**s, "saving_95": [-100.0, 900.0]})
    # Accurate alarms are not described as "flagging many vehicles".
    assert scania_takeaway({**s, "alarms": 150}).endswith(
        "It raises 150 alarms for 142 vehicles really at risk."
    )

    def block(cost):
        return {"chosen": {"cost": cost}}

    m = {
        "lifecycle_test": {
            "chosen": {"cost": 1.0, "engines": 20},
            "fixed_0.5": {"cost": 1.0},
            "fixed_0.5_same_confirm": {"cost": 0.0},
        },
        "baseline_logistic_regression": {"lifecycle_test": block(0.0)},
        "rul": {"rmse_model": 18.8, "rmse_ridge": 22.4},
    }
    text = nasa_takeaway(m)
    assert "ties with a fixed 0.5 threshold (cost 1 vs 1)" in text
    assert "same confirm cycles does better (0)" in text and "logistic regression" in text
    m["lifecycle_test"]["fixed_0.5"]["cost"] = 6.0
    m["lifecycle_test"]["fixed_0.5_same_confirm"]["cost"] = 4.0
    m["baseline_logistic_regression"]["lifecycle_test"] = block(4.0)
    text = nasa_takeaway(m)
    assert "beats a fixed 0.5 threshold (cost 1 vs 6)" in text and "logistic" not in text


def test_best_cells_and_lowest_cost_rows():
    df = pd.DataFrame(
        {
            "Dataset": ["A", "A", "B", "B"],
            "Model": ["x", "y", "x", "y"],
            "PR-AUC": [0.9, 0.8, np.nan, 0.5],
            "Lifecycle cost (NASA)": [3, 3, 1, 2],
        }
    )
    mask = best_cells(df)
    assert mask["PR-AUC"].tolist() == [True, False, False, True]  # higher is better; NaN never
    assert mask["Lifecycle cost (NASA)"].tolist() == [True, True, True, False]  # lower; ties all
    assert not mask["Model"].any()
    assert lowest_cost_rows([1.0, 1.0, 0.0]) == [2]
    assert lowest_cost_rows([3, 1, 1]) == [1, 2]


@pytest.fixture(scope="module")
def reports_only_app(tmp_path_factory):
    """A project folder with only streamlit_app.py and reports/ in it."""
    from conftest import write_cmapss, write_scania

    from mfp.config import Config, ScaniaConfig
    from mfp.pipeline import run_all
    from mfp.report import write_summary
    from mfp.scania.pipeline import run as run_scania

    tmp = tmp_path_factory.mktemp("dash")
    app, scratch = tmp / "app", tmp / "scratch"
    reports = app / "reports"
    scania_cfg = replace(
        ScaniaConfig(),
        data_dir=write_scania(scratch / "scania_data"),
        reports_dir=reports / "scania",
        models_dir=scratch / "models" / "scania",
        bootstrap_samples=50,
        selection_folds=3,
        boosting_params={"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 50},
        prior_weight_grid=(0.5, 1.0, 2.0),
    )
    run_scania(scania_cfg, tune=False, verbose=False)
    (scratch / "nasa_data").mkdir()
    nasa_cfg = replace(
        Config(),
        data_dir=write_cmapss(scratch / "nasa_data"),
        reports_dir=reports,
        models_dir=scratch / "models",
    )
    run_all(nasa_cfg, ["FD001"], tune=False)
    write_summary(reports)
    shutil.rmtree(scratch)  # no raw data and no model files anywhere near the app
    shutil.copy(ROOT / "streamlit_app.py", app / "streamlit_app.py")
    assert sorted(p.name for p in app.iterdir()) == ["reports", "streamlit_app.py"]
    return app


def test_dashboard_runs_from_reports_only(reports_only_app):
    from streamlit.testing.v1 import AppTest

    assert available_datasets(reports_only_app / "reports") == [SCANIA, "FD001"]
    at = AppTest.from_file(str(reports_only_app / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    radio = at.radio[0]
    assert radio.options == [SCANIA, "FD001"] and radio.value == SCANIA
    assert [t.label for t in at.tabs][:2] == ["Scania results", "All test vehicles"]
    assert at.tabs[-1].label == "Model comparison"
    assert any("Why this vehicle" in s.value for s in at.subheader)

    import json

    from mfp.dashboard import recommended_capacity, scania_summary

    reports = reports_only_app / "reports" / "scania"
    with open(reports / "metrics.json") as fh:
        metrics = json.load(fh)
    preds = pd.read_csv(reports / "test_predictions.csv")

    # Workshop capacity: defaults to the recommended capacity, and every view
    # (takeaway, confusion matrix) follows the selected capacity.
    capacity = at.radio[1]
    assert capacity.value == recommended_capacity(metrics)
    assert [o.split(" ")[0] for o in capacity.options] == ["2%", "5%", "10%", "20%", "unlimited"]
    assert sum("(recommended)" in o for o in capacity.options) == 1
    boxes = [b.value for b in list(at.success) + list(at.info)]
    assert boxes[0] == scania_takeaway(scania_summary(metrics, preds, capacity.value))
    capacity.set_value("capacity_2").run()
    assert not at.exception, at.exception
    boxes = [b.value for b in list(at.success) + list(at.info)]
    assert boxes[0] == scania_takeaway(scania_summary(metrics, preds, "capacity_2"))
    assert any("2% capacity" in s.value for s in at.subheader)
    # The SHAP panel offers every test vehicle and opens on the riskiest
    # vehicle that is truly at risk and was flagged at the selected capacity.
    vehicle = next(s for s in at.selectbox if s.label == "Test vehicle (riskiest first)")
    assert len(vehicle.options) == len(preds)
    opened = preds.set_index("vehicle_id").loc[vehicle.value]
    hits = preds[(preds["true_class"] > 0) & (preds["predicted_class_capacity_2"] > 0)]
    if not hits.empty:
        assert opened["true_class"] > 0 and opened["predicted_class_capacity_2"] > 0
    assert any("risk score (log-odds)" in c.value for c in at.caption)

    at.radio[0].set_value("FD001").run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == [
        "FD001 results",
        "Lifecycle alarm",
        "Engine timeline",
        "NASA test engines and SHAP",
        "Model comparison",
    ]
    assert at.info[0].value.startswith("On the 6 lifecycle test engines")
    warning = next(s for s in at.selectbox if s.label == "Required warning time (cycles)")
    assert warning.options == ["10", "20", "30"] and warning.value == 10
    warning.set_value(30).run()
    assert not at.exception, at.exception
    assert any("at least **30** cycles of warning" in m.value for m in at.markdown)
    assert any(s.label == "NASA test engine (riskiest first)" for s in at.selectbox)
    comparison = [d.value for d in at.dataframe if "Dataset" in d.value.columns]
    datasets = set(comparison[0]["Dataset"]) if comparison else set()
    assert "FD001" in datasets and len(datasets) >= 2
    assert all(d == "FD001" or d.startswith("Scania") for d in datasets)
