"""Dashboard logic for NASA lifecycle policies and engine timelines."""

from dataclasses import replace

import numpy as np
import pandas as pd

from mfp.config import Config
from mfp.dashboard import (
    alarm_sentence,
    available_subsets,
    choose_policy,
    engine_alarm,
    engine_timeline,
    lifecycle_results,
    load_subset,
)
from mfp.lifecycle import EARLY, GOOD, MISSED


def test_choose_policy_rechooses_from_sweep():
    sweep = pd.DataFrame(
        {
            "threshold": [0.2, 0.2, 0.5, 0.5, 0.8],
            "confirm_cycles": [1, 3, 1, 3, 1],
            "good_catches": [10, 12, 12, 11, 8],
            "early_alarms": [8, 4, 3, 1, 0],
            "missed": [0, 2, 3, 6, 10],
            "median_lead_time": [60.0, 45.0, 40.0, 30.0, 12.0],
        }
    )
    # 10:1 costs -> 8, 24, 33, 61, 100: ring early and often.
    t, k, priced = choose_policy(sweep, 10)
    assert (t, k) == (0.2, 1)
    assert priced["cost"].tolist() == [8, 24, 33, 61, 100]
    # 1:1 costs -> 8, 6, 6, 7, 10: tie at 6, lower threshold wins.
    t, k, _ = choose_policy(sweep, 1)
    assert (t, k) == (0.2, 3)


def _engine(probs, unit=1, set_name="lifecycle_test"):
    n = len(probs)
    return pd.DataFrame(
        {
            "set": set_name,
            "unit": unit,
            "cycle": np.arange(1, n + 1),
            "failure_probability": probs,
            "rul": np.arange(n - 1, -1, -1),
        }
    )


def test_engine_alarm_finds_alarm_cycle_and_explains_it():
    probs = np.full(100, 0.1)
    probs[60] = 0.95  # one-cycle spike at cycle 61 (RUL 39)
    probs[70:] = 0.8  # high from cycle 71 (RUL 29) to failure
    tl = _engine(probs)

    one = engine_alarm(tl, 0.5, 1, 10, 50)
    assert one["alarm_cycle"] == 61 and one["lead_time"] == 39 and one["outcome"] == GOOD
    assert one["alarm_probability"] == 0.95
    assert one["failure_cycle"] == 100
    assert one["sentence"] == "The alarm rang 39 cycles before failure: good catch."

    two = engine_alarm(tl, 0.5, 2, 10, 50)
    assert two["alarm_cycle"] == 72 and two["lead_time"] == 28

    never = engine_alarm(tl, 0.99, 1, 10, 50)
    assert never["alarm_cycle"] is None and never["outcome"] == MISSED
    assert "never rang" in never["sentence"]


def test_alarm_sentences_for_early_and_late():
    early = {"lead_time": 80, "outcome": EARLY}
    late = {"lead_time": 1, "outcome": MISSED}
    assert alarm_sentence(early, 10, 50).startswith(
        "The alarm rang 80 cycles before failure: early alarm"
    )
    assert alarm_sentence(late, 10, 50).startswith(
        "The alarm rang only 1 cycle before failure: missed"
    )


def test_lifecycle_results_use_lifecycle_test_engines_only():
    a = _engine(np.r_[np.full(70, 0.1), np.full(30, 0.9)], unit=1)  # rings at RUL 29
    b = _engine(np.full(100, 0.9), unit=2)  # rings at RUL 99
    nasa = _engine(np.full(100, 0.9), unit=3, set_name="nasa_test")  # must be ignored
    tl = pd.concat([a, b, nasa], ignore_index=True)
    res = lifecycle_results(tl, 0.5, 1, 10, 50, cost_ratio=10)
    assert res["engines"] == 2
    assert (res["good_catches"], res["early_alarms"], res["missed"]) == (1, 1, 0)
    assert res["cost"] == 1
    assert len(engine_timeline(tl, 3)) == 0
    assert len(engine_timeline(tl, 3, set_name="nasa_test")) == 100


def test_dashboard_reproduces_saved_pipeline_results(tmp_path, cmapss_dir):
    from mfp.pipeline import run_all

    base = replace(
        Config(),
        data_dir=cmapss_dir,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
    )
    m = run_all(base, ["FD001"], tune=False)["FD001"]

    assert available_subsets(base.reports_dir) == ["FD001"]
    data = load_subset(base.reports_dir, "FD001")
    s = m["settings"]
    t, k, _ = choose_policy(data["sweep"], s["cost_ratio_main"])
    assert (t, k) == (m["alert_threshold"], m["confirm_cycles"])

    res = lifecycle_results(
        data["timelines"], t, k, s["alert_min_lead"], s["alert_max_lead"], s["cost_ratio_main"]
    )
    saved = m["lifecycle_test"]["chosen"]
    for key in ["good_catches", "early_alarms", "missed", "cost"]:
        assert res[key] == saved[key], key
