"""Display helpers for the dashboard: headlines, 2x2 matrix, plain names and SHAP bars."""

import numpy as np
import pandas as pd

from mfp.dashboard import (
    at_risk_matrix,
    nasa_headline,
    plain_feature_name,
    scania_headline,
    shap_bars,
)


def test_scania_headline_rounds_recall_and_saving():
    s = {"recall": 0.394, "saving": 11633.0, "always_class_0_cost": 56100.0}
    assert scania_headline(s) == "Catches 4 in 10 failing trucks, saves 21% on cost."
    s = {"recall": 0.02, "saving": -561.0, "always_class_0_cost": 56100.0}
    assert scania_headline(s) == "Catches fewer than 1 in 10 failing trucks, costs 1% more."


def test_at_risk_matrix_folds_classes_1_to_4():
    cm = np.zeros((5, 5), dtype=int)
    cm[0, 0], cm[0, 4], cm[2, 0], cm[3, 3], cm[4, 1] = 90, 5, 2, 3, 4
    out = at_risk_matrix(cm).set_index("Outcome")["Vehicles"]
    assert out.to_dict() == {
        "Caught": 7,
        "Missed": 2,
        "False alarm": 5,
        "Correctly left alone": 90,
    }
    assert out.sum() == cm.sum()


def test_plain_feature_names():
    assert plain_feature_name("Time step") == "Truck age (time step)"
    assert plain_feature_name("Histogram 459, bin 13: share of total") == (
        "Sensor 459, range 13: share of readings"
    )
    assert plain_feature_name("Histogram 459: total count") == "Sensor 459: total readings"
    assert plain_feature_name(
        "Histogram 158, bin 2: change in share of total (last 5 readouts)"
    ) == ("Sensor 158, range 2: recent change in share")
    assert plain_feature_name("Counter 666: value") == "Counter 666"
    assert plain_feature_name("Counter 309: rate of change (last 5 readouts)") == (
        "Counter 309: recent rate of change"
    )
    assert plain_feature_name("Specification 7 is Cat6") == "Spec 7 = Cat6"
    assert plain_feature_name("Sensor 11: average (last 10 cycles)") == (
        "Sensor 11: average (last 10 cycles)"
    )


def test_shap_bars_sorts_by_size_and_labels_direction():
    top = pd.DataFrame(
        {
            "vehicle_id": [1, 1, 1, 2],
            "direction": ["up", "up", "down", "up"],
            "rank": [1, 2, 1, 1],
            "readable_name": ["Time step", "Counter 1: value", "Histogram 2: total count", "x"],
            "shap_value": [0.5, 2.0, -1.0, 9.0],
            "feature_value": [10.0, 3.0, 7.0, 1.0],
        }
    )
    out = shap_bars(top, "vehicle_id", 1)
    assert out["Feature"].tolist() == [
        "Counter 1",
        "Sensor 2: total readings",
        "Truck age (time step)",
    ]
    assert out["Effect"].tolist() == ["Raises risk", "Lowers risk", "Raises risk"]
    assert shap_bars(None, "vehicle_id", 1).empty


def test_nasa_headline():
    row = {"Good catches": 19, "Early alarms": 1, "Missed": 0}
    assert (
        nasa_headline(row, 20)
        == "Warns in time for 19 of 20 engines, with 1 early alarm and 0 missed."
    )
