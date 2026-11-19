"""Display helpers for the dashboard: headlines, 2x2 matrix, plain names and SHAP bars."""

import numpy as np
import pandas as pd

from mfp.dashboard import (
    at_risk_matrix,
    nasa_headline,
    plain_feature_name,
    risk_label,
    scania_headline,
    scania_subline,
    shap_bars,
    vehicle_label,
)


def test_scania_headline_leads_with_value():
    s = {"saving": 11633.0, "always_class_0_cost": 56100.0, "alarms": 1009}
    assert scania_headline(s, 5045) == (
        "Cuts maintenance cost by 21% by inspecting only the riskiest 20% of trucks."
    )
    assert scania_headline({**s, "alarms": 3894}, 5045, limited=False) == (
        "Cuts maintenance cost by 21% by inspecting the 77% of trucks the model flags."
    )
    s = {"saving": -561.0, "always_class_0_cost": 56100.0, "alarms": 10}
    assert scania_headline(s, 5045) == (
        "Costs 1% more than never inspecting at this workshop capacity."
    )


def test_scania_subline_compares_with_flagging_everything():
    best = {"saving": 12564.0, "alarms": 3894, "false_alarms": 3762}
    s = {
        "saving": 11633.0,
        "alarms": 1009,
        "false_alarms": 953,
        "recall": 0.394,
        "true_at_risk": 142,
    }
    assert scania_subline(s, best) == (
        "Keeps 93% of the saving from flagging every suspect truck, with 75% fewer false "
        "alarms. Flags 56 of the 142 trucks that later failed."
    )
    assert scania_subline({**best, "recall": 0.93, "true_at_risk": 142}, best) == (
        "Flags 132 of the 142 trucks that later failed."
    )


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
    assert plain_feature_name("Counter 666: value") == "Counter 666: value"
    assert plain_feature_name("Counter 309: rate of change (last 5 readouts)") == (
        "Counter 309: recent rate of change"
    )
    assert plain_feature_name("Specification 7 is Cat6") == ("Truck spec 7 = Cat6 (anonymised)")
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
        "Counter 1: value",
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


def test_risk_label_never_claims_certainty():
    assert risk_label(1.0) == "99%+"
    assert risk_label(0.99158) == "99%+"
    assert risk_label(0.3125) == "31%"


def test_vehicle_label_uses_words_not_class_numbers():
    assert vehicle_label(11484, 0.3125, 4, 4) == (
        "Vehicle 11484 · risk 31% · failed within 6 steps · flagged"
    )
    assert (
        vehicle_label(2815, 1.0, 0, 0) == "Vehicle 2815 · risk 99%+ · stayed healthy · not flagged"
    )
