"""Per-cycle alert thresholds and their costs."""

import numpy as np
import pandas as pd

from mfp.dashboard import engine_table, reprice
from mfp.policy import best_threshold, confusion_counts, expected_cost, threshold_sweep


def test_confusion_counts_known_case():
    y = [1, 1, 0, 0, 1]
    s = [0.9, 0.2, 0.7, 0.1, 0.6]
    c = confusion_counts(y, s, 0.5)
    assert c == {"true_alerts": 2, "false_alarms": 1, "missed_failures": 1, "correct_quiet": 1}


def test_expected_cost_formula():
    assert expected_cost(missed=3, false_alarms=7, cost_miss=10, cost_false=1) == 37


def test_higher_miss_cost_lowers_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = np.clip(y * 0.3 + rng.uniform(0, 0.7, 500), 0, 1)
    sweep = threshold_sweep(y, scores)
    assert best_threshold(sweep, 20, 1) <= best_threshold(sweep, 2, 1)


def test_best_threshold_minimises_cost():
    sweep = pd.DataFrame(
        {"threshold": [0.2, 0.4, 0.6], "missed_failures": [0, 1, 5], "false_alarms": [30, 8, 2]}
    )
    # costs at 10:1 -> 30, 18, 52
    assert best_threshold(sweep, 10, 1) == 0.4


def test_ties_go_to_lower_threshold():
    sweep = pd.DataFrame(
        {"threshold": [0.3, 0.5], "missed_failures": [0, 1], "false_alarms": [10, 0]}
    )
    assert best_threshold(sweep, 10, 1) == 0.3


def test_dashboard_reprice_and_outcomes():
    sweep = pd.DataFrame(
        {"threshold": [0.2, 0.4, 0.6], "missed_failures": [0, 1, 5], "false_alarms": [30, 8, 2]}
    )
    priced, t = reprice(sweep, 10)
    assert t == 0.4 and priced["cost"].tolist() == [30, 18, 52]

    preds = pd.DataFrame(
        {"failure_probability": [0.9, 0.5, 0.3, 0.1], "at_risk_true": [1, 0, 1, 0]}
    )
    table = engine_table(preds, 0.4)
    assert table["outcome"].tolist() == [
        "Correct alert",
        "False alarm",
        "Missed failure",
        "Correct, no alert",
    ]
