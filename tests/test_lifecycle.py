"""The once-per-engine alarm simulation and policy sweep."""

import numpy as np
import pandas as pd
import pytest

from mfp.lifecycle import (
    EARLY,
    GOOD,
    MISSED,
    SWEEP_COLUMNS,
    best_policy,
    classify,
    engine_outcome,
    first_alarm_index,
    lifecycle_sweep,
    simulate,
    summarize,
)

LIFE = 80  # cycles 1..80, RUL 79..0
CYCLES = np.arange(1, LIFE + 1)
RUL = LIFE - CYCLES


def probs_high_from(rul_start, high=0.9, low=0.1):
    """Probability `high` from the cycle whose RUL is rul_start until failure."""
    return np.where(RUL <= rul_start, high, low)


def outcome(probs, threshold=0.5, confirm=1):
    return engine_outcome(CYCLES, probs, RUL, threshold, confirm, min_lead=10, max_lead=50)


def test_good_catch():
    r = outcome(probs_high_from(30))
    assert r["outcome"] == GOOD
    assert r["lead_time"] == 30
    assert r["alarm_cycle"] == LIFE - 30
    assert r["failure_cycle"] == LIFE  # last cycle, where RUL is 0


def test_early_alarm():
    r = outcome(probs_high_from(70))
    assert r["outcome"] == EARLY
    assert r["lead_time"] == 70


def test_missed_when_alarm_never_rings():
    r = outcome(np.full(LIFE, 0.2))
    assert r["outcome"] == MISSED
    assert r["alarm_cycle"] is None and r["lead_time"] is None
    assert r["too_late"] is False


def test_alarm_that_rings_too_late_is_missed():
    r = outcome(probs_high_from(5))
    assert r["lead_time"] == 5
    assert r["outcome"] == MISSED
    assert r["too_late"] is True


def test_lead_window_is_inclusive():
    assert classify(10, 10, 50) == GOOD
    assert classify(50, 10, 50) == GOOD
    assert classify(9, 10, 50) == MISSED
    assert classify(51, 10, 50) == EARLY
    assert classify(None, 10, 50) == MISSED


def test_confirm_cycles_requires_consecutive_cycles():
    # A one-cycle spike at RUL 70, quiet, then high from RUL 40 to failure.
    p = probs_high_from(40)
    p[RUL == 70] = 0.9
    one = outcome(p, confirm=1)
    two = outcome(p, confirm=2)
    three = outcome(p, confirm=3)
    assert (one["outcome"], one["lead_time"]) == (EARLY, 70)
    # The spike is ignored; the alarm rings on the 2nd / 3rd high cycle in a row.
    assert (two["outcome"], two["lead_time"]) == (GOOD, 39)
    assert (three["outcome"], three["lead_time"]) == (GOOD, 38)

    # Alternating high/low never gives two in a row.
    alternating = np.where(CYCLES % 2 == 0, 0.9, 0.1)
    assert first_alarm_index(alternating, 0.5, 2) is None
    assert first_alarm_index(alternating, 0.5, 1) == 1


def test_threshold_is_at_or_above():
    p = np.full(LIFE, 0.1)
    p[RUL == 20] = 0.5
    assert outcome(p, threshold=0.5)["lead_time"] == 20
    assert outcome(p, threshold=0.51)["lead_time"] is None


def _timelines(prob_list):
    frames = []
    for unit, probs in enumerate(prob_list, start=1):
        frames.append(
            pd.DataFrame({"unit": unit, "cycle": CYCLES, "failure_probability": probs, "rul": RUL})
        )
    return pd.concat(frames, ignore_index=True)


def test_simulate_and_summarize_count_one_outcome_per_engine():
    tl = _timelines(
        [probs_high_from(30), probs_high_from(70), np.full(LIFE, 0.2), probs_high_from(5)]
    )
    out = simulate(tl, 0.5, 1, 10, 50)
    assert out["outcome"].tolist() == [GOOD, EARLY, MISSED, MISSED]
    s = summarize(out, cost_miss=10, cost_false=1)
    assert s["engines"] == 4
    assert (s["good_catches"], s["early_alarms"], s["missed"]) == (1, 1, 2)
    assert (s["missed_never_rang"], s["missed_too_late"]) == (1, 1)
    assert s["cost"] == 10 * 2 + 1 * 1
    assert s["median_lead_time"] == 30  # leads 30, 70, 5
    assert s["mean_lead_time_good_catches"] == 30


def test_sweep_matches_engine_by_engine_simulation():
    rng = np.random.default_rng(0)
    probs = [np.clip(np.linspace(0, 1, LIFE) + rng.normal(0, 0.15, LIFE), 0, 1) for _ in range(12)]
    tl = _timelines(probs)
    sweep = lifecycle_sweep(tl, 10, 50, confirm_options=(1, 2, 3))
    assert list(sweep.columns) == SWEEP_COLUMNS
    assert len(sweep) == 3 * 99
    for t in [0.05, 0.3, 0.5, 0.77, 0.95]:
        for k in (1, 2, 3):
            row = sweep[(sweep["threshold"] == t) & (sweep["confirm_cycles"] == k)].iloc[0]
            s = summarize(simulate(tl, t, k, 10, 50), 10, 1)
            assert row["good_catches"] == s["good_catches"]
            assert row["early_alarms"] == s["early_alarms"]
            assert row["missed"] == s["missed"]
            if np.isnan(s["median_lead_time"]):
                assert np.isnan(row["median_lead_time"])
            else:
                assert row["median_lead_time"] == pytest.approx(s["median_lead_time"])


def test_best_policy_minimises_cost_and_breaks_ties():
    sweep = pd.DataFrame(
        {
            "threshold": [0.2, 0.2, 0.4, 0.4, 0.6],
            "confirm_cycles": [1, 2, 1, 2, 1],
            "good_catches": [0] * 5,
            "early_alarms": [12, 2, 2, 2, 0],
            "missed": [0, 1, 1, 1, 3],
            "median_lead_time": [0.0] * 5,
        }
    )
    # 10:1 costs -> 12, 12, 12, 12, 30: tie, lower threshold then fewer confirm cycles.
    assert best_policy(sweep, 10, 1) == (0.2, 1)
    # 1:1 costs -> 12, 3, 3, 3, 3: lowest cost 3, lowest threshold 0.2, confirm 2.
    assert best_policy(sweep, 1, 1) == (0.2, 2)
