"""Engine lifecycle alarm simulation.

In real maintenance an alarm rings once per engine and a mechanic responds.
So instead of scoring every engine-cycle as a separate decision, this module
walks through each engine's life in order and asks: when would the alarm have
rung, and was that useful?

For one engine:

- The alarm rings at the first cycle where the failure probability has been
  at or above the threshold for `confirm_cycles` consecutive cycles.
- Lead time = remaining useful life (RUL) at that cycle.
- Exactly one outcome per engine:
    "good catch"   lead time between min_lead and max_lead (inclusive)
    "early alarm"  lead time above max_lead (a wasted check: the engine is
                   assumed to be serviced early)
    "missed"       the alarm never rings, or rings with lead time below
                   min_lead (too late to act)

Cost = C_miss x missed + C_false x early alarms.

The engines must run to failure, so their RUL is known at every cycle.
"""

import numpy as np
import pandas as pd

GOOD = "good catch"
EARLY = "early alarm"
MISSED = "missed"

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)

SWEEP_COLUMNS = [
    "threshold",
    "confirm_cycles",
    "good_catches",
    "early_alarms",
    "missed",
    "median_lead_time",
]


def first_alarm_index(probs, threshold: float, confirm_cycles: int):
    """Index of the cycle where the alarm rings, or None if it never rings.

    Written as a plain loop so it reads like the rule it implements.
    """
    run = 0
    for i, p in enumerate(np.asarray(probs, dtype=float)):
        run = run + 1 if p >= threshold else 0
        if run >= confirm_cycles:
            return i
    return None


def classify(lead_time, min_lead: int, max_lead: int) -> str:
    """Outcome for one engine given its lead time (None = alarm never rang)."""
    if lead_time is None or pd.isna(lead_time):
        return MISSED
    if lead_time < min_lead:
        return MISSED
    if lead_time > max_lead:
        return EARLY
    return GOOD


def engine_outcome(cycles, probs, ruls, threshold, confirm_cycles, min_lead, max_lead) -> dict:
    """Simulate one engine. Inputs must be in cycle order."""
    cycles = np.asarray(cycles)
    ruls = np.asarray(ruls)
    i = first_alarm_index(probs, threshold, confirm_cycles)
    alarm_cycle = None if i is None else int(cycles[i])
    lead = None if i is None else int(ruls[i])
    return {
        "alarm_cycle": alarm_cycle,
        "lead_time": lead,
        "failure_cycle": int(cycles[-1] + ruls[-1]),
        "outcome": classify(lead, min_lead, max_lead),
        "too_late": i is not None and lead < min_lead,
    }


def simulate(
    timelines: pd.DataFrame, threshold: float, confirm_cycles: int, min_lead: int, max_lead: int
) -> pd.DataFrame:
    """One row per engine. `timelines` needs unit, cycle, failure_probability, rul."""
    rows = []
    for unit, g in timelines.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        res = engine_outcome(
            g["cycle"],
            g["failure_probability"],
            g["rul"],
            threshold,
            confirm_cycles,
            min_lead,
            max_lead,
        )
        rows.append({"unit": int(unit), **res})
    return pd.DataFrame(
        rows, columns=["unit", "alarm_cycle", "lead_time", "failure_cycle", "outcome", "too_late"]
    )


def _median(values) -> float:
    vals = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.median(vals)) if vals else float("nan")


def _mean(values) -> float:
    vals = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(outcomes: pd.DataFrame, cost_miss: float, cost_false: float) -> dict:
    """Counts, cost and lead times for a set of simulated engines."""
    counts = outcomes["outcome"].value_counts()
    good = int(counts.get(GOOD, 0))
    early = int(counts.get(EARLY, 0))
    missed = int(counts.get(MISSED, 0))
    too_late = int(outcomes["too_late"].sum())
    # Alarms that rang: good catches, early alarms and alarms that came too late.
    rang = good + early + too_late
    return {
        "engines": int(len(outcomes)),
        "good_catches": good,
        "early_alarms": early,
        "missed": missed,
        "missed_never_rang": missed - too_late,
        "missed_too_late": too_late,
        "alarms_rang": rang,
        # Share of alarms that were good catches. Early alarms are the false
        # alarm count.
        "precision": good / rang if rang else float("nan"),
        "cost": float(cost_miss * missed + cost_false * early),
        "median_lead_time": _median(outcomes["lead_time"]),
        "mean_lead_time_good_catches": _mean(
            outcomes.loc[outcomes["outcome"] == GOOD, "lead_time"]
        ),
    }


def _confirmed(probs: np.ndarray, k: int) -> np.ndarray:
    """Lowest probability over the last k cycles (-inf until k cycles exist).

    The alarm condition "probability >= t for k consecutive cycles" is the
    same as "this value >= t".
    """
    out = np.full(len(probs), -np.inf)
    if len(probs) >= k:
        out[k - 1 :] = np.lib.stride_tricks.sliding_window_view(probs, k).min(axis=1)
    return out


def lifecycle_sweep(
    timelines: pd.DataFrame,
    min_lead: int,
    max_lead: int,
    confirm_options=(1, 2, 3),
    thresholds=THRESHOLDS,
) -> pd.DataFrame:
    """Outcome counts for every (threshold, confirm_cycles) pair.

    Gives the same answers as `simulate` (tests check this), computed for all
    thresholds at once. Costs are applied separately, so the same sweep can be
    re-priced under any cost ratio.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    engines = [
        (g["failure_probability"].to_numpy(float), g["rul"].to_numpy(float))
        for _, g in timelines.sort_values(["unit", "cycle"]).groupby("unit", sort=True)
    ]
    rows = []
    for k in confirm_options:
        leads = np.full((len(engines), len(thresholds)), np.nan)
        for e, (probs, ruls) in enumerate(engines):
            reached = np.maximum.accumulate(_confirmed(probs, k))
            idx = np.searchsorted(reached, thresholds, side="left")
            rang = idx < len(probs)
            leads[e, rang] = ruls[idx[rang]]
        rang = ~np.isnan(leads)
        good = rang & (leads >= min_lead) & (leads <= max_lead)
        early = rang & (leads > max_lead)
        missed = ~(good | early)
        any_rang = rang.any(axis=0)
        med = np.full(len(thresholds), np.nan)
        if any_rang.any():
            med[any_rang] = np.nanmedian(leads[:, any_rang], axis=0)
        for j, t in enumerate(thresholds):
            rows.append(
                {
                    "threshold": float(t),
                    "confirm_cycles": int(k),
                    "good_catches": int(good[:, j].sum()),
                    "early_alarms": int(early[:, j].sum()),
                    "missed": int(missed[:, j].sum()),
                    "median_lead_time": float(med[j]),
                }
            )
    return pd.DataFrame(rows, columns=SWEEP_COLUMNS)


def price(sweep: pd.DataFrame, cost_miss: float, cost_false: float) -> pd.DataFrame:
    """Sweep with a cost column added."""
    out = sweep.copy()
    out["cost"] = cost_miss * out["missed"] + cost_false * out["early_alarms"]
    return out


def best_policy(sweep: pd.DataFrame, cost_miss: float, cost_false: float):
    """Lowest-cost (threshold, confirm_cycles).

    Ties go to the lower threshold, then to fewer confirm cycles: both ring
    sooner, which protects against the more expensive mistake (a miss).
    """
    priced = price(sweep, cost_miss, cost_false)
    best = priced.sort_values(["cost", "threshold", "confirm_cycles"], kind="mergesort").iloc[0]
    return float(best["threshold"]), int(best["confirm_cycles"])
