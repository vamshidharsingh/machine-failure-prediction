"""Cost-based alert threshold.

Cost(t) = C_miss * missed_failures(t) + C_false * false_alarms(t)

Each row is one inspection decision for one engine at one cycle: raise a
maintenance alert or not. The costs are assumptions, set in config.py.
"""

import numpy as np
import pandas as pd

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def confusion_counts(y_true, scores, threshold: float) -> dict:
    """Alert outcome counts when alerting at score >= threshold."""
    y_true = np.asarray(y_true).astype(int)
    alert = np.asarray(scores) >= threshold
    return {
        "true_alerts": int(np.sum(alert & (y_true == 1))),
        "false_alarms": int(np.sum(alert & (y_true == 0))),
        "missed_failures": int(np.sum(~alert & (y_true == 1))),
        "correct_quiet": int(np.sum(~alert & (y_true == 0))),
    }


def expected_cost(missed: int, false_alarms: int, cost_miss: float, cost_false: float) -> float:
    """Cost of missed failures and false alarms."""
    return float(cost_miss * missed + cost_false * false_alarms)


def threshold_sweep(y_true, scores, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Confusion counts at every threshold.

    Costs are applied separately so the same sweep can be re-priced under any cost ratio (the UI
    does this).
    """
    rows = [{"threshold": float(t), **confusion_counts(y_true, scores, t)} for t in thresholds]
    return pd.DataFrame(rows)


def price_sweep(sweep: pd.DataFrame, cost_miss: float, cost_false: float) -> pd.DataFrame:
    """Threshold sweep with a cost column added."""
    out = sweep.copy()
    out["cost"] = cost_miss * out["missed_failures"] + cost_false * out["false_alarms"]
    return out


def best_threshold(sweep: pd.DataFrame, cost_miss: float, cost_false: float) -> float:
    """Lowest-cost threshold.

    Ties go to the lower threshold, which misses fewer failures, because a miss is the more
    expensive mistake.
    """
    priced = price_sweep(sweep, cost_miss, cost_false)
    min_cost = priced["cost"].min()
    return float(priced.loc[priced["cost"] == min_cost, "threshold"].min())
