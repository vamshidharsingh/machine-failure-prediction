"""Decisions and evaluation with Scania's official cost matrix.

COST_MATRIX[n][m] is the cost of predicting class m when the true class is n.
Predicting too late (m < n) is expensive (200-500); predicting too early or
raising an unneeded check (m > n) is cheap (7-10).
"""

from itertools import product

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)

N_CLASSES = 5
COST_MATRIX = np.array(
    [
        [0, 7, 8, 9, 10],
        [200, 0, 7, 8, 9],
        [300, 200, 0, 7, 8],
        [400, 300, 200, 0, 7],
        [500, 400, 300, 200, 0],
    ],
    dtype=float,
)

DECISIONS = ("argmax", "lowest_expected_cost")


def confusion(y_true, y_pred) -> np.ndarray:
    """5 x 5 counts, rows = true class, columns = predicted class."""
    return sk_confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))


def cost_from_confusion(cm, cost=COST_MATRIX) -> float:
    """Total official cost of a confusion matrix."""
    return float((np.asarray(cm) * cost).sum())


def total_cost(y_true, y_pred, cost=COST_MATRIX) -> float:
    """Total official cost of a set of predictions."""
    return float(cost[np.asarray(y_true, int), np.asarray(y_pred, int)].sum())


def always_class_0(n: int) -> np.ndarray:
    """Predictions of the do-nothing baseline."""
    return np.zeros(n, dtype=int)


def decide(probs, rule: str, cost=COST_MATRIX) -> np.ndarray:
    """Turn class probabilities into one class per vehicle.

    argmax: the most likely class. lowest_expected_cost: for each vehicle, the class m minimising
    sum_n P(n) * cost[n][m].
    """
    probs = np.asarray(probs, float)
    if rule == "argmax":
        return probs.argmax(axis=1)
    if rule == "lowest_expected_cost":
        return (probs @ cost).argmin(axis=1)
    raise ValueError(f"Unknown decision rule: {rule}")


def reweight(probs, weights) -> np.ndarray:
    """Prior-shift correction with the given class weights.

    Prior-shift correction: multiply each class probability by its weight and renormalise each row
    to sum to 1.
    """
    q = np.asarray(probs, float) * np.asarray(weights, float)
    return q / q.sum(axis=1, keepdims=True)


def fit_prior_weights(probs, y_true, rule: str, grid, cost=COST_MATRIX):
    """Search class weights that minimise cost on the given data.

    Class 0 keeps weight 1; the cost is that of `rule`. Call it with validation data only.

    Ties go to the weights closest to no correction (smallest sum of |log2 weight|), then to the
    first in grid order. Returns (weights, cost, n_candidates).
    """
    probs = np.asarray(probs, float)
    y_true = np.asarray(y_true, int)
    best_key, best_w = None, None
    n = 0
    for combo in product(grid, repeat=N_CLASSES - 1):
        n += 1
        w = np.array((1.0, *combo))
        c = total_cost(y_true, decide(reweight(probs, w), rule, cost), cost)
        key = (c, float(np.abs(np.log2(w)).sum()))
        if best_key is None or key < best_key:
            best_key, best_w = key, w
    return best_w, best_key[0], n


def capacity_decisions(probs, share: float, cost=COST_MATRIX, tiebreak=None) -> np.ndarray:
    """Decisions when the workshop can inspect only a `share` of vehicles.

    The riskiest share of vehicles by at-risk probability (P(fail within 48
    steps)) are flagged and get the class 1-4 with the lowest expected cost;
    everyone else gets class 0. Calibration can give many vehicles the same
    probability, so ties are broken by `tiebreak` (higher first; the pipeline
    passes the uncalibrated model score), then by input order.
    """
    probs = np.asarray(probs, float)
    n_flag = int(round(share * len(probs)))
    risk = at_risk_probability(probs)
    if tiebreak is None:
        order = np.argsort(-risk, kind="mergesort")
    else:
        order = np.lexsort((-np.asarray(tiebreak, float), -risk))
    pred = np.zeros(len(probs), dtype=int)
    flagged = order[:n_flag]
    pred[flagged] = 1 + (probs[flagged] @ cost[:, 1:]).argmin(axis=1)
    return pred


def rule_decisions(probs, rule: str, cost=COST_MATRIX, tiebreak=None) -> np.ndarray:
    """Decisions for one rule: unconstrained or capacity_<k>.

    "unconstrained" = lowest expected cost for every vehicle; "capacity_<k>" = capacity_decisions
    with k% of vehicles.
    """
    if rule == "unconstrained":
        return decide(probs, "lowest_expected_cost", cost)
    if rule.startswith("capacity_"):
        return capacity_decisions(probs, float(rule.split("_")[1]) / 100, cost, tiebreak)
    raise ValueError(f"Unknown decision rule: {rule}")


def capacity_rules(capacities) -> list:
    """Rule names: unconstrained first, then capacity_<k> for each share."""
    return ["unconstrained", *[f"capacity_{round(c * 100):g}" for c in capacities]]


def at_risk_probability(probs) -> np.ndarray:
    """P(class 1 to 4) = probability the vehicle fails within 48 time steps."""
    return np.asarray(probs, float)[:, 1:].sum(axis=1)


def summarize(y_true, y_pred, probs=None, cost=COST_MATRIX) -> dict:
    """Every reported number for one set of predictions."""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    cm = confusion(y_true, y_pred)
    total = cost_from_confusion(cm, cost)
    baseline = total_cost(y_true, always_class_0(len(y_true)), cost)
    true_risk, pred_risk = y_true > 0, y_pred > 0
    alerts = int(pred_risk.sum())
    caught = int((pred_risk & true_risk).sum())
    out = {
        "vehicles": int(len(y_true)),
        "total_cost": total,
        "cost_per_vehicle": total / len(y_true),
        "always_class_0_cost": baseline,
        "cost_saving_vs_always_class_0": baseline - total,
        "confusion_matrix": cm.tolist(),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, labels=list(range(N_CLASSES)), average="macro", zero_division=0
            )
        ),
        "at_risk": {
            "true_at_risk": int(true_risk.sum()),
            "alarms": alerts,
            "caught": caught,
            "missed": int((~pred_risk & true_risk).sum()),
            "false_alarms": int((pred_risk & ~true_risk).sum()),
            "flagged_share": alerts / len(y_true),
            "healthy_flagged_share": int((pred_risk & ~true_risk).sum())
            / max(int((~true_risk).sum()), 1),
            "precision": caught / alerts if alerts else None,
            "recall": caught / int(true_risk.sum()) if true_risk.any() else None,
            "pr_auc": None,
        },
    }
    if probs is not None and true_risk.any() and not true_risk.all():
        out["at_risk"]["pr_auc"] = float(
            average_precision_score(true_risk, at_risk_probability(probs))
        )
    return out


def _interval(values) -> list:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_costs(y_true, predictions: dict, n: int, seed: int, cost=COST_MATRIX) -> dict:
    """Resample vehicles with replacement `n` times (seeded).

    For each named prediction: 95% interval of total cost and of the saving vs always class 0. For
    each pair of predictions: 95% interval of the cost difference.
    """
    y_true = np.asarray(y_true, int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    base = cost[y_true, 0][idx].sum(axis=1)
    totals = {
        name: cost[y_true, np.asarray(p, int)][idx].sum(axis=1) for name, p in predictions.items()
    }
    out = {"samples": n, "seed": seed, "always_class_0_cost_95": _interval(base), "models": {}}
    for name, tot in totals.items():
        saving = base - tot
        out["models"][name] = {
            "total_cost_95": _interval(tot),
            "saving_vs_always_class_0_95": _interval(saving),
            "share_of_samples_with_saving_above_0": float((saving > 0).mean()),
        }
    names = list(totals)
    out["differences"] = {
        f"{b}_minus_{a}": _interval(totals[b] - totals[a])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    return out
