"""Pure data logic behind the Streamlit app, kept separate so it is tested.

Everything here reads only the saved files in reports/, so the dashboard
deploys without raw data or model files.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import lifecycle
from .policy import best_threshold, price_sweep

LIFECYCLE_SET = "lifecycle_test"


def reprice(sweep: pd.DataFrame, cost_ratio: float):
    """Re-choose the per-cycle threshold for a new miss:false-alarm cost ratio."""
    priced = price_sweep(sweep, cost_miss=cost_ratio, cost_false=1.0)
    threshold = best_threshold(sweep, cost_miss=cost_ratio, cost_false=1.0)
    return priced, threshold


def engine_table(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-cycle alert outcome for each engine at one threshold, riskiest first."""
    out = preds.copy()
    out["alert"] = out["failure_probability"] >= threshold
    out["outcome"] = "Correct, no alert"
    out.loc[out["alert"] & (out["at_risk_true"] == 1), "outcome"] = "Correct alert"
    out.loc[out["alert"] & (out["at_risk_true"] == 0), "outcome"] = "False alarm"
    out.loc[~out["alert"] & (out["at_risk_true"] == 1), "outcome"] = "Missed failure"
    return out.sort_values("failure_probability", ascending=False).reset_index(drop=True)


def available_subsets(reports_root: Path) -> list:
    """Subsets that have a saved metrics.json, e.g. ["FD001", "FD002"]."""
    return sorted(p.parent.name for p in Path(reports_root).glob("FD00*/metrics.json"))


def load_subset(reports_root: Path, subset: str) -> dict:
    """Saved metrics, validation sweep, timelines and test predictions for one NASA subset."""
    folder = Path(reports_root) / subset
    with open(folder / "metrics.json") as f:
        metrics = json.load(f)
    return {
        "metrics": metrics,
        "sweep": pd.read_csv(folder / "validation_lifecycle_sweep.csv"),
        "timelines": pd.read_csv(folder / "engine_timelines.csv"),
        "predictions": pd.read_csv(folder / "test_predictions.csv"),
    }


def choose_policy(sweep: pd.DataFrame, cost_ratio: float):
    """Re-choose the alarm policy for a new cost ratio.

    Re-choose (threshold, confirm_cycles) from the validation lifecycle sweep for a new miss:early-
    alarm cost ratio. Returns (threshold, confirm, priced).
    """
    threshold, confirm = lifecycle.best_policy(sweep, cost_miss=cost_ratio, cost_false=1.0)
    priced = lifecycle.price(sweep, cost_miss=cost_ratio, cost_false=1.0)
    return threshold, confirm, priced


def lifecycle_engines(timelines: pd.DataFrame) -> pd.DataFrame:
    """Timeline rows of the lifecycle test engines."""
    return timelines[timelines["set"] == LIFECYCLE_SET]


def lifecycle_results(timelines, threshold, confirm, min_lead, max_lead, cost_ratio) -> dict:
    """Outcome counts on the lifecycle test engines for one policy."""
    outcomes = lifecycle.simulate(
        lifecycle_engines(timelines), threshold, confirm, min_lead, max_lead
    )
    return lifecycle.summarize(outcomes, cost_miss=cost_ratio, cost_false=1.0)


def lifecycle_table(timelines, threshold, confirm, min_lead, max_lead) -> pd.DataFrame:
    """One row per lifecycle test engine with its outcome under one policy."""
    return lifecycle.simulate(lifecycle_engines(timelines), threshold, confirm, min_lead, max_lead)


def engine_timeline(timelines: pd.DataFrame, unit: int, set_name: str = LIFECYCLE_SET):
    """One engine's cycles in order."""
    g = timelines[(timelines["set"] == set_name) & (timelines["unit"] == unit)]
    return g.sort_values("cycle").reset_index(drop=True)


def engine_alarm(timeline: pd.DataFrame, threshold, confirm, min_lead, max_lead) -> dict:
    """When the alarm rings for one engine, and a plain sentence about it."""
    res = lifecycle.engine_outcome(
        timeline["cycle"],
        timeline["failure_probability"],
        timeline["rul"],
        threshold,
        confirm,
        min_lead,
        max_lead,
    )
    res["alarm_probability"] = (
        None
        if res["alarm_cycle"] is None
        else float(
            timeline.loc[timeline["cycle"] == res["alarm_cycle"], "failure_probability"].iloc[0]
        )
    )
    res["sentence"] = alarm_sentence(res, min_lead, max_lead)
    return res


def alarm_sentence(res: dict, min_lead: int, max_lead: int) -> str:
    """One plain sentence about when the alarm rang and what that means."""
    lead = res["lead_time"]
    if lead is None:
        return "The alarm never rang before failure: missed."
    cycles = "cycle" if lead == 1 else "cycles"
    if res["outcome"] == lifecycle.GOOD:
        return f"The alarm rang {lead} {cycles} before failure: good catch."
    if res["outcome"] == lifecycle.EARLY:
        return (
            f"The alarm rang {lead} {cycles} before failure: early alarm "
            f"(more than {max_lead} cycles ahead, so the check was wasted)."
        )
    return (
        f"The alarm rang only {lead} {cycles} before failure: missed "
        f"(fewer than {min_lead} cycles is too late to act)."
    )


SCANIA = "Scania"
MODEL_LABELS = {
    "gradient_boosting": "Gradient boosting",
    "logistic_regression": "Logistic regression",
    "searched_correction": "Searched correction",
    "always_class_0": "Always class 0",
}


def available_datasets(reports_root: Path) -> list:
    """Scania first (the main case), then the NASA subsets that have results."""
    out = [SCANIA] if (Path(reports_root) / "scania" / "metrics.json").exists() else []
    return out + available_subsets(reports_root)


def _read_csv_if(path: Path):
    return pd.read_csv(path) if Path(path).exists() else None


def load_subset_extras(reports_root: Path, subset: str) -> dict:
    """Optional NASA files: warning-time sweeps and SHAP outputs (None if missing)."""
    folder = Path(reports_root) / subset
    return {
        "sweeps": _read_csv_if(folder / "validation_lifecycle_sweeps.csv"),
        "shap_top": _read_csv_if(folder / "shap_top_features.csv"),
        "shap_global": _read_csv_if(folder / "shap_global.csv"),
    }


def warning_time_options(sweeps) -> list:
    """Minimum lead times with a saved validation sweep, e.g. [10, 20, 30]."""
    if sweeps is None:
        return []
    return sorted(int(v) for v in sweeps["alert_min_lead"].unique())


def sweep_for_min_lead(sweeps: pd.DataFrame, min_lead: int) -> pd.DataFrame:
    """The validation lifecycle sweep computed with this minimum lead time."""
    out = sweeps[sweeps["alert_min_lead"] == min_lead].drop(columns="alert_min_lead")
    if out.empty:
        raise ValueError(f"No saved validation sweep for a minimum lead of {min_lead} cycles")
    return out.reset_index(drop=True)


def policy_comparison(
    timelines, threshold, confirm, min_lead, max_lead, cost_ratio
) -> pd.DataFrame:
    """The chosen policy and the two fixed rules on the lifecycle test engines."""
    policies = [
        ("Cost-based policy", threshold, confirm),
        ("Fixed 0.5", 0.5, 1),
        ("Fixed 0.5, same confirm cycles", 0.5, confirm),
    ]
    rows = []
    for name, t, k in policies:
        r = lifecycle_results(timelines, t, k, min_lead, max_lead, cost_ratio)
        rows.append(
            {
                "Policy": name,
                "Threshold": round(float(t), 2),
                "Confirm cycles": int(k),
                "Good catches": r["good_catches"],
                "Early alarms": r["early_alarms"],
                "Missed": r["missed"],
                "Precision": None if pd.isna(r["precision"]) else round(r["precision"], 3),
                "Cost": r["cost"],
                "Median lead time": None
                if pd.isna(r["median_lead_time"])
                else round(r["median_lead_time"], 1),
            }
        )
    return pd.DataFrame(rows)


def _fmt_value(v) -> str:
    if v is None or pd.isna(v):
        return "missing"
    v = float(v)
    if v == int(v) and abs(v) >= 1:
        return f"{int(v):,}"
    return f"{v:,.3f}" if abs(v) < 1000 else f"{v:,.0f}"


def explanation(shap_top, id_name: str, item, unit: str = None) -> dict:
    """Top features pushing one item's risk up and down, as plain sentences.

    With a `unit`, each sentence names it; without one, the sentences talk about
    the risk score and the unit is stated once by the caller. Returns
    {"explained": bool, "up": [...], "down": [...]}.
    """
    if shap_top is None:
        return {"explained": False, "up": [], "down": []}
    rows = shap_top[shap_top[id_name] == item]
    if rows.empty:
        return {"explained": False, "up": [], "down": []}
    out = {"explained": True}
    for direction, verb in (("up", "raised"), ("down", "lowered")):
        part = rows[rows["direction"] == direction].sort_values("rank")
        target, suffix = ("the risk", f" ({unit})") if unit else ("the risk score", "")
        out[direction] = [
            f"{r.readable_name} = {_fmt_value(r.feature_value)} {verb} {target} by "
            f"{abs(r.shap_value):.3f}{suffix}"
            for r in part.itertuples()
        ]
    return out


def load_scania(reports_root: Path) -> dict:
    """Saved Scania metrics, test predictions and SHAP files."""
    folder = Path(reports_root) / "scania"
    with open(folder / "metrics.json") as f:
        metrics = json.load(f)
    return {
        "metrics": metrics,
        "predictions": pd.read_csv(folder / "test_predictions.csv"),
        "shap_top": _read_csv_if(folder / "shap_top_features.csv"),
        "shap_global": _read_csv_if(folder / "shap_global.csv"),
        "shap_global_flagged": _read_csv_if(folder / "shap_global_flagged.csv"),
    }


UNLIMITED = "unconstrained"


def capacity_options(metrics: dict) -> list:
    """Decision rules the saved predictions cover, smallest capacity first, unlimited last."""
    rules = metrics.get("capacity", {}).get("rules", [UNLIMITED])
    limited = sorted((r for r in rules if r != UNLIMITED), key=lambda r: float(r.split("_")[1]))
    return [*limited, UNLIMITED]


def capacity_label(rule: str) -> str:
    """ "20%" for capacity_20, "unlimited" for the unconstrained rule."""
    return "unlimited" if rule == UNLIMITED else f"{rule.split('_')[1]}%"


def recommended_capacity(metrics: dict) -> str:
    """The recommended decision rule, or unlimited if none was recommended."""
    return metrics.get("capacity", {}).get("recommended") or UNLIMITED


def prediction_column(rule: str) -> str:
    """Column of test_predictions.csv that holds a rule's decisions."""
    return "predicted_class" if rule == UNLIMITED else f"predicted_class_{rule}"


def scania_summary(metrics: dict, preds: pd.DataFrame = None, rule: str = UNLIMITED) -> dict:
    """Test numbers for the chosen model under one workshop capacity, against always class 0.

    With `preds`, they are computed from the saved predictions; otherwise read from the saved
    metrics.
    """
    from .scania.evaluate import summarize

    if preds is None:
        final = metrics["test"]["final"]
    else:
        final = summarize(preds["true_class"].to_numpy(), preds[prediction_column(rule)].to_numpy())
    ar = final["at_risk"]
    choice = metrics["final_choice"]
    boot_block = metrics.get("bootstrap_test", {})
    boot = boot_block.get("by_rule", {}).get(rule, {}).get("models", {}).get(choice["model"]) or (
        boot_block.get("models", {}).get(choice["model"], {}) if rule == UNLIMITED else {}
    )
    return {
        "model": MODEL_LABELS[choice["model"]],
        "capacity": capacity_label(rule),
        "decision": choice["decision"].replace("_", " "),
        "prior_correction": bool(choice["prior_correction"]),
        "test_cost": final["total_cost"],
        "always_class_0_cost": final["always_class_0_cost"],
        "saving": final["cost_saving_vs_always_class_0"],
        "saving_95": boot.get("saving_vs_always_class_0_95"),
        "precision": ar["precision"],
        "recall": ar["recall"],
        "false_alarms": ar["false_alarms"],
        "healthy_flagged_share": ar.get("healthy_flagged_share"),
        "alarms": ar["alarms"],
        "true_at_risk": ar["true_at_risk"],
        "confusion_matrix": final["confusion_matrix"],
    }


def confusion_long(cm) -> pd.DataFrame:
    """Confusion matrix as rows of (true class, predicted class, vehicles, cost)."""
    from .scania.evaluate import COST_MATRIX

    cm = np.asarray(cm)
    return pd.DataFrame(
        [
            {
                "True class": i,
                "Predicted class": j,
                "Vehicles": int(cm[i, j]),
                "Cost": float(cm[i, j] * COST_MATRIX[i, j]),
            }
            for i in range(cm.shape[0])
            for j in range(cm.shape[1])
        ]
    )


def scania_vehicle_table(preds: pd.DataFrame, shap_top=None, rule: str = UNLIMITED) -> pd.DataFrame:
    """One row per test vehicle under one workshop capacity, riskiest first."""
    explained = set() if shap_top is None else set(shap_top["vehicle_id"])
    out = pd.DataFrame(
        {
            "Vehicle": preds["vehicle_id"],
            "True class": preds["true_class"],
            "Predicted class": preds[prediction_column(rule)],
            "At-risk probability": preds["at_risk_probability"],
            "Explained": preds["vehicle_id"].isin(explained),
        }
    )
    # Same order as the capacity rule: calibrated risk, then the uncalibrated score.
    score = preds["model_score"] if "model_score" in preds else preds["at_risk_probability"]
    out = out.assign(_score=score.to_numpy())
    out = out.sort_values(
        ["At-risk probability", "_score", "Vehicle"], ascending=[False, False, True]
    ).drop(columns="_score")
    return out.reset_index(drop=True)


def default_vehicle(table: pd.DataFrame):
    """The riskiest correctly flagged vehicle, or the riskiest vehicle if there is none."""
    hits = table[(table["True class"] > 0) & (table["Predicted class"] > 0)]
    return int((hits if not hits.empty else table)["Vehicle"].iloc[0])


def load_comparison(reports_root: Path):
    """reports/comparison.csv with readable column and model names, or None."""
    path = Path(reports_root) / "comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["model"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "pr_auc": "PR-AUC",
            "precision": "Precision",
            "recall": "Recall",
            "false_alarms": "False alarms",
            "official_cost": "Official cost (Scania)",
            "lifecycle_cost": "Lifecycle cost (NASA)",
            "rmse": "RMSE (NASA remaining life)",
            "nasa_score": "NASA score",
        }
    )


COMPARISON_FORMATS = {
    "PR-AUC": "{:.3f}",
    "Precision": "{:.3f}",
    "Recall": "{:.3f}",
    "False alarms": "{:,.0f}",
    "Official cost (Scania)": "{:,.0f}",
    "Lifecycle cost (NASA)": "{:,.0f}",
    "RMSE (NASA remaining life)": "{:.1f}",
    "NASA score": "{:,.0f}",
}


def comparison_display(df: pd.DataFrame) -> pd.DataFrame:
    """Comparison table formatted for display.

    Formatted copy for display: rounded numbers, blank where a metric does not apply (same
    formatting as the README table).
    """
    out = df.copy()
    for col, fmt in COMPARISON_FORMATS.items():
        if col in out:
            out[col] = [("" if pd.isna(v) else fmt.format(v)) for v in out[col]]
    return out


def scania_takeaway(s: dict) -> str:
    """One-line summary of the Scania test result, from scania_summary()."""
    pct = s["saving"] / s["always_class_0_cost"] if s["always_class_0_cost"] else 0
    where = (
        "with unlimited workshop capacity"
        if s.get("capacity", "unlimited") == "unlimited"
        else f"at {s['capacity']} workshop capacity"
    )
    lead = (
        f"{s['model']} {where} saves {s['saving']:,.0f} ({pct:.0%}) on the official test cost "
        "against always predicting class 0"
    )
    if s["saving_95"]:
        lo, hi = s["saving_95"]
        verdict = (
            "above zero, so the saving is unlikely to be luck"
            if lo > 0
            else "includes zero, so the saving may be luck"
        )
        lead += f"; the 95% interval ({lo:,.0f} to {hi:,.0f}) is {verdict}"
    if s["alarms"] > 2 * s["true_at_risk"]:
        tail = (
            f"It gets there by flagging many vehicles: {s['alarms']:,} alarms for "
            f"{s['true_at_risk']} vehicles really at risk."
        )
    else:
        tail = f"It raises {s['alarms']:,} alarms for {s['true_at_risk']} vehicles really at risk."
    return f"{lead}. {tail}"


def _compare_word(a: float, b: float) -> str:
    return "beats" if a < b else ("ties with" if a == b else "loses to")


def nasa_takeaway(metrics: dict) -> str:
    """One-line summary of one NASA subset, from its saved metrics."""
    lt = metrics["lifecycle_test"]
    c, f = lt["chosen"]["cost"], lt["fixed_0.5"]["cost"]
    n = lt["chosen"]["engines"]
    parts = [
        f"On the {n} lifecycle test engines the cost-based alarm {_compare_word(c, f)} a fixed "
        f"0.5 threshold (cost {c:.0f} vs {f:.0f})"
    ]
    fk = lt.get("fixed_0.5_same_confirm")
    if fk and fk["cost"] < c:
        parts.append(f"fixed 0.5 with the same confirm cycles does better ({fk['cost']:.0f})")
    lr = metrics.get("baseline_logistic_regression")
    if lr and lr["lifecycle_test"]["chosen"]["cost"] < c:
        parts.append(
            f"logistic regression with its own policy costs less "
            f"({lr['lifecycle_test']['chosen']['cost']:.0f})"
        )
    r = metrics["rul"]
    rul = (
        f"Remaining life: gradient boosting's typical error (RMSE) is {r['rmse_model']:.1f} cycles"
        + (f", against {r['rmse_ridge']:.1f} for ridge" if "rmse_ridge" in r else "")
    )
    return "; ".join(parts) + ". " + rul + "."


# For each comparison column: True if higher is better, False if lower is better.
COMPARISON_DIRECTION = {
    "PR-AUC": True,
    "Precision": True,
    "Recall": True,
    "False alarms": False,
    "Official cost (Scania)": False,
    "Lifecycle cost (NASA)": False,
    "RMSE (NASA remaining life)": False,
    "NASA score": False,
}


def best_cells(df: pd.DataFrame, group: str = "Dataset", directions=None) -> pd.DataFrame:
    """Mask of the best value in each column within each group.

    Same shape as df: True where a value is the best in its column within its
    group. Ties all count; missing values never do.
    """
    directions = directions or COMPARISON_DIRECTION
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col, higher in directions.items():
        if col not in df:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        target = values.groupby(df[group]).transform("max" if higher else "min")
        mask[col] = values.notna() & (values == target)
    return mask


def lowest_cost_rows(costs) -> list:
    """Positions of the cheapest entries (all of them on a tie)."""
    costs = pd.Series(costs, dtype=float).reset_index(drop=True)
    return costs.index[costs == costs.min()].tolist()
