"""Scania Component X pipeline.

    python -m mfp.scania.pipeline              # full run
    python -m mfp.scania.pipeline --no-tune    # skip the tuning pass
    python -m mfp.scania.pipeline --select     # rerun model selection first

Steps: load and validate -> labels and past-only features -> model selection
on training and validation data (mfp.scania.selection; reused from
reports/scania/selection.json when present) -> fit the chosen model, the
logistic-regression baseline and the searched-correction model with out-of-fold
predictions on the simulated validation set -> calibrate -> choose the
recommended workshop capacity on the simulated validation set -> report the
official validation set -> score the test set once -> save reports.

Nothing is fitted, calibrated or chosen on test vehicles.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, replace

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss, precision_recall_curve  # noqa: E402

from .. import __version__  # noqa: E402
from ..config import ROOT, ScaniaConfig  # noqa: E402
from ..explain import risk_shap, save_explanations, tree_shap  # noqa: E402
from ..models import tune_boosting  # noqa: E402
from ..report import write_summary  # noqa: E402
from .data import N_CLASSES, load_split  # noqa: E402
from .evaluate import (  # noqa: E402
    COST_MATRIX,
    always_class_0,
    at_risk_probability,
    bootstrap_costs,
    capacity_rules,
    rule_decisions,
    summarize,
)
from .features import last_readout_rows, readout_features  # noqa: E402
from .horizons import HorizonModel, class_shares  # noqa: E402
from .selection import (  # noqa: E402
    Calibration,
    Data,
    FittedCandidate,
    _training_mask,
    fit_candidate,
    run_selection,
)

MODELS = ("gradient_boosting", "logistic_regression", "searched_correction")
LABELS = {
    "gradient_boosting": "Gradient boosting",
    "logistic_regression": "Logistic regression",
    "searched_correction": "Searched correction",
    "always_class_0": "Always class 0",
}
SEARCHED_CORRECTION = {
    "target": "five_class",
    "learner": "gradient_boosting",
    "weighted": False,
    "features": [],
}
SHAP_FILES = (
    "feature_names.csv",
    "shap_global.csv",
    "shap_global_flagged.csv",
    "shap_top_features.csv",
    "shap_summary.png",
)


def class_probabilities(model, X) -> np.ndarray:
    """n x 5 probabilities, one column per class 0-4 (zero for a class the model never saw)."""
    raw = model.predict_proba(X)
    out = np.zeros((len(X), N_CLASSES))
    out[:, np.asarray(model.classes_, int)] = raw
    return out


def neg_log_loss(model, X, y) -> float:
    """Negated multiclass log loss (higher is better), for tuning."""
    return -float(log_loss(y, class_probabilities(model, X), labels=list(range(N_CLASSES))))


def _num(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return _num(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _vehicle_set(split: dict, cfg: ScaniaConfig):
    """One feature row per vehicle, from all of its readouts.

    One row per vehicle from all of its readouts (its last readout is the labelled point), with
    every optional feature group computed.
    """
    feats = last_readout_rows(
        readout_features(split["readouts"], cfg.change_window, extras=("intensity", "rate10"))
    )
    y = split["labels"].set_index("vehicle_id").loc[feats["vehicle_id"], "class_label"]
    return feats, y.to_numpy(int)


def _candidate(setup: dict) -> dict:
    return {k: setup[k] for k in ("target", "learner", "weighted", "features")}


def recommend_capacity(costs: dict, tolerance: float):
    """Smallest capacity whose cost is within `tolerance` of the unconstrained cost.

    `costs` maps rule name -> cost. Returns (rule or None, reason).
    """
    limit = costs["unconstrained"] * (1 + tolerance)
    capacity = sorted(
        (r for r in costs if r != "unconstrained"), key=lambda r: float(r.split("_")[1])
    )
    for rule in capacity:
        if costs[rule] <= limit:
            return rule, (
                f"smallest capacity with cost within {tolerance:.0%} of the "
                f"unconstrained cost ({costs[rule]:,.0f} vs {costs['unconstrained']:,.0f})"
            )
    return None, (
        f"no capacity tried is within {tolerance:.0%} of the unconstrained cost "
        f"({costs['unconstrained']:,.0f})"
    )


def _tune(data: Data, cfg: ScaniaConfig, log) -> dict:
    """Tune the five-class gradient boosting settings.

    Grid search for the five-class gradient boosting on a seeded share of training vehicles,
    vehicle-grouped folds, multiclass log loss.
    """
    rng = np.random.default_rng(cfg.seed)
    vehicles = np.unique(data.vehicle)
    n_tune = max(cfg.tuning_folds, int(round(len(vehicles) * cfg.tuning_fraction)))
    chosen = rng.choice(vehicles, size=min(n_tune, len(vehicles)), replace=False)
    mask = np.isin(data.vehicle, chosen) & (data.five_class >= 0)
    X = data.matrix(data.feats[mask], data.train_specs, ())
    best, rows = tune_boosting(
        X,
        data.five_class[mask],
        data.vehicle[mask],
        cfg.tuning_grid,
        cfg.tuning_folds,
        cfg.seed,
        scorer=neg_log_loss,
        score_name="cv_neg_log_loss",
    )
    pd.DataFrame(rows).to_csv(cfg.reports_dir / "tuning_results.csv", index=False)
    return {
        "score": "multiclass log loss (negated; higher is better)",
        "tuning_vehicles": int(len(chosen)),
        "tuning_rows": int(mask.sum()),
        "best_params": best,
        "results": rows,
    }


def run(
    cfg: ScaniaConfig,
    tune: bool = True,
    verbose: bool = True,
    select: bool = False,
    shap: bool = True,
) -> dict:
    """Run the Scania pipeline and write every report; returns the saved metrics."""
    log = print if verbose else (lambda *a, **k: None)
    start = time.time()
    timings = {}

    def tick(name, t0):
        """Record and print how long a step took."""
        timings[name] = round(time.time() - t0, 1)
        log(f"  {name}: {timings[name]:.1f}s", flush=True)

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model selection (training and validation data only).
    selection_path = cfg.reports_dir / "selection.json"
    if select or not selection_path.exists():
        t0 = time.time()
        run_selection(cfg, log)
        tick("model selection", t0)
    selection = json.loads(selection_path.read_text())

    # 2. Data, labels and features.
    t0 = time.time()
    data = Data(cfg, log)
    test_split = load_split(cfg.data_dir, "test", cfg.use_cache)
    test_feats, y_test = _vehicle_set(test_split, cfg)
    test_ids = test_feats["vehicle_id"].to_numpy()
    y_sim, folds = data.sim["class_label"].to_numpy(), data.sim["fold"].to_numpy()
    tick("data", t0)

    # 3. Optional tuning of the gradient boosting settings.
    tuning = None
    if tune:
        t0 = time.time()
        tuning = _tune(data, cfg, log)
        if tuning["best_params"]:
            cfg = replace(cfg, boosting_params={**cfg.boosting_params, **tuning["best_params"]})
        tick("tuning", t0)

    # 4. Fit the three models with out-of-fold predictions on the simulated
    #    validation set (all training rows), then calibrate each.
    setups = {
        "gradient_boosting": selection,
        "logistic_regression": selection["baseline"],
        "searched_correction": SEARCHED_CORRECTION,
    }
    calibration_kind = {
        "gradient_boosting": selection["calibration"],
        "logistic_regression": selection["baseline"]["calibration"],
        "searched_correction": "searched_on_validation",
    }
    fitted, oof, calibrations = {}, {}, {}
    for name in MODELS:
        cand = _candidate(setups[name])
        same = next((m for m in fitted if _candidate(setups[m]) == cand), None)
        if same is not None:
            fitted[name], oof[name] = fitted[same], oof[same]
        else:
            t0 = time.time()
            log(f"  {LABELS[name]}: {cand}", flush=True)
            oof[name], fitted[name] = fit_candidate(data, cand, cfg, 1.0, log)
            tick(f"fit {name}", t0)
    val_raw = {n: fitted[n].predict_proba(data.val_feats, data.val_specs) for n in MODELS}
    known = data.five_class >= 0
    train_shares = class_shares(data.five_class[known])
    for name in MODELS:
        calibrations[name] = Calibration(calibration_kind[name]).fit(
            oof[name], y_sim, folds, train_shares, val_raw[name], data.y_val, cfg.prior_weight_grid
        )

    # 5. Decision rules on the simulated and official validation sets; the
    #    recommended capacity is chosen on the simulated set only.
    rules = capacity_rules(cfg.capacities)
    sim_probs = {n: calibrations[n].oof_ for n in MODELS}
    val_probs = {n: calibrations[n].transform(val_raw[n]) for n in MODELS}
    # Ties in calibrated risk are broken by the uncalibrated model score.
    sim_score = {n: at_risk_probability(oof[n]) for n in MODELS}
    val_score = {n: at_risk_probability(val_raw[n]) for n in MODELS}
    simulated = {
        n: {
            r: summarize(
                y_sim, rule_decisions(sim_probs[n], r, tiebreak=sim_score[n]), sim_probs[n]
            )
            for r in rules
        }
        for n in MODELS
    }
    validation_by_model = {
        n: {
            r: summarize(
                data.y_val, rule_decisions(val_probs[n], r, tiebreak=val_score[n]), val_probs[n]
            )
            for r in rules
        }
        for n in MODELS
    }
    sim_costs = {r: simulated["gradient_boosting"][r]["total_cost"] for r in rules}
    recommended, reason = recommend_capacity(sim_costs, cfg.capacity_tolerance)

    # 6. Score the test set once: every model with every decision rule.
    t0 = time.time()
    test_raw = {n: fitted[n].predict_proba(test_feats, test_split["specs"]) for n in MODELS}
    test_probs = {n: calibrations[n].transform(test_raw[n]) for n in MODELS}
    test_score = {n: at_risk_probability(test_raw[n]) for n in MODELS}
    test_preds = {
        n: {r: rule_decisions(test_probs[n], r, tiebreak=test_score[n]) for r in rules}
        for n in MODELS
    }
    by_rule = {
        n: {r: summarize(y_test, test_preds[n][r], test_probs[n]) for r in rules} for n in MODELS
    }
    always = summarize(y_test, always_class_0(len(y_test)))
    by_model = {
        n: {
            "choice": {**setups[n], "calibration": calibration_kind[n]},
            **by_rule[n]["unconstrained"],
        }
        for n in MODELS
    }
    by_model["always_class_0"] = always
    boot = bootstrap_costs(
        y_test, {n: test_preds[n]["unconstrained"] for n in MODELS}, cfg.bootstrap_samples, cfg.seed
    )
    boot["by_rule"] = {}
    for r in rules:
        b = bootstrap_costs(
            y_test, {n: test_preds[n][r] for n in MODELS}, cfg.bootstrap_samples, cfg.seed
        )
        boot["by_rule"][r] = {
            "models": b["models"],
            "searched_correction_minus_gradient_boosting": b["differences"][
                "searched_correction_minus_gradient_boosting"
            ],
        }
    tick("test scoring and bootstrap", t0)

    # 7. SHAP for the headline's at-risk probability.
    shap_block = None
    for name in SHAP_FILES:
        (cfg.reports_dir / name).unlink(missing_ok=True)
    if shap:
        t0 = time.time()
        flagged = test_preds["gradient_boosting"][recommended or "unconstrained"] > 0
        shap_block = _shap(
            cfg,
            data,
            fitted["gradient_boosting"],
            calibrations["gradient_boosting"],
            test_feats,
            test_split["specs"],
            test_ids,
            flagged,
        )
        tick("SHAP", t0)

    headline_val = validation_by_model["gradient_boosting"]["unconstrained"]
    final_mask = _training_mask(data, np.ones(len(data.feats), bool), cfg, 1.0)
    rows_used = int((final_mask & known).sum())
    final = {
        "model": "gradient_boosting",
        "candidate": selection["candidate"],
        "target": selection["target"],
        "features": selection["features"],
        "weighted": selection["weighted"],
        "calibration": selection["calibration"],
        "decision": "lowest_expected_cost",
        "prior_correction": selection["calibration"] != "none",
        "weights": calibrations["gradient_boosting"].weights_.tolist(),
        "recommended_capacity": recommended,
        "recommended_capacity_reason": reason,
        "simulated_cost": sim_costs["unconstrained"],
        "validation_cost": headline_val["total_cost"],
    }
    metrics = {
        "project_version": __version__,
        "dataset": {
            "train_vehicles": data.train_vehicles,
            "validation_vehicles": int(len(data.y_val)),
            "test_vehicles": int(len(y_test)),
            "simulated_validation_vehicles": int(len(y_sim)),
            "labels": {**data.label_report, "rows_used_for_training": rows_used},
            "simulated_class_counts": np.bincount(y_sim, minlength=N_CLASSES).tolist(),
            "validation_class_counts": np.bincount(data.y_val, minlength=N_CLASSES).tolist(),
            "test_class_counts": np.bincount(y_test, minlength=N_CLASSES).tolist(),
            "n_features": len(data.names(tuple(selection["features"]))),
        },
        "settings": {
            "seed": cfg.seed,
            "change_window_readouts": cfg.change_window,
            "max_rows_per_vehicle": cfg.max_rows_per_vehicle,
            "selection_folds": cfg.selection_folds,
            "selection_row_fraction": cfg.selection_row_fraction,
            "boosting_params": cfg.boosting_params,
            "capacities": list(cfg.capacities),
            "capacity_tolerance": cfg.capacity_tolerance,
            "prior_weight_grid": list(cfg.prior_weight_grid),
            "bootstrap_samples": cfg.bootstrap_samples,
            "cost_matrix": COST_MATRIX.astype(int).tolist(),
            "class_rule": "time left t: class 4 if t <= 6, 3 if t <= 12, 2 if t <= 24, "
            "1 if t <= 48, else 0",
        },
        "tuning": tuning,
        "selection": selection,
        "calibration": {n: calibrations[n].info() for n in MODELS},
        "final_choice": final,
        "simulated": simulated,
        "validation_by_model": validation_by_model,
        "validation": headline_val,
        "capacity": {
            "rules": rules,
            "recommended": recommended,
            "reason": reason,
            "simulated_costs": sim_costs,
        },
        "test": {"final": by_model["gradient_boosting"], "by_model": by_model, "by_rule": by_rule},
        "bootstrap_test": boot,
        "shap": shap_block,
        "timings_seconds": timings,
    }

    preds = pd.DataFrame(
        {
            "vehicle_id": test_ids,
            "true_class": y_test,
            "predicted_class": test_preds["gradient_boosting"]["unconstrained"],
        }
    )
    probs = test_probs["gradient_boosting"]
    for k in range(N_CLASSES):
        preds[f"p_class_{k}"] = np.round(probs[:, k], 5)
    preds["at_risk_probability"] = np.round(np.clip(at_risk_probability(probs), 0, 1), 5)
    preds["model_score"] = np.round(np.clip(test_score["gradient_boosting"], 0, 1), 8)
    for r in rules[1:]:
        preds[f"predicted_class_{r}"] = test_preds["gradient_boosting"][r]
    preds.to_csv(cfg.reports_dir / "test_predictions.csv", index=False)
    _capacity_table(simulated, validation_by_model, by_rule, always, rules).to_csv(
        cfg.reports_dir / "capacity.csv", index=False
    )
    (cfg.reports_dir / "validation_decision_search.csv").unlink(missing_ok=True)

    joblib.dump(
        {
            "models": {n: fitted[n].model for n in MODELS},
            "fleet": {n: fitted[n].fleet for n in MODELS},
            "calibrations": calibrations,
            "spec_encoder": data.encoder,
            "final_choice": final,
            "config": asdict(cfg),
        },
        cfg.models_dir / "model.joblib",
        compress=3,
    )

    _plot_confusion(cfg, by_rule["gradient_boosting"], recommended)
    _plot_results(cfg, by_rule, always, boot, y_test, test_probs, rules)
    metrics["run_time_seconds"] = round(time.time() - start, 1)
    with open(cfg.reports_dir / "metrics.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    return metrics


def _shap(cfg, data, fitted, calibration, test_feats, test_specs, test_ids, flagged) -> dict:
    """SHAP explanations for every test vehicle.

    SHAP for every test vehicle: the at-risk log-odds of a five-class model (prior shift included),
    or the 48-step log-odds of a horizon model.
    """
    X = fitted.matrix(test_feats, test_specs)
    model = fitted.model
    if isinstance(model, HorizonModel):
        values, base, info = tree_shap(model.models_[48], X)
        output = model.models_[48].decision_function(X)
        info["output"] = "log-odds of failing within 48 time steps (48-step horizon model)"
    else:
        values, base, info, output = risk_shap(model.model, X, calibration.weights_)
    info["explained_vehicles"] = "all test vehicles"
    info["flagged_by"] = "recommended capacity"
    return save_explanations(
        cfg.reports_dir,
        values,
        base,
        info,
        X,
        test_ids,
        "vehicle_id",
        data.names(fitted.groups),
        "scania",
        cfg.change_window,
        "Scania: what drives the at-risk log-odds (test vehicles)",
        model_output=output,
        flagged=flagged,
    )


def explain_saved(cfg: ScaniaConfig, log=print) -> dict:
    """Recompute SHAP from the saved model and update metrics.json.

    Recompute SHAP from the saved model and update metrics.json (no retraining; test labels are not
    used).
    """
    bundle = joblib.load(cfg.models_dir / "model.joblib")
    metrics = json.loads((cfg.reports_dir / "metrics.json").read_text())
    data = Data(cfg, log)
    test_split = load_split(cfg.data_dir, "test", cfg.use_cache)
    test_feats, _ = _vehicle_set(test_split, cfg)
    setup = _candidate(metrics["selection"])
    fitted = FittedCandidate(
        setup, bundle["models"]["gradient_boosting"], bundle["fleet"]["gradient_boosting"], data
    )
    preds = pd.read_csv(cfg.reports_dir / "test_predictions.csv")
    rec = metrics["capacity"]["recommended"]
    column = "predicted_class" if rec is None else f"predicted_class_{rec}"
    metrics["shap"] = _shap(
        cfg,
        data,
        fitted,
        bundle["calibrations"]["gradient_boosting"],
        test_feats,
        test_split["specs"],
        test_feats["vehicle_id"].to_numpy(),
        preds[column].to_numpy() > 0,
    )
    with open(cfg.reports_dir / "metrics.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    return metrics


def _capacity_table(simulated, validation, test, always, rules) -> pd.DataFrame:
    rows = []
    for set_name, block in (("simulated", simulated), ("validation", validation), ("test", test)):
        for model, by_rule in block.items():
            for r in rules:
                s = by_rule[r]
                ar = s["at_risk"]
                rows.append(
                    {
                        "set": set_name,
                        "model": model,
                        "rule": r,
                        "cost": s["total_cost"],
                        "always_class_0_cost": s["always_class_0_cost"],
                        "recall": ar["recall"],
                        "precision": ar["precision"],
                        "false_alarms": ar["false_alarms"],
                        "flagged_share": ar["flagged_share"],
                        "healthy_flagged_share": ar["healthy_flagged_share"],
                    }
                )
    ar = always["at_risk"]
    rows.append(
        {
            "set": "test",
            "model": "always_class_0",
            "rule": "unconstrained",
            "cost": always["total_cost"],
            "always_class_0_cost": always["total_cost"],
            "recall": ar["recall"],
            "precision": ar["precision"],
            "false_alarms": ar["false_alarms"],
            "flagged_share": 0.0,
            "healthy_flagged_share": 0.0,
        }
    )
    return pd.DataFrame(rows)


def _rule_label(rule) -> str:
    return "unlimited" if rule in (None, "unconstrained") else f"{rule.split('_')[1]}% capacity"


def _plot_confusion(cfg, headline: dict, recommended) -> None:
    panels = [("unconstrained", headline["unconstrained"])]
    if recommended:
        panels.append((recommended, headline[recommended]))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6), squeeze=False)
    for ax, (rule, s) in zip(axes[0], panels, strict=True):
        cm = np.array(s["confusion_matrix"])
        ax.imshow(np.log1p(cm), cmap="Blues")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 3 else "#1a202c",
                    fontsize=9,
                )
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(
            f"Test, {_rule_label(rule)}: cost {s['total_cost']:,.0f} "
            f"(always class 0: {s['always_class_0_cost']:,.0f})",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(cfg.reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_results(cfg, by_rule, always, boot, y_test, probs, rules) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {
        "gradient_boosting": "#2b6cb0",
        "logistic_regression": "#2f855a",
        "searched_correction": "#a0aec0",
    }
    x = np.arange(len(rules))
    width = 0.27
    ax = axes[0]
    for i, n in enumerate(MODELS):
        costs = [by_rule[n][r]["total_cost"] for r in rules]
        ax.bar(x + (i - 1) * width, costs, width, color=colors[n], label=LABELS[n])
    ax.axhline(always["total_cost"], color="#c53030", ls="--", lw=1, label="Always class 0")
    ax.set_xticks(x, [_rule_label(r) for r in rules], fontsize=8)
    ax.set_ylabel("Official total cost (lower is better)")
    ax.set_title("Test vehicles: cost by workshop capacity", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for n in MODELS:
        ax.plot(
            [_rule_label(r) for r in rules],
            [100 * by_rule[n][r]["at_risk"]["healthy_flagged_share"] for r in rules],
            marker="o",
            color=colors[n],
            label=LABELS[n],
        )
    ax.set_ylabel("Healthy vehicles flagged (%)")
    ax.set_title("Test vehicles: false alarms by workshop capacity", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    risk = y_test > 0
    for n in MODELS:
        p, r, _ = precision_recall_curve(risk, np.clip(at_risk_probability(probs[n]), 0, 1))
        auc = by_rule[n]["unconstrained"]["at_risk"]["pr_auc"]
        ax.plot(r, p, color=colors[n], lw=2, label=f"{LABELS[n]} (PR-AUC {auc:.3f})")
    ax.axhline(risk.mean(), color="#718096", ls=":", label=f"base rate {risk.mean():.3f}")
    ax.set_xlabel("Recall (at-risk vehicles found)")
    ax.set_ylabel("Precision")
    ax.set_title("Test vehicles: at risk (class 1-4) vs class 0", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(cfg.reports_dir / "results.png", dpi=150)
    plt.close(fig)


def print_summary(m: dict) -> None:
    """Print the headline numbers of a run."""
    d, f, t = m["dataset"], m["final_choice"], m["test"]
    print("=== Scania Component X ===")
    print(
        f"Vehicles: {d['train_vehicles']:,} train ({d['simulated_validation_vehicles']:,} in the "
        f"simulated validation set), {d['validation_vehicles']:,} validation, "
        f"{d['test_vehicles']:,} test"
    )
    print(
        f"Headline: {f['candidate']} | calibration {f['calibration']} | "
        f"simulated cost {f['simulated_cost']:,.0f} | official validation cost "
        f"{f['validation_cost']:,.0f}"
    )
    print(
        f"Recommended capacity: {_rule_label(f['recommended_capacity'])} "
        f"({f['recommended_capacity_reason']})"
    )
    boot = m["bootstrap_test"]
    print("Test set (scored once):")
    print(
        f"  {'always class 0':<22}{'':<16} cost "
        f"{t['by_model']['always_class_0']['total_cost']:>8,.0f}"
    )
    for n in MODELS:
        for r in m["capacity"]["rules"]:
            s = t["by_rule"][n][r]
            ar = s["at_risk"]
            b = boot["by_rule"][r]["models"][n]
            print(
                f"  {LABELS[n]:<22}{_rule_label(r):<16} cost {s['total_cost']:>8,.0f} "
                f"[{b['total_cost_95'][0]:,.0f}, {b['total_cost_95'][1]:,.0f}] | saving "
                f"[{b['saving_vs_always_class_0_95'][0]:,.0f}, "
                f"{b['saving_vs_always_class_0_95'][1]:,.0f}] | recall {ar['recall']:.3f}, "
                f"precision {(ar['precision'] or 0):.3f}, false alarms {ar['false_alarms']:,}, "
                f"healthy flagged {ar['healthy_flagged_share']:.1%}"
            )
    for r in m["capacity"]["rules"]:
        iv = boot["by_rule"][r]["searched_correction_minus_gradient_boosting"]
        print(
            f"  searched correction minus headline, {_rule_label(r)}: 95% interval "
            f"[{iv[0]:,.0f}, {iv[1]:,.0f}]"
        )
    print(f"Run time: {m['run_time_seconds']:.0f}s")


def main(argv=None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Scania Component X pipeline")
    parser.add_argument("--no-tune", action="store_true", help="Skip the tuning pass (faster)")
    parser.add_argument(
        "--select",
        action="store_true",
        help="Rerun model selection even if reports/scania/selection.json exists",
    )
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP explanations")
    parser.add_argument(
        "--explain-only",
        action="store_true",
        help="Recompute SHAP from the saved model and reports, without retraining",
    )
    parser.add_argument(
        "--max-rows-per-vehicle",
        type=int,
        default=None,
        help="Use at most this many training rows per vehicle",
    )
    args = parser.parse_args(argv)
    cfg = ScaniaConfig()
    if args.max_rows_per_vehicle is not None:
        cfg = replace(cfg, max_rows_per_vehicle=args.max_rows_per_vehicle)
    if args.explain_only:
        explain_saved(cfg)
        write_summary(cfg.reports_dir.parent, ROOT / "README.md")
        print(f"SHAP recomputed from the saved model: {cfg.reports_dir}")
        return
    m = run(cfg, tune=not args.no_tune, select=args.select, shap=not args.no_shap)
    print_summary(m)
    print(f"Saved reports to {cfg.reports_dir}")
    readme = ROOT / "README.md"
    write_summary(cfg.reports_dir.parent, readme)
    print(f"Comparison table and README sections rebuilt from saved metrics: {readme}")


if __name__ == "__main__":
    main()
