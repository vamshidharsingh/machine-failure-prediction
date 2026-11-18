"""NASA C-MAPSS pipeline.

    python -m mfp.pipeline

Runs all four C-MAPSS subsets (FD001-FD004) by default; use --subset to pick.

Steps per subset: load and validate -> labels -> three-way engine split (fit, validation, lifecycle
test) -> operating-regime normalisation -> features -> (optional) tuning -> train gradient boosting
(the headline model) and logistic regression (the baseline) -> for each, choose the alarm policy
(threshold and confirm cycles) on validation engines -> simulate the alarm on lifecycle test engines
-> standard benchmark on the NASA test set -> save reports. After all subsets, reports/summary.json
and the README results section are rebuilt from the saved metrics.
"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402

from . import __version__  # noqa: E402
from .config import ALL_SUBSETS, ROOT, Config  # noqa: E402
from .data import load_cmapss  # noqa: E402
from .explain import save_explanations, tree_shap  # noqa: E402
from .features import (  # noqa: E402
    add_label,
    add_test_rul,
    add_train_rul,
    build_features,
    feature_columns,
    last_cycle_rows,
)
from .lifecycle import best_policy, lifecycle_sweep, price, simulate, summarize  # noqa: E402
from .models import (  # noqa: E402
    make_baseline,
    make_boosting,
    make_rul_baseline,
    make_rul_regressor,
    nasa_score,
    pr_auc,
    tune_boosting,
)
from .policy import confusion_counts  # noqa: E402
from .regimes import RegimeNormalizer, informative_normalised  # noqa: E402
from .report import write_summary  # noqa: E402
from .split import three_way_split  # noqa: E402

HEADLINE = "gradient_boosting"
BASELINE = "logistic_regression"


def _num(x):
    """JSON-safe number: NaN becomes null."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _clean(d: dict) -> dict:
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in d.items()}


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _timeline(feats: pd.DataFrame, scores, set_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "set": set_name,
            "unit": feats["unit"].astype(int).to_numpy(),
            "cycle": feats["cycle_id"].astype(int).to_numpy(),
            "failure_probability": np.round(scores, 4),
            "rul": feats["rul"].astype(int).to_numpy(),
        }
    )


def _policy_result(timeline, threshold, confirm, min_lead, max_lead, cost_miss, cost_false) -> dict:
    outcomes = simulate(timeline, threshold, confirm, min_lead, max_lead)
    return _clean(
        {
            "threshold": float(threshold),
            "confirm_cycles": int(confirm),
            **summarize(outcomes, cost_miss, cost_false),
        }
    )


def _compare_policies(
    timeline, threshold, confirm, min_lead, max_lead, cost_miss, cost_false
) -> dict:
    """The chosen policy next to two fixed rules on the same engines.

    The chosen policy against two fixed rules on the same engines:
    - fixed 0.5: rings the first cycle the probability reaches 0.5
    - fixed 0.5, same confirm cycles: separates the effect of the threshold
      from the effect of confirming over several cycles.
    """
    args = (min_lead, max_lead, cost_miss, cost_false)
    chosen = _policy_result(timeline, threshold, confirm, *args)
    fixed = _policy_result(timeline, 0.5, 1, *args)
    fixed_k = _policy_result(timeline, 0.5, confirm, *args)
    return {
        "chosen": chosen,
        "fixed_0.5": fixed,
        "fixed_0.5_same_confirm": fixed_k,
        "cost_saving_vs_0.5": fixed["cost"] - chosen["cost"],
        "cost_saving_vs_0.5_same_confirm": fixed_k["cost"] - chosen["cost"],
    }


def _alarm_counts(y_true, scores, threshold) -> dict:
    """Precision, recall and false alarms when alerting at score >= threshold."""
    c = confusion_counts(y_true, scores, threshold)
    alerts = c["true_alerts"] + c["false_alarms"]
    positives = c["true_alerts"] + c["missed_failures"]
    return {
        "threshold": float(threshold),
        "precision": _num(c["true_alerts"] / alerts) if alerts else None,
        "recall": _num(c["true_alerts"] / positives) if positives else None,
        "false_alarms": c["false_alarms"],
        "true_alerts": c["true_alerts"],
        "missed_failures": c["missed_failures"],
    }


def _benchmark(y_last, last_scores, y_all, all_scores, threshold) -> tuple:
    """NASA test benchmark for one model.

    NASA test set: last-cycle benchmark at 0.5 and at the chosen threshold, and PR-AUC over all
    cycles.
    """
    last = {
        "pr_auc": _num(pr_auc(y_last, last_scores)),
        "at_0.5": _alarm_counts(y_last, last_scores, 0.5),
        "at_chosen_threshold": _alarm_counts(y_last, last_scores, threshold),
    }
    return last, _num(pr_auc(y_all, all_scores))


def run(cfg: Config, tune: bool = True, update_docs: bool = True) -> dict:
    """Run one subset and write its outputs to cfg.reports_dir and cfg.models_dir."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate
    train_raw, test_raw, test_rul = load_cmapss(cfg.data_dir, cfg.subset)

    # 2. Labels
    train_lab = add_label(add_train_rul(train_raw), cfg.failure_window)
    test_lab = add_label(add_test_rul(test_raw, test_rul), cfg.failure_window)

    # 3. Three-way split by whole engine. Everything below is fitted or chosen
    #    on fit and validation engines only.
    fit_raw, val_raw, life_raw = three_way_split(
        train_lab, cfg.val_fraction, cfg.lifecycle_fraction, cfg.seed
    )

    # 4. Operating-regime normalisation, fitted on fit engines only and
    #    applied unchanged to every other set.
    normalizer = RegimeNormalizer(cfg.n_regimes, cfg.seed).fit(fit_raw)
    fit_n, val_n, life_n, test_n = (
        normalizer.transform(d) for d in (fit_raw, val_raw, life_raw, test_lab)
    )
    sensors = informative_normalised(fit_n)

    fit = build_features(fit_n, sensors, cfg.rolling_window)
    val = build_features(val_n, sensors, cfg.rolling_window)
    life = build_features(life_n, sensors, cfg.rolling_window)
    test = build_features(test_n, sensors, cfg.rolling_window)
    cols = feature_columns(fit)

    X_fit, y_fit = fit[cols].to_numpy(), fit["label"].to_numpy()
    X_val, y_val = val[cols].to_numpy(), val["label"].to_numpy()

    # 5. Optional tuning pass for gradient boosting (grouped CV on fit engines)
    tuning = None
    boost_params = {}
    if tune:
        boost_params, tuning_rows = tune_boosting(
            X_fit, y_fit, fit["unit"].to_numpy(), cfg.tuning_grid, cfg.tuning_folds, cfg.seed
        )
        tuning = {"best_params": boost_params, "results": tuning_rows}
        pd.DataFrame(tuning_rows).to_csv(cfg.reports_dir / "tuning_results.csv", index=False)

    # 6. Gradient boosting is the headline model on every subset; logistic
    #    regression is the baseline in every comparison. Both are fitted on
    #    fit engines and compared on validation engines.
    makers = {
        HEADLINE: lambda: make_boosting(cfg.seed, **boost_params),
        BASELINE: lambda: make_baseline(cfg.seed),
    }
    val_scores = {
        name: make().fit(X_fit, y_fit).predict_proba(X_val)[:, 1] for name, make in makers.items()
    }
    val_pr = {name: pr_auc(y_val, s) for name, s in val_scores.items()}

    # 7. Alarm policy per model, from validation engines only: sweep threshold
    #    x confirm cycles through the lifecycle simulation, keep the cheapest.
    cost_false = cfg.cost_false_alarm
    cost_miss = cfg.cost_ratio_main * cost_false
    max_lead = cfg.alert_max_lead
    val_timelines = {name: _timeline(val, s, "validation") for name, s in val_scores.items()}

    def choose(name, min_lead, ratio=cfg.cost_ratio_main):
        """Validation sweep and the cheapest policy for one model and minimum lead."""
        sweep = lifecycle_sweep(val_timelines[name], min_lead, max_lead, cfg.confirm_cycles_options)
        t, k = best_policy(sweep, ratio * cost_false, cost_false)
        return sweep, t, k

    life_sweep, threshold, confirm = choose(HEADLINE, cfg.alert_min_lead)
    life_sweep.to_csv(cfg.reports_dir / "validation_lifecycle_sweep.csv", index=False)
    lr_sweep, lr_threshold, lr_confirm = choose(BASELINE, cfg.alert_min_lead)

    # Warning-time option: one validation sweep (and policy) per minimum lead.
    warning_sweeps = {ml: choose(HEADLINE, ml) for ml in cfg.alert_min_lead_options}
    pd.concat(
        [s.assign(alert_min_lead=ml) for ml, (s, _, _) in warning_sweeps.items()],
        ignore_index=True,
    )[["alert_min_lead", *life_sweep.columns]].to_csv(
        cfg.reports_dir / "validation_lifecycle_sweeps.csv", index=False
    )

    # 8. Refit both models on fit + validation engines (never lifecycle test
    #    engines), then score lifecycle test and NASA test engines once.
    full = pd.concat([fit, val], ignore_index=True)
    X_full, y_full = full[cols].to_numpy(), full["label"].to_numpy()
    finals = {name: make().fit(X_full, y_full) for name, make in makers.items()}
    final = finals[HEADLINE]

    test_last = last_cycle_rows(test)
    life_scores, test_scores, last_scores = {}, {}, {}
    for name, model in finals.items():
        life_scores[name] = model.predict_proba(life[cols].to_numpy())[:, 1]
        test_scores[name] = model.predict_proba(test[cols].to_numpy())[:, 1]
        last_scores[name] = model.predict_proba(test_last[cols].to_numpy())[:, 1]
    life_timelines = {name: _timeline(life, s, "lifecycle_test") for name, s in life_scores.items()}
    life_timeline = life_timelines[HEADLINE]
    pd.concat(
        [life_timeline, _timeline(test, test_scores[HEADLINE], "nasa_test")], ignore_index=True
    ).to_csv(cfg.reports_dir / "engine_timelines.csv", index=False)

    # 9. Lifecycle comparison on lifecycle test engines.
    min_lead = cfg.alert_min_lead
    lifecycle_test = {
        "cost_ratio": cfg.cost_ratio_main,
        **_compare_policies(
            life_timeline, threshold, confirm, min_lead, max_lead, cost_miss, cost_false
        ),
    }
    sensitivity = []
    for ratio in cfg.cost_ratios_sensitivity:
        _, t, k = choose(HEADLINE, min_lead, ratio)
        sensitivity.append(
            {
                "cost_ratio": ratio,
                **_compare_policies(
                    life_timeline, t, k, min_lead, max_lead, ratio * cost_false, cost_false
                ),
            }
        )
    warning_time = []
    for ml, (sweep, t, k) in warning_sweeps.items():
        vb = price(sweep, cost_miss, cost_false)
        vb = vb[(vb["threshold"] == t) & (vb["confirm_cycles"] == k)].iloc[0]
        warning_time.append(
            {
                "alert_min_lead": ml,
                "alert_threshold": t,
                "confirm_cycles": k,
                "validation_cost": float(vb["cost"]),
                **_compare_policies(life_timeline, t, k, ml, max_lead, cost_miss, cost_false),
            }
        )

    def validation_block(sweep, t, k):
        """Validation outcome counts and cost of one policy."""
        vb = price(sweep, cost_miss, cost_false)
        vb = vb[(vb["threshold"] == t) & (vb["confirm_cycles"] == k)].iloc[0]
        return {
            "cost": float(vb["cost"]),
            "good_catches": int(vb["good_catches"]),
            "early_alarms": int(vb["early_alarms"]),
            "missed": int(vb["missed"]),
        }

    # 10. NASA test set: standard benchmark only
    y_last = test_last["label"].to_numpy()
    y_all = test["label"].to_numpy()
    test_last_cycle, test_all_pr = _benchmark(
        y_last, last_scores[HEADLINE], y_all, test_scores[HEADLINE], threshold
    )
    lr_last_cycle, lr_all_pr = _benchmark(
        y_last, last_scores[BASELINE], y_all, test_scores[BASELINE], lr_threshold
    )

    y_rul_full = full["rul"].clip(upper=cfg.rul_cap).to_numpy()
    reg = make_rul_regressor(cfg.seed).fit(X_full, y_rul_full)
    ridge = make_rul_baseline().fit(X_full, y_rul_full)
    rul_pred = reg.predict(test_last[cols].to_numpy())
    ridge_pred = ridge.predict(test_last[cols].to_numpy())
    true_rul = test_last["rul"].to_numpy()
    naive = np.full_like(true_rul, y_rul_full.mean(), dtype=float)

    metrics = {
        "project_version": __version__,
        "dataset": {
            "subset": cfg.subset,
            "n_regimes": cfg.n_regimes,
            "train_engines": int(train_raw["unit"].nunique()),
            "test_engines": int(test_raw["unit"].nunique()),
            "fit_engines": int(fit["unit"].nunique()),
            "validation_engines": int(val["unit"].nunique()),
            "lifecycle_test_engines": int(life["unit"].nunique()),
            "sensors_used": sensors,
            "n_features": len(cols),
            "test_positive_engines": int(y_last.sum()),
        },
        "settings": {
            "seed": cfg.seed,
            "failure_window_cycles": cfg.failure_window,
            "rolling_window_cycles": cfg.rolling_window,
            "rul_cap": cfg.rul_cap,
            "val_fraction": cfg.val_fraction,
            "lifecycle_fraction": cfg.lifecycle_fraction,
            "cost_false_alarm": cost_false,
            "cost_missed_failure": cost_miss,
            "cost_ratio_main": cfg.cost_ratio_main,
            "alert_min_lead": cfg.alert_min_lead,
            "alert_max_lead": cfg.alert_max_lead,
            "alert_min_lead_options": list(cfg.alert_min_lead_options),
            "confirm_cycles_options": list(cfg.confirm_cycles_options),
        },
        "model_selection": {
            "validation_pr_auc": {k: _num(v) for k, v in val_pr.items()},
            "chosen_model": HEADLINE,
            "rule": "gradient boosting is the headline model on every subset; "
            "logistic regression is the baseline in every comparison",
            "boosting_params": boost_params or "defaults",
        },
        "tuning": tuning,
        "alert_threshold": threshold,
        "confirm_cycles": confirm,
        "validation_lifecycle": validation_block(life_sweep, threshold, confirm),
        "lifecycle_test": lifecycle_test,
        "lifecycle_sensitivity": sensitivity,
        "warning_time": warning_time,
        "test_last_cycle": test_last_cycle,
        "test_all_cycles_pr_auc": test_all_pr,
        "baseline_logistic_regression": {
            "validation_pr_auc": _num(val_pr[BASELINE]),
            "alert_threshold": lr_threshold,
            "confirm_cycles": lr_confirm,
            "validation_lifecycle": validation_block(lr_sweep, lr_threshold, lr_confirm),
            "lifecycle_test": {
                "cost_ratio": cfg.cost_ratio_main,
                **_compare_policies(
                    life_timelines[BASELINE],
                    lr_threshold,
                    lr_confirm,
                    min_lead,
                    max_lead,
                    cost_miss,
                    cost_false,
                ),
            },
            "test_last_cycle": lr_last_cycle,
            "test_all_cycles_pr_auc": lr_all_pr,
        },
        "rul": {
            "rmse_model": _rmse(true_rul, rul_pred),
            "rmse_ridge": _rmse(true_rul, ridge_pred),
            "rmse_naive_mean": _rmse(true_rul, naive),
            "nasa_score_model": nasa_score(true_rul, rul_pred),
            "nasa_score_ridge": nasa_score(true_rul, ridge_pred),
            "nasa_score_naive_mean": nasa_score(true_rul, naive),
        },
    }

    # 11. SHAP: why the headline model rates each NASA test engine as it does
    #     (last recorded cycle).
    X_last = test_last[cols].to_numpy()
    values, base, info = tree_shap(final, X_last)
    metrics["shap"] = save_explanations(
        cfg.reports_dir,
        values,
        base,
        info,
        X_last,
        test_last["unit"].astype(int).to_numpy(),
        "unit",
        cols,
        "nasa",
        cfg.rolling_window,
        f"{cfg.subset}: what drives the failure risk (NASA test engines, last cycle)",
        model_output=final.decision_function(X_last),
    )

    with open(cfg.reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    preds = pd.DataFrame(
        {
            "unit": test_last["unit"].astype(int),
            "last_cycle": test_last["cycle_id"].astype(int),
            "failure_probability": np.round(last_scores[HEADLINE], 4),
            "at_risk_true": y_last,
            "true_rul": true_rul.astype(int),
            "predicted_rul": np.round(rul_pred, 1),
            "failure_probability_logistic": np.round(last_scores[BASELINE], 4),
            "predicted_rul_ridge": np.round(ridge_pred, 1),
        }
    )
    preds.to_csv(cfg.reports_dir / "test_predictions.csv", index=False)

    joblib.dump(
        {
            "classifier": final,
            "baseline_classifier": finals[BASELINE],
            "regressor": reg,
            "rul_baseline": ridge,
            "normalizer": normalizer,
            "features": cols,
            "sensors": sensors,
            "threshold": threshold,
            "confirm_cycles": confirm,
            "config": cfg,
        },
        cfg.models_dir / "model.joblib",
    )

    _plot(
        cfg, life_sweep, threshold, confirm, cost_miss, cost_false, true_rul, rul_pred, ridge_pred
    )

    if update_docs:
        write_summary(cfg.reports_dir.parent, cfg.reports_dir.parent.parent / "README.md")
    return metrics


def subset_config(cfg: Config, subset: str) -> Config:
    """Per-subset outputs: reports/<subset>/ and models/<subset>/."""
    return replace(
        cfg, subset=subset, reports_dir=cfg.reports_dir / subset, models_dir=cfg.models_dir / subset
    )


def run_all(
    cfg: Config, subsets=ALL_SUBSETS, tune: bool = True, readme: Path = None, on_done=None
) -> dict:
    """Run every subset, then rebuild the summary.

    Run each subset separately, then rebuild reports/summary.json (and the README section, if a
    README path is given) from every saved subset.
    """
    results = {}
    for subset in subsets:
        sub_cfg = subset_config(cfg, subset)
        results[subset] = run(sub_cfg, tune=tune, update_docs=False)
        if on_done is not None:
            on_done(sub_cfg, results[subset])
    write_summary(cfg.reports_dir, readme)
    return results


def _plot(
    cfg, sweep, threshold, confirm, cost_miss, cost_false, true_rul, rul_pred, ridge_pred
) -> None:
    priced = price(sweep, cost_miss, cost_false)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    colors = ["#2b6cb0", "#2f855a", "#805ad5"]
    for color, (k, g) in zip(colors, priced.groupby("confirm_cycles"), strict=False):
        ax.plot(g["threshold"], g["cost"], color=color, lw=2, label=f"confirm {k} cycle(s)")
    ax.axvline(
        threshold, color="#c05621", ls="--", label=f"chosen = {threshold:.2f}, confirm {confirm}"
    )
    ax.axvline(0.5, color="#718096", ls=":", label="fixed 0.5")
    ax.set_title(
        f"{cfg.subset} validation engines: lifecycle cost by threshold "
        f"({cost_miss / cost_false:g}:1)"
    )
    ax.set_xlabel("Alarm threshold")
    ax.set_ylabel("Cost (missed x C_miss + early alarms x C_false)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    lim = max(float(np.max(true_rul)), float(np.max(rul_pred)), float(np.max(ridge_pred))) * 1.05
    ax.scatter(true_rul, ridge_pred, s=14, color="#a0aec0", alpha=0.7, label="ridge baseline")
    ax.scatter(true_rul, rul_pred, s=18, color="#2b6cb0", alpha=0.75, label="gradient boosting")
    ax.plot([0, lim], [0, lim], color="#718096", ls=":")
    ax.set_xlim(0, lim)
    ax.set_ylim(min(0, float(np.min(ridge_pred)) * 1.05), lim)
    ax.set_title(f"{cfg.subset} NASA test engines: predicted vs true remaining life")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel(f"Predicted RUL (cycles, trained on RUL capped at {cfg.rul_cap})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(cfg.reports_dir / "results.png", dpi=150)
    plt.close(fig)


def _f(x, fmt=".3f"):
    return "n/a" if x is None else format(x, fmt)


def _print_subset(cfg: Config, m: dict) -> None:
    d, lt = m["dataset"], m["lifecycle_test"]
    lr = m["baseline_logistic_regression"]

    def line(name, r):
        """One printed line for a lifecycle policy."""
        lead = r["median_lead_time"]
        return (
            f"  {name}: cost {r['cost']:.0f} | good catches {r['good_catches']}, "
            f"early alarms {r['early_alarms']}, missed {r['missed']} | precision "
            f"{_f(r['precision'])} | median lead {_f(lead, '.1f')}"
        )

    def bench(name, b):
        """One printed line for the NASA test benchmark."""
        c, z = b["at_chosen_threshold"], b["at_0.5"]
        return (
            f"  {name}: PR-AUC {_f(b['pr_auc'])} | at {c['threshold']:.2f}: precision "
            f"{_f(c['precision'])}, recall {_f(c['recall'])}, false alarms {c['false_alarms']}"
            f" | at 0.50: precision {_f(z['precision'])}, recall {_f(z['recall'])}, "
            f"false alarms {z['false_alarms']}"
        )

    vpr = m["model_selection"]["validation_pr_auc"]
    print(f"=== {d['subset']} ({d['n_regimes']} regime{'s' if d['n_regimes'] != 1 else ''}) ===")
    print(
        f"Engines: {d['fit_engines']} fit, {d['validation_engines']} validation, "
        f"{d['lifecycle_test_engines']} lifecycle test, {d['test_engines']} NASA test"
    )
    print(
        f"Validation PR-AUC: gradient boosting {_f(vpr[HEADLINE])}, "
        f"logistic regression {_f(vpr[BASELINE])}"
    )
    print(
        f"Alarm policy (validation, {m['settings']['cost_ratio_main']:g}:1): threshold "
        f"{m['alert_threshold']:.2f}, confirm {m['confirm_cycles']} cycle(s); logistic baseline "
        f"{lr['alert_threshold']:.2f}, confirm {lr['confirm_cycles']}"
    )
    print(f"Lifecycle test ({lt['chosen']['engines']} engines), gradient boosting:")
    print(line("cost-based policy       ", lt["chosen"]))
    print(line("fixed 0.5               ", lt["fixed_0.5"]))
    print(line("fixed 0.5, same confirm ", lt["fixed_0.5_same_confirm"]))
    print(line("logistic, cost-based    ", lr["lifecycle_test"]["chosen"]))
    print("Required warning time (policy re-chosen on validation for each):")
    for w in m["warning_time"]:
        c = w["chosen"]
        print(
            f"  >= {w['alert_min_lead']} cycles: threshold {w['alert_threshold']:.2f}, confirm "
            f"{w['confirm_cycles']} | cost {c['cost']:.0f} vs fixed 0.5 "
            f"{w['fixed_0.5']['cost']:.0f}"
            f" | good {c['good_catches']}, early {c['early_alarms']}, missed {c['missed']}"
        )
    print("NASA test, last cycle:")
    print(bench("gradient boosting  ", m["test_last_cycle"]))
    print(bench("logistic regression", lr["test_last_cycle"]))
    print(
        f"NASA test, all cycles PR-AUC: gradient boosting {_f(m['test_all_cycles_pr_auc'])}, "
        f"logistic {_f(lr['test_all_cycles_pr_auc'])}"
    )
    r = m["rul"]
    print(
        f"Remaining life: RMSE gradient boosting {r['rmse_model']:.1f}, ridge "
        f"{r['rmse_ridge']:.1f},"
        f" naive {r['rmse_naive_mean']:.1f} | NASA score gradient boosting "
        f"{r['nasa_score_model']:,.0f}, ridge {r['nasa_score_ridge']:,.0f}, "
        f"naive {r['nasa_score_naive_mean']:,.0f}"
    )
    print(f"Saved reports to {cfg.reports_dir}", flush=True)


def main(argv=None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Machine failure prediction pipeline")
    parser.add_argument(
        "--subset",
        nargs="+",
        choices=ALL_SUBSETS,
        default=list(ALL_SUBSETS),
        help="Subsets to run (default: all four)",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder with the C-MAPSS files")
    parser.add_argument("--no-tune", action="store_true", help="Skip the tuning pass (faster)")
    args = parser.parse_args(argv)

    cfg = Config()
    if args.data_dir is not None:
        cfg = replace(cfg, data_dir=args.data_dir)
    readme = ROOT / "README.md"
    run_all(cfg, args.subset, tune=not args.no_tune, readme=readme, on_done=_print_subset)
    print(f"Summary saved to {cfg.reports_dir / 'summary.json'}")
    print(f"README results section rebuilt from saved metrics: {readme}")


if __name__ == "__main__":
    main()
