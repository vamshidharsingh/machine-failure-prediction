"""Generated README sections, summary.json and comparison.csv.

Write the generated sections of README.md, reports/summary.json and reports/comparison.csv from
saved metrics.

The README numbers are generated, never typed by hand, so they always match the saved metrics. Run
on its own to rebuild from the saved files:

python -m mfp.report
"""

import json
import sys
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _f(x, fmt=".1f", missing="n/a") -> str:
    return missing if x is None else format(x, fmt)


# Two scores closer than these are reported as "close".
CLOSE_PR_AUC = 0.01
CLOSE_COST = 2


def _model_name(m: dict) -> str:
    return m["model_selection"]["chosen_model"].replace("_", " ")


def _regimes(m: dict) -> str:
    n = m["dataset"].get("n_regimes", 1)
    return f"{n} operating regime{'s' if n != 1 else ''}"


def _verdict(saving: float) -> str:
    if saving > 0:
        return "Yes"
    if saving < 0:
        return "No"
    return "Tie"


def _compare(gb, lr, close, higher_is_better=True, fmt=".3f") -> str:
    """'0.951 vs 0.948 (close)' style comparison, boosting first."""
    if gb is None or lr is None:
        return f"{_f(gb, fmt)} vs {_f(lr, fmt)}"
    diff = gb - lr if higher_is_better else lr - gb
    if abs(gb - lr) < close:
        note = "close"
    elif diff > 0:
        note = "boosting better"
    else:
        note = "logistic better"
    return f"{format(gb, fmt)} vs {format(lr, fmt)} ({note})"


def _alarm_cell(a: dict) -> str:
    return f"{_f(a['precision'], '.2f')} / {_f(a['recall'], '.2f')} / {a['false_alarms']}"


def results_markdown(m: dict, heading_level: int = 3) -> str:
    """Details for one subset."""
    d, s, sel = m["dataset"], m["settings"], m["model_selection"]
    lt = m["lifecycle_test"]
    c, f = lt["chosen"], lt["fixed_0.5"]
    fk = lt.get("fixed_0.5_same_confirm", f)
    vpr = sel["validation_pr_auc"]
    lr = m.get("baseline_logistic_regression")
    lines = [
        f"{'#' * heading_level} {d['subset']} ({_regimes(m)})",
        "",
        f"Model: **{_model_name(m)}**, the headline model on every subset; logistic regression "
        f"is the baseline. Engines: {d['fit_engines']} fit, {d['validation_engines']} validation, "
        f"{d['lifecycle_test_engines']} lifecycle test, {d['test_engines']} NASA test.",
        "",
    ]
    if lr is not None:
        lc = lr["lifecycle_test"]["chosen"]
        val_pr = _compare(vpr["gradient_boosting"], vpr["logistic_regression"], CLOSE_PR_AUC)
        test_pr = _compare(
            m["test_last_cycle"]["pr_auc"], lr["test_last_cycle"]["pr_auc"], CLOSE_PR_AUC
        )
        cost = _compare(c["cost"], lc["cost"], CLOSE_COST, higher_is_better=False, fmt=".0f")
        lines += [
            'Gradient boosting vs logistic regression ("close" = within '
            f"{CLOSE_PR_AUC} PR-AUC or {CLOSE_COST} cost units):",
            "",
            f"- Validation PR-AUC: {val_pr}",
            f"- NASA test PR-AUC (last cycle): {test_pr}",
            f"- Lifecycle test cost, each with its own validation-chosen policy: {cost}",
            "",
        ]
    lines += [
        f"Alarm policy chosen on validation engines at {s['cost_ratio_main']:g}:1: ring when the "
        f"failure probability is at least **{m['alert_threshold']:.2f}** for "
        f"**{m['confirm_cycles']}** consecutive cycle(s).",
        "",
        f"Lifecycle test: {c['engines']} engines run to failure. A good catch rings "
        f"{s['alert_min_lead']} to {s['alert_max_lead']} cycles before failure. Precision = good "
        "catches / alarms that rang; early alarms are the false alarms.",
        "",
        "| | Cost-based policy | Fixed 0.5 threshold | Fixed 0.5, same confirm cycles |",
        "|---|---|---|---|",
        f"| Threshold | {c['threshold']:.2f} | {f['threshold']:.2f} | {fk['threshold']:.2f} |",
        f"| Confirm cycles | {c['confirm_cycles']} | {f['confirm_cycles']} | "
        f"{fk['confirm_cycles']} |",
        f"| Good catches | {c['good_catches']} | {f['good_catches']} | {fk['good_catches']} |",
        f"| Early alarms (false alarms) | {c['early_alarms']} | {f['early_alarms']} | "
        f"{fk['early_alarms']} |",
        f"| Missed (never rang / too late) | {c['missed']} ({c['missed_never_rang']} / "
        f"{c['missed_too_late']}) | {f['missed']} ({f['missed_never_rang']} / "
        f"{f['missed_too_late']}) "
        f"| {fk['missed']} ({fk['missed_never_rang']} / {fk['missed_too_late']}) |",
        f"| Precision | {_f(c.get('precision'), '.2f')} | {_f(f.get('precision'), '.2f')} | "
        f"{_f(fk.get('precision'), '.2f')} |",
        f"| Cost | {c['cost']:.0f} | {f['cost']:.0f} | {fk['cost']:.0f} |",
        f"| Median lead time (cycles) | {_f(c['median_lead_time'])} | {_f(f['median_lead_time'])} "
        f"| {_f(fk['median_lead_time'])} |",
        f"| Mean lead time of good catches | {_f(c['mean_lead_time_good_catches'])} | "
        f"{_f(f['mean_lead_time_good_catches'])} | {_f(fk['mean_lead_time_good_catches'])} |",
        "",
    ]
    if lr is not None:
        lc = lr["lifecycle_test"]["chosen"]
        lines += [
            f"Logistic regression baseline with its own validation-chosen policy (threshold "
            f"{lr['alert_threshold']:.2f}, confirm {lr['confirm_cycles']}): cost {lc['cost']:.0f}, "
            f"good catches {lc['good_catches']}, early alarms {lc['early_alarms']}, missed "
            f"{lc['missed']}, precision {_f(lc.get('precision'), '.2f')}.",
            "",
        ]
    sens = "; ".join(
        f"{r['cost_ratio']:g}:1 → {r['chosen']['cost']:.0f} vs {r['fixed_0.5']['cost']:.0f}"
        for r in m.get("lifecycle_sensitivity", [])
    )
    if sens:
        lines += [
            f"Cost sensitivity (policy re-chosen on validation; cost-based vs fixed 0.5): {sens}.",
            "",
        ]
    warn = "; ".join(
        f"at least {w['alert_min_lead']} cycles → threshold {w['alert_threshold']:.2f}, confirm "
        f"{w['confirm_cycles']}, cost {w['chosen']['cost']:.0f} vs {w['fixed_0.5']['cost']:.0f}"
        for w in m.get("warning_time", [])
    )
    if warn:
        lines += [
            f"Required warning time (policy re-chosen on validation; cost-based vs fixed 0.5, "
            f"each judged with that warning window): {warn}.",
            "",
        ]

    tl = m["test_last_cycle"]
    if "at_0.5" in tl and lr is not None:
        ll = lr["test_last_cycle"]
        lines += [
            "NASA test benchmark (last recorded cycle of each engine; precision / recall / false "
            "alarms):",
            "",
            "| Model | PR-AUC (last cycle) | At its chosen threshold | At 0.5 | PR-AUC (all "
            "cycles) |",
            "|---|---|---|---|---|",
            f"| Gradient boosting | {_f(tl['pr_auc'], '.3f')} | "
            f"{tl['at_chosen_threshold']['threshold']:.2f}: "
            f"{_alarm_cell(tl['at_chosen_threshold'])} "
            f"| {_alarm_cell(tl['at_0.5'])} | {_f(m['test_all_cycles_pr_auc'], '.3f')} |",
            f"| Logistic regression | {_f(ll['pr_auc'], '.3f')} | "
            f"{ll['at_chosen_threshold']['threshold']:.2f}: "
            f"{_alarm_cell(ll['at_chosen_threshold'])} "
            f"| {_alarm_cell(ll['at_0.5'])} | {_f(lr['test_all_cycles_pr_auc'], '.3f')} |",
            "",
        ]
    else:
        lines += [
            f"NASA test benchmark: PR-AUC {_f(tl['pr_auc'], '.3f')} (last cycle), "
            f"{_f(m['test_all_cycles_pr_auc'], '.3f')} (all cycles).",
            "",
        ]
    r = m["rul"]
    if "nasa_score_model" in r:
        lines += [
            "Remaining useful life on NASA test engines (NASA score: lower is better; late "
            "predictions cost more):",
            "",
            "| Model | RMSE (cycles) | NASA score |",
            "|---|---|---|",
            f"| Gradient boosting | {r['rmse_model']:.1f} | {r['nasa_score_model']:,.0f} |",
            f"| Ridge regression | {r['rmse_ridge']:.1f} | {r['nasa_score_ridge']:,.0f} |",
            f"| Training mean (naive) | {r['rmse_naive_mean']:.1f} | "
            f"{r['nasa_score_naive_mean']:,.0f} |",
            "",
        ]
    else:
        lines += [
            f"Remaining useful life RMSE {r['rmse_model']:.1f} cycles (predicting the "
            f"training mean gives {r['rmse_naive_mean']:.1f}).",
            "",
        ]
    lines.append(f"![{d['subset']} results](reports/{d['subset']}/results.png)")
    return "\n".join(lines)


def build_summary(subsets: dict) -> dict:
    """Compact cross-subset summary (saved as reports/summary.json)."""
    out = {}
    for name, m in sorted(subsets.items()):
        lt = m["lifecycle_test"]
        lr = m.get("baseline_logistic_regression") or {}
        out[name] = {
            "n_regimes": m["dataset"].get("n_regimes", 1),
            "chosen_model": m["model_selection"]["chosen_model"],
            "validation_pr_auc": m["model_selection"]["validation_pr_auc"],
            "nasa_test_pr_auc_last_cycle": m["test_last_cycle"]["pr_auc"],
            "nasa_test_pr_auc_all_cycles": m["test_all_cycles_pr_auc"],
            "logistic_nasa_test_pr_auc_last_cycle": (lr.get("test_last_cycle") or {}).get("pr_auc"),
            "rul_rmse_model": m["rul"]["rmse_model"],
            "rul_rmse_ridge": m["rul"].get("rmse_ridge"),
            "rul_rmse_naive_mean": m["rul"]["rmse_naive_mean"],
            "rul_nasa_score_model": m["rul"].get("nasa_score_model"),
            "rul_nasa_score_ridge": m["rul"].get("nasa_score_ridge"),
            "alert_threshold": m["alert_threshold"],
            "confirm_cycles": m["confirm_cycles"],
            "lifecycle_test_engines": m["dataset"]["lifecycle_test_engines"],
            "lifecycle_chosen": lt["chosen"],
            "lifecycle_fixed_0.5": lt["fixed_0.5"],
            "lifecycle_fixed_0.5_same_confirm": lt.get("fixed_0.5_same_confirm"),
            "lifecycle_logistic_chosen": (lr.get("lifecycle_test") or {}).get("chosen"),
            "cost_saving_vs_0.5": lt["cost_saving_vs_0.5"],
            "cost_saving_vs_0.5_same_confirm": lt.get("cost_saving_vs_0.5_same_confirm"),
            "cost_based_policy_wins": _verdict(lt["cost_saving_vs_0.5"]),
        }
    first = next(iter(subsets.values())) if subsets else {}
    return {
        "project_version": first.get("project_version"),
        "settings": first.get("settings"),
        "subsets": out,
    }


def nasa_findings(subsets: dict) -> list:
    """NASA lead paragraph from the saved metrics.

    Lead paragraph: required warning time first, then where the default window's result comes from.
    Every number is read from the metrics.
    """
    rows, wins, losses, ties = [], [], [], []
    leads = sorted(
        {w["alert_min_lead"] for m in subsets.values() for w in m.get("warning_time", [])}
    )
    for name, m in sorted(subsets.items()):
        by_lead = {w["alert_min_lead"]: w for w in m.get("warning_time", [])}
        cells = []
        for ml in leads:
            w = by_lead.get(ml)
            if w is None:
                cells.append("n/a")
                continue
            c, f = w["chosen"]["cost"], w["fixed_0.5"]["cost"]
            cells.append(f"{c:.0f} / {f:.0f}")
            if ml > min(leads):
                (wins if c < f else losses if c > f else ties).append(
                    f"{name} at {ml} cycles ({c:.0f} vs {f:.0f})"
                )
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    if not rows or len(leads) < 2:
        return []
    longer = " or ".join(str(ml) for ml in leads[1:])
    n_cases = len(wins) + len(losses) + len(ties)
    first = (
        f"When at least {longer} cycles of warning are required, the cost-based policy is "
        f"cheaper than a fixed 0.5 threshold in {len(wins)} of {n_cases} cases"
    )
    if losses or ties:
        others = losses + ties
        first += (
            "; the exception is " if len(others) == 1 else "; the exceptions are "
        ) + ", ".join(others)
    first += (
        ". A fixed 0.5 threshold rings too late once more warning is needed, while the "
        "cost-based threshold is re-chosen on validation engines for each window."
    )
    default = min(leads)
    gain_threshold, gain_confirm = [], []
    for name, m in sorted(subsets.items()):
        lt = m["lifecycle_test"]
        c, f = lt["chosen"]["cost"], lt["fixed_0.5"]["cost"]
        k = (lt.get("fixed_0.5_same_confirm") or {}).get("cost")
        if k is None:
            continue
        gain_confirm.append(f"{name}: {c:.0f} / {f:.0f} / {k:.0f}")
        if c < k:
            gain_threshold.append(name)
    second = (
        f"At the default {default}-cycle window the picture is different. Cost of the "
        "cost-based policy / fixed 0.5 / fixed 0.5 with the same confirm cycles: "
        + "; ".join(gain_confirm)
        + ". "
        + (
            "Wherever the cost-based policy beats fixed 0.5, fixed 0.5 with the same confirm "
            "cycles does as well or better, so the gain comes from confirming over several "
            "cycles, not from the threshold."
            if not gain_threshold
            else "The threshold itself adds a gain only on " + ", ".join(gain_threshold) + "."
        )
    )
    return [
        "**Required warning time.** Lifecycle test cost, cost-based policy / fixed 0.5 threshold, "
        "each judged with that warning window:",
        "",
        "| Subset | " + " | ".join(f"At least {ml} cycles" for ml in leads) + " |",
        "|---|" + "---|" * len(leads),
        *rows,
        "",
        first,
        "",
        second,
        "",
    ]


def summary_markdown(subsets: dict) -> str:
    """Summary table across subsets, then short details for each."""
    s = next(iter(subsets.values()))["settings"]
    lines = [
        *nasa_findings(subsets),
        f"Gradient boosting is the headline model on every subset; logistic regression is the "
        f"baseline. Lifecycle costs assume a missed failure costs {s['cost_ratio_main']:g}x an "
        f"early "
        f"alarm, and that an alarm is useful {s['alert_min_lead']} to {s['alert_max_lead']} cycles "
        "before failure. Both are assumptions. Alarm policies are always chosen on validation "
        "engines; lifecycle test and NASA test engines are used only to report results.",
        "",
        "| Subset | Regimes | NASA test PR-AUC: boosting / logistic | RUL RMSE: boosting / "
        "ridge / naive "
        "| NASA score: boosting / ridge | Lifecycle test engines | Policy (threshold, confirm) "
        "| Lifecycle cost: cost-based / fixed 0.5 / fixed 0.5 same confirm | Logistic lifecycle "
        "cost "
        "| Cost-based beats fixed 0.5? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in sorted(subsets.items()):
        lt = m["lifecycle_test"]
        lr = m.get("baseline_logistic_regression") or {}
        r = m["rul"]
        fk = lt.get("fixed_0.5_same_confirm") or {}
        lr_pr = (lr.get("test_last_cycle") or {}).get("pr_auc")
        lr_cost = ((lr.get("lifecycle_test") or {}).get("chosen") or {}).get("cost")
        lines.append(
            f"| {name} | {m['dataset'].get('n_regimes', 1)} | "
            f"{_f(m['test_last_cycle']['pr_auc'], '.3f')} / {_f(lr_pr, '.3f')} | "
            f"{r['rmse_model']:.1f} / {_f(r.get('rmse_ridge'))} / {r['rmse_naive_mean']:.1f} | "
            f"{_f(r.get('nasa_score_model'), ',.0f')} / {_f(r.get('nasa_score_ridge'), ',.0f')} | "
            f"{m['dataset']['lifecycle_test_engines']} | "
            f"{m['alert_threshold']:.2f}, {m['confirm_cycles']} | "
            f"{lt['chosen']['cost']:.0f} / {lt['fixed_0.5']['cost']:.0f} / "
            f"{_f(fk.get('cost'), '.0f')} | "
            f"{_f(lr_cost, '.0f')} | "
            f"{_verdict(lt['cost_saving_vs_0.5'])} |"
        )
    shap_rows = [
        f"| {name} | "
        + " | ".join(t["readable_name"] for t in m["shap"]["top_features"][:3])
        + " |"
        for name, m in sorted(subsets.items())
        if m.get("shap")
    ]
    if shap_rows:
        lines += [
            "",
            "Top three SHAP features per subset (mean |SHAP| of gradient boosting's log-odds of "
            "failure, last cycle of every NASA test engine):",
            "",
            "| Subset | 1st | 2nd | 3rd |",
            "|---|---|---|---|",
            *shap_rows,
        ]
    charts = [
        f"![{name} SHAP summary](reports/{name}/shap_summary.png)"
        for name, m in sorted(subsets.items())
        if m.get("shap")
    ]
    if charts:
        lines += [
            "",
            "SHAP summary charts (mean |SHAP| per feature, NASA test engines at their last cycle):",
            "",
            *charts,
        ]
    lines += ["", "<details>", "<summary>Per-subset details</summary>", ""]
    for _, m in sorted(subsets.items()):
        lines += [results_markdown(m, heading_level=3), ""]
    lines.append("</details>")
    return "\n".join(lines)


def _markers(name: str) -> tuple:
    return f"<!-- {name}:START -->", f"<!-- {name}:END -->"


def replace_section(text: str, name: str, body: str):
    """Replace the text between <!-- NAME:START --> and <!-- NAME:END -->.

    Returns the new text, or None if the markers are missing.
    """
    start, end = _markers(name)
    if start not in text or end not in text:
        return None
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return f"{before}{start}\n{body}\n{end}{after}"


def update_readme(readme: Path, metrics: dict) -> bool:
    """Replace the README NASA results section.

    `metrics` is either one subset's metrics, or {"subsets": {name: metrics}}
    for the cross-subset summary.
    """
    readme = Path(readme)
    if not readme.exists():
        return False
    body = (
        summary_markdown(metrics["subsets"]) if "subsets" in metrics else results_markdown(metrics)
    )
    text = replace_section(readme.read_text(), "RESULTS", body)
    if text is None:
        return False
    readme.write_text(text)
    return True


def load_subset_metrics(reports_root: Path) -> dict:
    """Every reports/<subset>/metrics.json that exists, keyed by subset."""
    out = {}
    for path in sorted(Path(reports_root).glob("FD00*/metrics.json")):
        with open(path) as fh:
            out[path.parent.name] = json.load(fh)
    return out


def load_scania_metrics(reports_root: Path):
    """reports/scania/metrics.json, or None if it does not exist."""
    path = Path(reports_root) / "scania" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


SCANIA_NAMES = {
    "gradient_boosting": "Gradient boosting",
    "logistic_regression": "Logistic regression",
    "searched_correction": "Searched correction",
    "always_class_0": "Always class 0",
}
SCANIA_MODELS = ("gradient_boosting", "logistic_regression", "searched_correction")


def _interval(iv, fmt=",.0f") -> str:
    return f"[{format(iv[0], fmt)}, {format(iv[1], fmt)}]"


def _pct(x, digits=1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _rule(rule) -> str:
    return "unlimited" if rule in (None, "unconstrained") else f"{rule.split('_')[1]}% capacity"


def _calibration(kind: str) -> str:
    return {
        "none": "none",
        "prior_shift": "prior shift",
        "prior_shift_isotonic": "prior shift + isotonic",
        "searched_on_validation": "searched on validation",
    }[kind]


def _selection_rows(table, rows) -> list:
    lines = [
        "| Candidate | Calibration | Sim. PR-AUC | Sim. ROC-AUC | Sim. Brier | Sim. cost "
        "| Sim. healthy flagged | Sim. recall | Val. PR-AUC | Val. ROC-AUC | Val. Brier | Val. "
        "cost "
        "| Val. healthy flagged | Val. recall |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['candidate']} | {_calibration(r['calibration'])} | {r['sim_pr_auc']:.3f} | "
            f"{r['sim_roc_auc']:.3f} | {r['sim_brier']:.4f} | {r['sim_cost']:,.0f} | "
            f"{_pct(r['sim_healthy_flagged_share'])} | {_pct(r['sim_recall'])} | "
            f"{r['val_pr_auc']:.3f} | {r['val_roc_auc']:.3f} | {r['val_brier']:.4f} | "
            f"{r['val_cost']:,.0f} | {_pct(r['val_healthy_flagged_share'])} | "
            f"{_pct(r['val_recall'])} |"
        )
    return lines


def scania_markdown(m: dict, reports_root: Path = None) -> str:
    """Scania results, all from reports/scania/metrics.json and model_selection.csv."""
    import csv

    d, lab, f = m["dataset"], m["dataset"]["labels"], m["final_choice"]
    sel, cap = m["selection"], m["capacity"]
    by_rule, boot = m["test"]["by_rule"], m["bootstrap_test"]
    always = m["test"]["by_model"]["always_class_0"]
    rec = cap["recommended"]
    sim_set = sel["simulated_set"]
    lines = [
        f"Training data: {lab['labelled_readouts']:,} labelled readouts from "
        f"{lab['labelled_vehicles']:,} vehicles for the five-class model; the horizon models also "
        f"use censored readouts where their horizon is known. Simulated validation set: "
        f"{sim_set['vehicles']:,} training vehicles cut at a random readout "
        f"({sim_set['at_risk']} at "
        f"risk), predicted out-of-fold. Official validation: {d['validation_vehicles']:,} vehicles "
        f"({sum(d['validation_class_counts'][1:])} at risk). Test: {d['test_vehicles']:,} vehicles "
        f"({sum(d['test_class_counts'][1:])} at risk).",
        "",
        "#### Model selection",
        "",
        "Every candidate was scored on both evaluation sets with lowest-expected-cost decisions. "
        f"Candidates were compared on a seeded {sel.get('row_fraction', 1):.0%} of training rows; "
        "the chosen model was then trained on all rows. Each candidate is shown with its "
        "calibration that has the lowest simulated cost (calibrations searched on the official "
        "validation set are excluded from the choice).",
        "",
    ]
    rows = []
    if (
        reports_root is not None
        and (Path(reports_root) / "scania" / "model_selection.csv").exists()
    ):
        with open(Path(reports_root) / "scania" / "model_selection.csv") as fh:
            rows = [
                {
                    k: (float(v) if k.startswith(("sim_", "val_")) and v not in ("", "None") else v)
                    for k, v in r.items()
                }
                for r in csv.DictReader(fh)
            ]
    if rows:
        best = {}
        for r in rows:
            if r["calibration"] == "searched_on_validation":
                continue
            if r["candidate"] not in best or r["sim_cost"] < best[r["candidate"]]["sim_cost"]:
                best[r["candidate"]] = r
        lines += _selection_rows(None, sorted(best.values(), key=lambda r: r["sim_cost"]))
        lines += ["", "<details>", "<summary>Every candidate with every calibration</summary>", ""]
        lines += _selection_rows(None, rows)
        lines += ["", "</details>", ""]
    steps = "; ".join(
        f"{s['step']}: {'kept' if s['kept'] else 'not kept'} "
        f"({s['sim_cost_without']:,.0f} → {s['sim_cost_with']:,.0f})"
        for s in sel.get("steps", [])
    )
    if steps:
        lines += [
            f"Changes tried on the horizon gradient boosting model (simulated cost): {steps}.",
            "",
        ]
    agree = sel.get("official_validation_agrees")
    lines.append(
        f"Chosen: **{f['candidate']}** with **{_calibration(f['calibration'])}** calibration "
        f"(simulated cost {f['simulated_cost']:,.0f}, official validation cost "
        f"{f['validation_cost']:,.0f})."
        + (
            ""
            if agree
            else f" The official validation set prefers "
            f"{sel['official_validation_best']['candidate']} "
            f"with {_calibration(sel['official_validation_best']['calibration'])} calibration "
            f"(cost {sel['official_validation_best']['val_cost']:,.0f}); the simulated set has "
            f"{sim_set['vehicles'] / d['validation_vehicles']:.1f} times as many vehicles, so it "
            "decides."
        )
    )
    if f["target"] == "five_class":
        lines.append(
            "The horizon models did not beat the five-class model on simulated cost, so "
            "the five-class model is the headline."
        )
    lines += [
        "",
        "#### Workshop capacity",
        "",
        "The official cost matrix assumes unlimited workshop capacity. With a capacity of k%, only "
        "the riskiest k% of vehicles (by probability of failing within 48 steps) may be flagged; "
        "each gets the class 1-4 with the lowest expected cost, the rest class 0. Chosen model:",
        "",
        "| Capacity | Sim. cost | Sim. recall | Sim. healthy flagged | Val. cost | Val. recall "
        "| Val. healthy flagged |",
        "|---|---|---|---|---|---|---|",
    ]
    sim, val = m["simulated"]["gradient_boosting"], m["validation_by_model"]["gradient_boosting"]
    for r in cap["rules"]:
        lines.append(
            f"| {_rule(r)} | {sim[r]['total_cost']:,.0f} | {_pct(sim[r]['at_risk']['recall'])} | "
            f"{_pct(sim[r]['at_risk']['healthy_flagged_share'])} | {val[r]['total_cost']:,.0f} | "
            f"{_pct(val[r]['at_risk']['recall'])} | "
            f"{_pct(val[r]['at_risk']['healthy_flagged_share'])} |"
        )
    lines += [
        "",
        f"Recommended operating point: **{_rule(rec)}**, the {cap['reason']} on the "
        "simulated validation set.",
        "",
    ]

    lines += [
        "#### Test set (scored once)",
        "",
        f"95% intervals from {boot['samples']:,} bootstrap resamples of test vehicles.",
        "",
        "| Model | Capacity | Official cost (95% interval) | Saving vs always class 0 (95% "
        "interval) "
        "| Recall | Precision | False alarms | Healthy flagged | Macro F1 |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| Always class 0 | - | {always['total_cost']:,.0f} "
        f"{_interval(boot['always_class_0_cost_95'])} "
        f"| - | 0.0% | n/a | 0 | 0.0% | {always['macro_f1']:.3f} |",
    ]
    for n in SCANIA_MODELS:
        for r in cap["rules"]:
            s, b = by_rule[n][r], boot["by_rule"][r]["models"][n]
            ar = s["at_risk"]
            lines.append(
                f"| {SCANIA_NAMES[n]} | {_rule(r)} | {s['total_cost']:,.0f} "
                f"{_interval(b['total_cost_95'])} | {s['cost_saving_vs_always_class_0']:,.0f} "
                f"{_interval(b['saving_vs_always_class_0_95'])} | {_pct(ar['recall'])} | "
                f"{_pct(ar['precision'])} | {ar['false_alarms']:,} | "
                f"{_pct(ar['healthy_flagged_share'])} | {s['macro_f1']:.3f} |"
            )
    lines += [
        "",
        "Always class 0 has a higher macro F1 than the models because it is right about almost "
        "every healthy vehicle, and macro F1 does not price mistakes; the official cost does, and "
        "it is the metric that matters here.",
        "",
    ]
    head, prev = by_rule["gradient_boosting"], by_rule["searched_correction"]
    hu, pu = head["unconstrained"], prev["unconstrained"]
    diff = boot["by_rule"]["unconstrained"]["searched_correction_minus_gradient_boosting"]

    def clear(iv):
        """Plain-language verdict on a cost-difference interval."""
        return (
            "is clear: the 95% interval is above zero"
            if iv[0] > 0
            else "is not clear: the 95% interval includes zero"
            if iv[0] <= 0 <= iv[1]
            else "goes the other way: the searched-correction model is cheaper"
        )

    lines += ["In plain words:", ""]
    lines.append(
        f"- With unlimited capacity, the chosen model costs {hu['total_cost']:,.0f} against "
        f"{pu['total_cost']:,.0f} for the searched-correction model and "
        f"{always['total_cost']:,.0f} for always "
        f"class 0. The difference from the searched-correction model {clear(diff)} "
        f"({_interval(diff)})."
    )
    if rec:
        hr = head[rec]
        kept = (
            hr["cost_saving_vs_always_class_0"] / hu["cost_saving_vs_always_class_0"]
            if hu["cost_saving_vs_always_class_0"]
            else None
        )
        lines.append(
            f"- At the recommended {_rule(rec)}, it flags {hr['at_risk']['false_alarms']:,} "
            f"healthy "
            f"vehicles ({_pct(hr['at_risk']['healthy_flagged_share'])}) instead of "
            f"{pu['at_risk']['false_alarms']:,} "
            f"({_pct(pu['at_risk']['healthy_flagged_share'])}) for "
            f"the searched-correction model with unlimited capacity, and keeps "
            f"{_pct(kept, 0) if kept is not None else 'n/a'} of the unlimited-capacity saving "
            f"against always class 0 (cost {hr['total_cost']:,.0f}; recall "
            f"{_pct(hr['at_risk']['recall'])})."
        )
    shown = rec or "unconstrained"
    lines += [
        "",
        f"Test confusion matrix of the chosen model at {_rule(shown)} (rows: true class, columns: "
        "predicted class):",
        "",
        "| True \\ predicted | 0 | 1 | 2 | 3 | 4 |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(head[shown]["confusion_matrix"]):
        lines.append(f"| {i} | " + " | ".join(f"{v:,}" for v in row) + " |")
    lines += [
        "",
        "![Scania confusion matrices](reports/scania/confusion_matrix.png)",
        "",
        "![Scania results](reports/scania/results.png)",
    ]
    return "\n".join(lines)


def scania_shap_markdown(m: dict, reports_root: Path = None) -> str:
    """Scania SHAP section: method, additivity check and the two top-10 rankings."""
    s = m.get("shap")
    if not s:
        return "SHAP results appear here after `python -m mfp.scania.pipeline`."
    cap = m.get("capacity", {}).get("recommended")
    gap = s.get("additivity_max_error")
    lines = [
        f"Method: {s['method']}, explaining the {s['output']}, for all "
        f"{s['explained_rows']:,} test vehicles. Largest gap between a vehicle's contributions "
        f"plus the base value and its model output: {gap:.1e}."
        if gap is not None
        else f"Method: {s['method']}, explaining the {s['output']}, for all "
        f"{s['explained_rows']:,} test vehicles.",
        "",
    ]
    flagged = s.get("top_features_flagged")
    header = "| Rank | All test vehicles | Mean \\|SHAP\\| |"
    sep = "|---|---|---|"
    if flagged:
        header += f" Vehicles flagged at {_rule(cap)} ({s['flagged_rows']:,}) | Mean \\|SHAP\\| |"
        sep += "---|---|"
    lines += [header, sep]
    for i, t in enumerate(s["top_features"][:10]):
        row = f"| {i + 1} | {t['readable_name']} | {t['mean_abs_shap']:.3f} |"
        if flagged and i < len(flagged):
            row += f" {flagged[i]['readable_name']} | {flagged[i]['mean_abs_shap']:.3f} |"
        lines.append(row)
    lines += [
        "",
        "Mean |SHAP| is in log-odds of being at risk. SHAP shows what the model relies "
        "on, not what causes failures; the variables are anonymised.",
        "",
        "![Scania SHAP summary](reports/scania/shap_summary.png)",
    ]
    return "\n".join(lines)


COMPARISON_COLUMNS = [
    "dataset",
    "model",
    "pr_auc",
    "precision",
    "recall",
    "false_alarms",
    "official_cost",
    "lifecycle_cost",
    "rmse",
    "nasa_score",
]


SCANIA_COMPARED = ("gradient_boosting", "logistic_regression", "searched_correction")


def _scania_row(dataset: str, name: str, summary: dict) -> dict:
    ar = summary["at_risk"]
    return {
        "dataset": dataset,
        "model": name,
        "pr_auc": ar["pr_auc"],
        "precision": ar["precision"],
        "recall": ar["recall"],
        "false_alarms": ar["false_alarms"],
        "official_cost": summary["total_cost"],
        "lifecycle_cost": None,
        "rmse": None,
        "nasa_score": None,
    }


def comparison_rows(subsets: dict, scania) -> list:
    """One row per dataset and model. None where a metric does not apply.

    Scania is shown first at the recommended workshop capacity (with always
    class 0), then with unlimited capacity.
    """
    rows = []
    if scania:
        by = scania["test"]["by_model"]
        rec = (scania.get("capacity") or {}).get("recommended")
        by_rule = scania["test"].get("by_rule", {})
        if rec:
            label = f"Scania ({rec.split('_')[1]}% capacity)"
            rows += [
                _scania_row(label, n, by_rule[n][rec]) for n in SCANIA_COMPARED if n in by_rule
            ]
            if "always_class_0" in by:
                rows.append(_scania_row(label, "always_class_0", by["always_class_0"]))
            rows += [
                _scania_row("Scania (unlimited capacity)", n, by[n])
                for n in SCANIA_COMPARED
                if n in by
            ]
        else:
            rows += [
                _scania_row("Scania", n, by[n])
                for n in (*SCANIA_COMPARED, "always_class_0")
                if n in by
            ]
    for name, m in sorted(subsets.items()):
        lr = m.get("baseline_logistic_regression")
        r = m["rul"]
        blocks = [
            (
                "gradient_boosting",
                m["test_last_cycle"],
                m["lifecycle_test"],
                r["rmse_model"],
                r.get("nasa_score_model"),
            )
        ]
        if lr:
            blocks.append(
                (
                    "logistic_regression",
                    lr["test_last_cycle"],
                    lr["lifecycle_test"],
                    r.get("rmse_ridge"),
                    r.get("nasa_score_ridge"),
                )
            )
        for model, tl, lt, rmse, score in blocks:
            at = tl.get("at_chosen_threshold", {})
            rows.append(
                {
                    "dataset": name,
                    "model": model,
                    "pr_auc": tl["pr_auc"],
                    "precision": at.get("precision"),
                    "recall": at.get("recall"),
                    "false_alarms": at.get("false_alarms"),
                    "official_cost": None,
                    "lifecycle_cost": lt["chosen"]["cost"],
                    "rmse": rmse,
                    "nasa_score": score,
                }
            )
    return rows


def comparison_markdown(rows: list, scania=None, subsets=None) -> str:
    """The comparison table and the notes on empty cells."""
    if not rows:
        return "The comparison appears here after the pipelines have run."
    lines = [
        "| Dataset | Model | PR-AUC | Precision | Recall | False alarms | Official cost (Scania) "
        "| Lifecycle cost (NASA) | RMSE (NASA remaining life) | NASA score |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:

        def c(key, fmt, r=r):
            """Formatted cell, blank when the metric does not apply."""
            v = r[key]
            return "" if v is None else format(v, fmt)

        lines.append(
            f"| {r['dataset']} | {SCANIA_NAMES[r['model']]} | {c('pr_auc', '.3f')} | "
            f"{c('precision', '.3f')} | {c('recall', '.3f')} | {c('false_alarms', ',')} | "
            f"{c('official_cost', ',.0f')} | {c('lifecycle_cost', '.0f')} | {c('rmse', '.1f')} | "
            f"{c('nasa_score', ',.0f')} |"
        )
    lines += [
        "",
        "Empty cells, and what each column means:",
        "",
        "- **PR-AUC, precision, recall, false alarms.** Scania: the at-risk view (classes 1-4 vs "
        "class 0) on test vehicles, first at the recommended workshop capacity (only the riskiest "
        "share of vehicles may be flagged), then with unlimited capacity. NASA: the last "
        "recorded cycle of each NASA test engine, alerting at the model's validation-chosen "
        "threshold. Always class 0 has no score, so no PR-AUC, and never raises an alarm, so "
        "precision is undefined.",
        "- **Official cost** uses Scania's cost matrix, which exists only for Scania.",
        "- **Lifecycle cost** needs engines observed all the way to failure. NASA lifecycle test "
        "engines are; Scania validation and test vehicles are cut off at a random point, so there "
        "is no Scania lifecycle cost.",
        "- **RMSE and NASA score** measure remaining-life predictions. Scania's validation and "
        "test "
        "labels are classes, not remaining life, so these are NASA only. On NASA rows, the "
        "logistic regression row shows the ridge regression baseline for remaining life, and the "
        "gradient boosting row shows the gradient boosting regressor.",
    ]
    if scania and subsets:
        engines = [m["dataset"]["lifecycle_test_engines"] for m in subsets.values()]
        ratio = next(iter(subsets.values()))["settings"]["cost_ratio_main"]
        lines.append(
            f"- Costs are not comparable across datasets: Scania uses the official matrix on "
            f"{scania['dataset']['test_vehicles']:,} test vehicles, NASA uses the assumed "
            f"{ratio:g}:1 lifecycle cost on {min(engines)} to {max(engines)} lifecycle test "
            f"engines."
        )
    else:
        lines.append("- Costs are not comparable across datasets.")
    return "\n".join(lines)


def write_comparison(reports_root: Path, subsets: dict, scania) -> list:
    """Write reports/comparison.csv and return its rows."""
    import csv

    rows = comparison_rows(subsets, scania)
    if rows:
        with open(Path(reports_root) / "comparison.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COMPARISON_COLUMNS)
            w.writeheader()
            w.writerows(rows)
    return rows


def write_summary(reports_root: Path, readme: Path = None) -> dict:
    """Rebuild every generated report from saved metrics.

    From saved metrics only: write reports/summary.json and reports/comparison.csv, and (if a README
    path is given) rebuild every generated README section that has markers.
    """
    reports_root = Path(reports_root)
    subsets = load_subset_metrics(reports_root)
    scania = load_scania_metrics(reports_root)
    summary = {}
    if subsets:
        summary = build_summary(subsets)
        with open(reports_root / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
    rows = write_comparison(reports_root, subsets, scania)
    if readme is not None and Path(readme).exists():
        text = Path(readme).read_text()
        sections = {"COMPARISON": comparison_markdown(rows, scania, subsets)}
        if subsets:
            sections["RESULTS"] = summary_markdown(subsets)
        if scania:
            sections["SCANIA"] = scania_markdown(scania, reports_root)
            sections["SCANIA_SHAP"] = scania_shap_markdown(scania, reports_root)
        for name, body in sections.items():
            new = replace_section(text, name, body)
            text = text if new is None else new
        Path(readme).write_text(text)
    return summary or ({"scania": True} if scania else {})


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ok = bool(write_summary(root / "reports", root / "README.md"))
    sys.exit(0 if ok else 1)
