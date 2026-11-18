"""Streamlit dashboard.

Run locally with: streamlit run streamlit_app.py

Reads the saved files in reports/, so it needs no raw data or model file and deploys as-is. The data
logic lives in mfp/dashboard.py, where it is tested.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from mfp.dashboard import (
    SCANIA,
    available_datasets,
    best_cells,
    capacity_label,
    capacity_options,
    choose_policy,
    comparison_display,
    confusion_long,
    default_vehicle,
    engine_alarm,
    engine_timeline,
    explanation,
    lifecycle_table,
    load_comparison,
    load_scania,
    load_subset,
    load_subset_extras,
    lowest_cost_rows,
    nasa_takeaway,
    policy_comparison,
    recommended_capacity,
    scania_summary,
    scania_takeaway,
    scania_vehicle_table,
    sweep_for_min_lead,
    warning_time_options,
)

REPORTS = Path(__file__).resolve().parent / "reports"
HIGHLIGHT = "background-color: rgba(47, 133, 90, 0.22); font-weight: 600"

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance for Trucks and Engines")
st.caption(
    "Scania Component X (real truck data, the main case) and NASA's simulated C-MAPSS "
    "turbofan engines (FD001 to FD004). Every model, threshold and correction was chosen on "
    "validation data; test data was scored once."
)

datasets = available_datasets(REPORTS)
if not datasets:
    st.error(
        "No results yet. Run `python -m mfp.scania.pipeline` and `python -m mfp.pipeline` "
        "first, then reload this page."
    )
    st.stop()


@st.cache_data
def _scania() -> dict:
    return load_scania(REPORTS)


@st.cache_data
def _nasa(subset: str) -> dict:
    return {**load_subset(REPORTS, subset), **load_subset_extras(REPORTS, subset)}


@st.cache_data
def _comparison():
    return load_comparison(REPORTS)


def _pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


def _shap_lists(expl: dict) -> None:
    up, down = st.columns(2)
    up.markdown("**Pushing the risk up**")
    for line in expl["up"] or ["(none)"]:
        up.write(f"- {line}")
    down.markdown("**Pushing the risk down**")
    for line in expl["down"] or ["(none)"]:
        down.write(f"- {line}")


def _global_shap_chart(shap_global, unit: str) -> None:
    if shap_global is None:
        return
    top = shap_global.head(10)
    chart = (
        alt.Chart(top)
        .mark_bar(color="#2b6cb0")
        .encode(
            x=alt.X("mean_abs_shap:Q", title=f"Mean |SHAP| ({unit})"),
            y=alt.Y("readable_name:N", sort="-x", title=None),
            tooltip=["readable_name", alt.Tooltip("mean_abs_shap:Q", format=".4f")],
        )
    )
    st.altair_chart(chart, width="stretch")


def scania_view() -> None:
    """Scania results at a chosen workshop capacity."""
    data = _scania()
    m, preds = data["metrics"], data["predictions"]
    d = m["dataset"]
    options = capacity_options(m)
    recommended = recommended_capacity(m)

    def option_label(rule):
        """Capacity label, marking the recommended one."""
        label = capacity_label(rule)
        return f"{label} (recommended)" if rule == recommended else label

    rule = st.radio(
        "Workshop capacity: share of vehicles that may be flagged",
        options,
        index=options.index(recommended),
        horizontal=True,
        format_func=option_label,
    )
    s = scania_summary(m, preds, rule)

    above_zero = bool(s["saving_95"]) and s["saving_95"][0] > 0
    (st.success if above_zero else st.info)(scania_takeaway(s))

    with st.expander("How to read this page"):
        st.markdown(
            "- **The task**: for each test vehicle, predict how soon Component X will need "
            "repair: class 0 is more than 48 time steps away, then 48-24 (1), 24-12 (2), 12-6 (3) "
            "and 6-0 (4).\n"
            "- **Cost** uses Scania's official matrix: an unneeded check costs 7 to 10, a late or "
            'missed failure 200 to 500. Lower is better. **Always class 0** means "do nothing": '
            "the baseline to beat.\n"
            "- **Workshop capacity**: the official matrix assumes every vehicle can be inspected. "
            "With a capacity of k%, only the riskiest k% of vehicles are flagged. The recommended "
            "capacity was chosen on training data, before the test set was scored.\n"
            "- **At risk** means classes 1-4. **Precision**: of the vehicles flagged, the share "
            "really at risk. **Recall**: of the vehicles really at risk, the share flagged. "
            "**False alarms**: healthy vehicles flagged.\n"
            "- **SHAP** (at the bottom) shows which readouts pushed one vehicle's risk up or down."
        )

    reason = m.get("capacity", {}).get("reason")
    st.caption(
        f"Model: {s['model'].lower()}, {s['decision']} decisions, calibrated on training data. "
        f"{d['test_vehicles']:,} test vehicles, {s['true_at_risk']} of them in classes 1-4. "
        + (f"Recommended capacity {capacity_label(recommended)}: {reason}." if reason else "")
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Test cost (official matrix)",
        f"{s['test_cost']:,.0f}",
        delta=f"{s['test_cost'] - s['always_class_0_cost']:,.0f} vs always class 0",
        delta_color="inverse",
    )
    c2.metric("Always class 0 cost", f"{s['always_class_0_cost']:,.0f}")
    c3.metric(
        "At-risk recall",
        _pct(s["recall"]),
        help="Share of vehicles really in class 1-4 that were flagged.",
    )
    c4.metric(
        "At-risk precision",
        _pct(s["precision"]),
        help="Share of flagged vehicles that really are in class 1-4.",
    )
    c5.metric(
        "False alarms",
        f"{s['false_alarms']:,}",
        help=f"Healthy (class 0) vehicles flagged: {_pct(s['healthy_flagged_share'])} of them.",
    )

    st.subheader(f"Confusion matrix (test vehicles, {capacity_label(rule)} capacity)")
    cm = confusion_long(s["confusion_matrix"])
    base = alt.Chart(cm).encode(
        x=alt.X("Predicted class:O"),
        y=alt.Y("True class:O"),
        tooltip=["True class", "Predicted class", "Vehicles", alt.Tooltip("Cost:Q", format=",.0f")],
    )
    heat = base.mark_rect().encode(
        color=alt.Color("Vehicles:Q", scale=alt.Scale(type="symlog", scheme="blues"), legend=None)
    )
    text = base.mark_text(fontSize=13).encode(
        text=alt.Text("Vehicles:Q", format=","),
        color=alt.condition(alt.datum.Vehicles > 500, alt.value("white"), alt.value("#1a202c")),
    )
    st.altair_chart((heat + text).properties(height=260), width="stretch")
    st.caption(
        "Rows: true class. Columns: predicted class. The diagonal is correct; above it "
        "the vehicle was called in too early (cheap), below it too late (expensive). Hover "
        "a cell to see its cost."
    )

    st.subheader("Test vehicles")
    table = scania_vehicle_table(preds, data["shap_top"], rule)
    st.dataframe(table.drop(columns="Explained"), hide_index=True, width="stretch", height=280)

    st.subheader("Why this vehicle? (SHAP)")
    shap_info = m.get("shap") or {}
    explained = table[table["Explained"]]
    if explained.empty:
        st.info("No SHAP explanations saved. Run `python -m mfp.scania.pipeline`.")
        return
    log_odds = "log-odds" in shap_info.get("output", "")
    unit = "risk score, log-odds" if log_odds else "probability"
    st.caption(
        "Contributions to the **risk score (log-odds)**: the log of the odds that the vehicle "
        "fails within 48 time steps; +0.7 roughly doubles the odds, -0.7 halves them."
        if log_odds
        else "Contributions to the at-risk probability."
    )
    labels = {
        int(v): f"Vehicle {int(v)}: at-risk {p:.3f}, true class {int(t)}, predicted {int(c)}"
        for v, t, c, p in zip(
            explained["Vehicle"],
            explained["True class"],
            explained["Predicted class"],
            explained["At-risk probability"],
            strict=True,
        )
    }
    options = list(labels)
    vid = st.selectbox(
        "Test vehicle (riskiest first)",
        options,
        help="Opens on the riskiest vehicle that is truly at risk and was flagged.",
        index=options.index(default_vehicle(explained)),
        format_func=labels.get,
    )
    _shap_lists(explanation(data["shap_top"], "vehicle_id", vid, None if log_odds else unit))
    with st.expander("Features that matter most"):
        every, flagged = st.tabs(
            ["All test vehicles", f"Flagged at {capacity_label(recommended)} capacity"]
        )
        with every:
            _global_shap_chart(data["shap_global"], unit)
        with flagged:
            _global_shap_chart(data.get("shap_global_flagged"), unit)


def _highlight_rows(df: pd.DataFrame, rows: list) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    styles.iloc[rows, :] = HIGHLIGHT
    return styles


def _highlight_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    styles.iloc[:, cols] = HIGHLIGHT
    return styles


def nasa_view(subset: str) -> None:
    """Lifecycle alarm, engine timeline and NASA test results for one subset."""
    data = _nasa(subset)
    metrics, timelines, preds = data["metrics"], data["timelines"], data["predictions"]
    settings, ds = metrics["settings"], metrics["dataset"]
    max_lead = settings["alert_max_lead"]
    min_lead_main = settings["alert_min_lead"]
    regimes = ds.get("n_regimes", 1)

    st.info(nasa_takeaway(metrics))

    with st.expander("How to read this page"):
        st.markdown(
            "- Each **lifecycle test engine** runs to failure. The alarm rings once, when the "
            "failure probability has stayed at or above a threshold for 1, 2 or 3 cycles in a "
            "row.\n"
            f"- **Good catch**: rings {min_lead_main} to {max_lead} cycles before failure. **Early "
            f"alarm**: earlier than {max_lead} cycles (a wasted check, the false alarm). "
            f"**Missed**: "
            "never, or too late to act.\n"
            "- **Cost** = cost of a miss × misses + early alarms. The 10:1 ratio and the warning "
            "window are assumptions: change them with the slider and the warning-time selector.\n"
            "- **Three policies**: the threshold chosen on validation engines, a fixed 0.5 that "
            "rings at once, and a fixed 0.5 that waits the same number of cycles as the chosen "
            "policy. Green marks the cheapest.\n"
            "- **NASA test engines** stop before failure, so they give the standard benchmark: "
            "remaining-life error and, per engine, the SHAP explanation."
        )

    st.caption(
        f"{subset}: {regimes} operating regime{'s' if regimes != 1 else ''}. Engines: "
        f"{ds['fit_engines']} fit, {ds['validation_engines']} validation, "
        f"{ds['lifecycle_test_engines']} lifecycle test, {ds['test_engines']} NASA test. "
        "Gradient boosting is the headline model; logistic regression is the baseline."
    )
    lr = metrics.get("baseline_logistic_regression", {})
    c1, c2, c3, c4 = st.columns(4)
    lt_main = metrics["lifecycle_test"]
    gap = lt_main["chosen"]["cost"] - lt_main["fixed_0.5"]["cost"]
    c1.metric(
        "Lifecycle cost (cost-based)",
        f"{lt_main['chosen']['cost']:.0f}",
        delta=f"{gap:+.0f} vs fixed 0.5" if gap else "same as fixed 0.5",
        delta_color="inverse" if gap else "off",
        help=f"Saved result at {settings['cost_ratio_main']:g}:1 on the lifecycle test engines.",
    )
    c2.metric(
        "NASA test PR-AUC",
        f"{metrics['test_last_cycle']['pr_auc']:.3f}",
        delta=(
            f"{metrics['test_last_cycle']['pr_auc'] - lr['test_last_cycle']['pr_auc']:+.3f}"
            " vs logistic"
        )
        if lr
        else None,
        help="Last recorded cycle of each NASA test engine.",
    )
    c3.metric(
        "RUL RMSE (cycles)",
        f"{metrics['rul']['rmse_model']:.1f}",
        delta=f"{metrics['rul']['rmse_model'] - metrics['rul']['rmse_naive_mean']:.1f} vs naive",
        delta_color="inverse",
    )
    c4.metric("Lifecycle test engines", ds["lifecycle_test_engines"])

    # Shared controls: they drive both the lifecycle tab and the timeline tab.
    left, right = st.columns([3, 1])
    ratio = left.slider(
        "Cost of one missed failure, in early alarms",
        min_value=1,
        max_value=50,
        value=int(settings["cost_ratio_main"]),
    )
    options = warning_time_options(data["sweeps"]) or [min_lead_main]
    min_lead = right.selectbox(
        "Required warning time (cycles)",
        options,
        index=options.index(min_lead_main) if min_lead_main in options else 0,
    )
    sweep = (
        data["sweep"] if data["sweeps"] is None else sweep_for_min_lead(data["sweeps"], min_lead)
    )
    threshold, confirm, priced = choose_policy(sweep, ratio)
    st.write(
        f"At **{ratio}:1** with at least **{min_lead}** cycles of warning (good catch = "
        f"{min_lead} to {max_lead} cycles before failure), the lowest validation cost comes from "
        f"ringing when the failure probability is at least **{threshold:.2f}** for **{confirm}** "
        "consecutive cycle(s). These costs and windows are assumptions, not real maintenance "
        "figures."
    )

    alarm_tab, timeline_tab, test_tab = st.tabs(
        ["Lifecycle alarm", "Engine timeline", "NASA test engines and SHAP"]
    )

    with alarm_tab:
        st.markdown(
            f"**These settings, on the {ds['lifecycle_test_engines']} lifecycle test engines**"
        )
        live = policy_comparison(timelines, threshold, confirm, min_lead, max_lead, ratio)
        st.dataframe(
            live.style.apply(
                lambda d: _highlight_rows(d, lowest_cost_rows(live["Cost"])), axis=None
            ).format(
                {
                    "Threshold": "{:.2f}",
                    "Precision": "{:.3f}",
                    "Cost": "{:.0f}",
                    "Median lead time": "{:.1f}",
                },
                na_rep="n/a",
            ),
            hide_index=True,
            width="stretch",
        )
        cost_chart = (
            alt.Chart(priced.assign(confirm=priced["confirm_cycles"].astype(str) + " cycle(s)"))
            .mark_line()
            .encode(
                x=alt.X("threshold:Q", title="Alarm threshold"),
                y=alt.Y("cost:Q", title="Validation cost"),
                color=alt.Color("confirm:N", title="Confirm"),
                tooltip=[
                    "threshold",
                    "confirm_cycles",
                    "good_catches",
                    "early_alarms",
                    "missed",
                    "cost",
                ],
            )
        )
        chosen_rule = (
            alt.Chart(pd.DataFrame({"threshold": [threshold]}))
            .mark_rule(strokeDash=[6, 4], color="#c05621")
            .encode(x="threshold:Q")
        )
        st.altair_chart(cost_chart + chosen_rule, width="stretch")
        st.caption("Validation cost for every threshold; the dashed line is the chosen one.")

        st.markdown(
            f"**Saved results at the main settings ({settings['cost_ratio_main']:g}:1, "
            f"{min_lead_main} to {max_lead} cycles)**"
        )
        lt = metrics["lifecycle_test"]
        cols = {"Cost-based policy": lt["chosen"], "Fixed 0.5": lt["fixed_0.5"]}
        if "fixed_0.5_same_confirm" in lt:
            cols["Fixed 0.5, same confirm cycles"] = lt["fixed_0.5_same_confirm"]
        if lr:
            cols["Logistic regression (own policy)"] = lr["lifecycle_test"]["chosen"]

        def _lead(x):
            return "n/a" if x is None else f"{x:.1f}"

        saved = pd.DataFrame(
            {
                name: [
                    f"{r['threshold']:.2f}",
                    r["confirm_cycles"],
                    r["good_catches"],
                    r["early_alarms"],
                    r["missed"],
                    _pct(r.get("precision")),
                    f"{r['cost']:.0f}",
                    _lead(r["median_lead_time"]),
                ]
                for name, r in cols.items()
            },
            index=[
                "Threshold",
                "Confirm cycles",
                "Good catches",
                "Early alarms",
                "Missed",
                "Precision",
                "Cost",
                "Median lead time",
            ],
        ).astype(str)
        cheapest = lowest_cost_rows([r["cost"] for r in cols.values()])
        st.dataframe(
            saved.style.apply(lambda d: _highlight_columns(d, cheapest), axis=None), width="stretch"
        )

    with timeline_tab:
        table = lifecycle_table(timelines, threshold, confirm, min_lead, max_lead)
        labels = {int(r.unit): f"Engine {int(r.unit)}: {r.outcome}" for r in table.itertuples()}
        unit = st.selectbox("Lifecycle test engine", list(labels), format_func=labels.get)

        tl = engine_timeline(timelines, unit)
        alarm = engine_alarm(tl, threshold, confirm, min_lead, max_lead)
        failure_cycle = alarm["failure_cycle"]
        st.write(alarm["sentence"])

        window = pd.DataFrame(
            {"start": [failure_cycle - max_lead], "end": [failure_cycle - min_lead]}
        )
        layers = [
            alt.Chart(window)
            .mark_rect(opacity=0.15, color="#2f855a")
            .encode(
                x="start:Q",
                x2="end:Q",
                tooltip=[
                    alt.Tooltip("start:Q", title="Good-catch window from"),
                    alt.Tooltip("end:Q", title="to"),
                ],
            ),
            alt.Chart(tl)
            .mark_line(color="#2b6cb0")
            .encode(
                x=alt.X("cycle:Q", title="Cycle"),
                y=alt.Y(
                    "failure_probability:Q",
                    title="Failure probability",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=["cycle", "failure_probability", "rul"],
            ),
            alt.Chart(pd.DataFrame({"y": [threshold]}))
            .mark_rule(strokeDash=[6, 4], color="#c05621")
            .encode(y="y:Q"),
            alt.Chart(pd.DataFrame({"x": [failure_cycle]}))
            .mark_rule(color="#c53030")
            .encode(x="x:Q"),
            alt.Chart(pd.DataFrame({"x": [failure_cycle], "y": [1.0], "label": ["failure"]}))
            .mark_point(shape="cross", size=160, color="#c53030", filled=True)
            .encode(x="x:Q", y="y:Q", tooltip=["label", alt.Tooltip("x:Q", title="cycle")]),
        ]
        if alarm["alarm_cycle"] is not None:
            layers.append(
                alt.Chart(
                    pd.DataFrame(
                        {
                            "x": [alarm["alarm_cycle"]],
                            "y": [alarm["alarm_probability"]],
                            "label": ["alarm rings"],
                        }
                    )
                )
                .mark_point(shape="triangle-down", size=220, color="#c05621", filled=True)
                .encode(x="x:Q", y="y:Q", tooltip=["label", alt.Tooltip("x:Q", title="cycle")])
            )
        st.altair_chart(alt.layer(*layers), width="stretch")
        st.caption(
            f"Dashed line: threshold {threshold:.2f}. Triangle: where the alarm rings "
            f"(after {confirm} consecutive cycle(s) at or above it). Red line and cross: failure. "
            f"Green band: the good-catch window ({min_lead} to {max_lead} cycles before failure)."
        )
        with st.expander("All lifecycle test engines under this policy"):
            st.dataframe(table, hide_index=True)

    with test_tab:
        st.scatter_chart(
            preds,
            x="true_rul",
            y="predicted_rul",
            x_label="True RUL (cycles)",
            y_label="Predicted RUL (cycles)",
        )
        r = metrics["rul"]
        if "nasa_score_model" in r:
            st.caption(
                f"RMSE: gradient boosting {r['rmse_model']:.1f}, ridge {r['rmse_ridge']:.1f}, "
                f"naive "
                f"{r['rmse_naive_mean']:.1f}. NASA score (lower is better, late predictions cost "
                f"more): gradient boosting {r['nasa_score_model']:,.0f}, ridge "
                f"{r['nasa_score_ridge']:,.0f}."
            )
        if data["shap_top"] is not None:
            st.markdown("**Why this engine? (SHAP)**")
            engine_labels = {
                int(
                    p.unit
                ): f"Engine {int(p.unit)}: failure probability {p.failure_probability:.3f}, "
                f"true RUL {int(p.true_rul)}, predicted {p.predicted_rul:.0f}"
                for p in preds.sort_values("failure_probability", ascending=False).itertuples()
            }
            test_unit = st.selectbox(
                "NASA test engine (riskiest first)",
                list(engine_labels),
                format_func=engine_labels.get,
            )
            st.caption(
                "SHAP at the engine's last recorded cycle: contributions to the **risk score "
                "(log-odds)**, the log of the odds that the engine fails within 30 cycles."
            )
            _shap_lists(explanation(data["shap_top"], "unit", test_unit))
            with st.expander("Features that matter most across NASA test engines"):
                _global_shap_chart(data["shap_global"], "risk score, log-odds")


def comparison_view() -> None:
    """Every model on every dataset."""
    df = _comparison()
    if df is None:
        st.info("No comparison yet. Run the pipelines, or `python -m mfp.report`.")
        return
    mask = best_cells(df)
    styles = mask.map(lambda hit: HIGHLIGHT if hit else "")
    st.dataframe(
        comparison_display(df).style.apply(lambda _: styles, axis=None),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Green: the best value in each column within the same dataset (ties are all "
        "highlighted). Blank: the metric does not apply."
    )
    st.markdown(
        "- **PR-AUC, precision, recall, false alarms**: Scania uses the at-risk view (classes 1-4 "
        "vs 0) on test vehicles, first at the recommended workshop capacity, then with unlimited "
        "capacity; NASA uses the last cycle of each NASA test engine at the model's "
        "validation-chosen threshold.\n"
        "- **Official cost** exists only for Scania (its cost matrix). **Lifecycle cost** needs "
        "engines watched until failure, so it is NASA only. **RMSE and NASA score** measure "
        "remaining life, so they are NASA only; the logistic regression rows show the ridge "
        "baseline there.\n"
        "- Always class 0 has no score (no PR-AUC) and never alarms (no precision), so it has the "
        "fewest false alarms by definition.\n"
        "- Costs cannot be compared across datasets."
    )


dataset = st.radio("Dataset", datasets, horizontal=True, index=0)
view_tab, compare_tab = st.tabs([f"{dataset} results", "Model comparison"])
with view_tab:
    if dataset == SCANIA:
        scania_view()
    else:
        nasa_view(dataset)
with compare_tab:
    comparison_view()
