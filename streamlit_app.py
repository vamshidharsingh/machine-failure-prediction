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
    at_risk_matrix,
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
    lifecycle_table,
    load_comparison,
    load_scania,
    load_subset,
    load_subset_extras,
    lowest_cost_rows,
    nasa_headline,
    nasa_takeaway,
    plain_feature_name,
    policy_comparison,
    recommended_capacity,
    scania_headline,
    scania_summary,
    scania_takeaway,
    scania_vehicle_table,
    shap_bars,
    sweep_for_min_lead,
    warning_time_options,
)

REPORTS = Path(__file__).resolve().parent / "reports"

# One colour story: red for risk, green for savings, neutral greys for everything else.
RED, GREEN, BLUE = "#dc2626", "#16a34a", "#2563eb"
INK, MUTED, LINE = "#0f172a", "#64748b", "#e5e7eb"
HIGHLIGHT = "background-color: rgba(22, 163, 74, 0.14); font-weight: 600"

CARD = "0 1px 2px rgba(15, 23, 42, 0.05), 0 8px 24px rgba(15, 23, 42, 0.06)"
FONT = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
CSS = f"""
@import url('{FONT}');
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label,
[data-testid="stMetricValue"] div {{font-family: 'Inter', sans-serif;}}
.chips {{display: flex; flex-wrap: wrap; gap: 0.5rem; margin: -0.4rem 0 0.6rem;}}
.chips span {{
  font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 600; color: #334155;
  background: #f1f5f9; border: 1px solid {LINE}; border-radius: 999px; padding: 0.25rem 0.75rem;
}}
[data-testid="stMainBlockContainer"] {{max-width: 1250px; padding-top: 3rem;}}
h1 {{color: {INK}; font-weight: 800; letter-spacing: -0.02em; font-size: 2.4rem !important;}}
.st-key-headline {{
  background: #ffffff; border: 1px solid {LINE}; border-left: 6px solid {INK};
  border-radius: 16px; padding: 1.3rem 1.8rem 1rem; box-shadow: {CARD};
}}
.st-key-headline h2 {{
  font-size: 2.1rem; font-weight: 800; color: {INK}; line-height: 1.2; padding: 0.1rem 0 0.3rem;
}}
[class*="st-key-card"] {{
  background: #ffffff; border: 1px solid {LINE} !important; border-radius: 16px;
  padding: 1.2rem 1.4rem; box-shadow: {CARD};
}}
[class*="st-key-card"] h3 {{color: {INK}; font-weight: 700; font-size: 1.4rem;}}
[data-testid="stMetric"] {{
  background: #ffffff; border: 1px solid {LINE}; border-left: 5px solid #94a3b8;
  border-radius: 14px; padding: 1rem 1.2rem; box-shadow: {CARD};
}}
.st-key-metric-saved [data-testid="stMetric"] {{border-left-color: {GREEN};}}
.st-key-metric-risk [data-testid="stMetric"] {{border-left-color: {RED};}}
[data-testid="stMetricLabel"] p {{
  text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.78rem; font-weight: 600;
  color: {MUTED};
}}
[data-testid="stMetricValue"] {{font-weight: 800; color: {INK};}}
.st-key-metric-saved [data-testid="stMetricValue"] {{color: {GREEN};}}
.st-key-metric-risk [data-testid="stMetricValue"] {{color: {RED};}}
[data-testid="stMetricDelta"] * {{white-space: normal; overflow: visible; text-overflow: clip;}}
[data-testid^="stMetricDeltaIcon"] {{display: none;}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{display: none;}}
[data-baseweb="tab-list"] {{gap: 0.5rem;}}
[data-testid="stTab"] {{
  background: #ffffff; border: 1px solid {LINE}; border-radius: 999px; padding: 0.35rem 1.1rem;
}}
[data-testid="stTab"][aria-selected="true"] {{background: {INK}; border-color: {INK};}}
[data-testid="stTab"][aria-selected="true"] p {{color: #ffffff;}}
[data-testid="stTab"] p {{font-size: 0.98rem; font-weight: 600;}}
"""

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.html(f"<style>{CSS}</style>")
st.title("Predictive Maintenance for Trucks and Engines")
st.html(
    '<div class="chips"><span>Scania Component X: real fleet data</span>'
    "<span>NASA C-MAPSS: engine benchmark</span>"
    "<span>Models chosen on validation data</span>"
    "<span>Test data scored once</span></div>"
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


def _shap_chart(bars: pd.DataFrame) -> None:
    """Red bars push the risk up, blue bars push it down."""
    if bars.empty:
        st.info("No SHAP explanation saved for this item.")
        return
    order = bars.sort_values("Contribution", ascending=False)["Feature"].tolist()
    base = alt.Chart(bars).encode(
        y=alt.Y("Feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
    )
    chart = base.mark_bar(cornerRadius=3).encode(
        x=alt.X("Contribution:Q", title="Effect on the risk score (log-odds)"),
        color=alt.Color(
            "Effect:N",
            scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RED, BLUE]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=["Feature", "Value", alt.Tooltip("Contribution:Q", format="+.3f"), "Effect"],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")
    st.altair_chart((chart + zero).properties(height=34 * len(bars) + 40), width="stretch")


def _global_shap_chart(shap_global, unit: str) -> None:
    if shap_global is None:
        return
    top = shap_global.head(10).assign(
        Feature=lambda d: [plain_feature_name(n) for n in d["readable_name"]]
    )
    chart = (
        alt.Chart(top)
        .mark_bar(color=MUTED, cornerRadius=3)
        .encode(
            x=alt.X("mean_abs_shap:Q", title=f"Mean |SHAP| ({unit})"),
            y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            tooltip=["Feature", alt.Tooltip("mean_abs_shap:Q", format=".4f")],
        )
    )
    st.altair_chart(chart, width="stretch")


def _headline(text: str, detail: str) -> None:
    with st.container(key="headline"):
        st.header(text, anchor=False)
        st.caption(detail)


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

    rule = st.sidebar.radio(
        "Workshop capacity: share of trucks that may be flagged",
        options,
        index=options.index(recommended),
        format_func=option_label,
    )
    s = scania_summary(m, preds, rule)
    caught = round(s["recall"] * s["true_at_risk"]) if s["recall"] is not None else 0

    _headline(scania_headline(s), scania_takeaway(s))

    c1, c2, c3 = st.columns(3)
    with c1.container(key="metric-saved"):
        st.metric(
            "Cost saved vs never inspecting",
            f"{s['saving']:,.0f}",
            delta=(
                f"{s['saving'] / s['always_class_0_cost']:.0%} of {s['always_class_0_cost']:,.0f}"
            ),
            help="Official Scania cost of always predicting class 0, minus the model's cost.",
        )
    c2.container(key="metric-risk").metric(
        "Failing trucks caught",
        _pct(s["recall"]),
        delta=f"{caught} of {s['true_at_risk']} trucks",
        delta_color="off",
        help="Share of trucks really in classes 1-4 that were flagged (recall).",
    )
    c3.metric(
        "False alarms",
        f"{s['false_alarms']:,}",
        delta=f"{_pct(s['healthy_flagged_share'])} of healthy trucks",
        delta_color="off",
        help="Healthy (class 0) trucks flagged for a check.",
    )
    cost_col, capacity_col = st.columns([1, 1], gap="large")
    with cost_col.container(border=True, key="card-cost"):
        st.subheader("Cost impact")
        costs = pd.DataFrame(
            {
                "Scenario": ["Never inspecting", "With the model", "Saved"],
                "Cost": [s["always_class_0_cost"], s["test_cost"], s["saving"]],
            }
        )
        bars = alt.Chart(costs).encode(
            y=alt.Y("Scenario:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13)),
            x=alt.X("Cost:Q", title="Official Scania cost (test trucks)"),
        )
        st.altair_chart(
            bars.mark_bar(cornerRadius=4, size=30).encode(
                color=alt.Color(
                    "Scenario:N",
                    scale=alt.Scale(
                        domain=["Never inspecting", "With the model", "Saved"],
                        range=["#94a3b8", INK, GREEN],
                    ),
                    legend=None,
                ),
                tooltip=["Scenario", alt.Tooltip("Cost:Q", format=",.0f")],
            )
            + bars.mark_text(align="left", dx=6, fontWeight="bold", color=INK)
            .encode(text=alt.Text("Cost:Q", format=",.0f"))
            .properties(height=190),
            width="stretch",
        )
        st.markdown(
            f"**Why so many false alarms?** Checking a healthy truck is cheap (7 to 10); "
            f"missing a failing one is expensive (200 to 500). One missed failure costs as much "
            f"as 20 to 70 unnecessary checks, so checking {s['false_alarms']:,} healthy trucks "
            f"to catch {caught} failing ones still saves money."
        )

    with capacity_col.container(border=True, key="card-capacity"):
        st.subheader("Savings by workshop capacity")
        what_if = pd.DataFrame(
            [
                {
                    "Capacity": capacity_label(r),
                    "Saved": scania_summary(m, preds, r)["saving"],
                    "Caught": f"{scania_summary(m, preds, r)['recall']:.0%} caught",
                    "Selected": r == rule,
                }
                for r in options
            ]
        )
        base = alt.Chart(what_if).encode(
            x=alt.X(
                "Capacity:N",
                sort=None,
                title="Share of trucks the workshop checks",
                axis=alt.Axis(labelAngle=0, labelOverlap=False),
            ),
            y=alt.Y("Saved:Q", title="Cost saved"),
        )
        st.altair_chart(
            base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=42).encode(
                color=alt.condition(alt.datum.Selected, alt.value(GREEN), alt.value("#bbf7d0")),
                tooltip=["Capacity", alt.Tooltip("Saved:Q", format=",.0f"), "Caught"],
            )
            + base.mark_text(dy=-9, fontWeight="bold", color=INK).encode(text="Caught:N"),
            width="stretch",
        )
        st.caption(
            "In plain words: each bar is the money saved if the workshop checks that share of "
            "the fleet, with the share of failing trucks it catches on top. The dark green bar "
            "is the setting chosen in the left panel; "
            f"{capacity_label(recommended)} is recommended."
        )
    with st.expander("How to read this page and method details"):
        st.markdown(
            "- **The task**: for each test truck, predict how soon Component X will need "
            "repair: class 0 is more than 48 time steps away, then 48-24 (1), 24-12 (2), "
            "12-6 (3) and 6-0 (4).\n"
            "- **Cost** uses Scania's official matrix: an unneeded check costs 7 to 10, a late "
            'or missed failure 200 to 500. Lower is better. **Never inspecting** means "always '
            'predict class 0": the baseline to beat.\n'
            "- **Workshop capacity** (left panel): only the riskiest k% of trucks are flagged. "
            "The recommended capacity was chosen on training data, before the test set was "
            "scored.\n"
            "- **At risk** means classes 1-4. **False alarms**: healthy trucks flagged.\n"
            "- **SHAP** (at the bottom) shows which readouts pushed one truck's risk up or down."
        )
        reason = m.get("capacity", {}).get("reason")
        st.caption(
            f"Model: {s['model'].lower()}, {s['decision']} decisions, calibrated on training "
            f"data. {d['test_vehicles']:,} test trucks, {s['true_at_risk']} of them in classes "
            "1-4. "
            + (f"Recommended capacity {capacity_label(recommended)}: {reason}." if reason else "")
        )

    left, right = st.columns([1, 1], gap="large")
    with left.container(border=True, key="card-matrix"):
        st.subheader(f"At risk vs healthy ({capacity_label(rule)} capacity)")
        cells = at_risk_matrix(s["confusion_matrix"])
        cells["Label"] = cells["Outcome"] + ": " + cells["Vehicles"].map("{:,}".format)
        outcomes = ["Caught", "Missed", "False alarm", "Correctly left alone"]
        base = alt.Chart(cells).encode(
            x=alt.X(
                "Decision:N",
                sort=["Flagged", "Not flagged"],
                title=None,
                axis=alt.Axis(labelAngle=0, orient="top"),
            ),
            y=alt.Y("Truck:N", sort=["At risk", "Healthy"], title=None),
        )
        heat = base.mark_rect(cornerRadius=6, stroke="white", strokeWidth=4).encode(
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(domain=outcomes, range=[GREEN, RED, "#fecaca", "#f1f5f9"]),
                legend=None,
            ),
            tooltip=["Truck", "Decision", "Outcome", alt.Tooltip("Vehicles:Q", format=",")],
        )
        text = base.mark_text(fontSize=15, fontWeight="bold").encode(
            text="Label:N",
            color=alt.condition(
                alt.FieldOneOfPredicate("Outcome", ["Caught", "Missed"]),
                alt.value("white"),
                alt.value(INK),
            ),
        )
        st.altair_chart(
            (heat + text).properties(height=230).configure_axis(labelFontSize=13),
            width="stretch",
        )
        st.caption(
            "In plain words: green is failing trucks caught in time, red is failing trucks "
            "missed, pink is healthy trucks checked for nothing (cheap), grey is healthy "
            "trucks correctly left alone."
        )
        with st.expander("Full 5 x 5 matrix (by time-to-failure class)"):
            cm = confusion_long(s["confusion_matrix"])
            base5 = alt.Chart(cm).encode(
                x=alt.X("Predicted class:O"),
                y=alt.Y("True class:O"),
                tooltip=[
                    "True class",
                    "Predicted class",
                    "Vehicles",
                    alt.Tooltip("Cost:Q", format=",.0f"),
                ],
            )
            heat5 = base5.mark_rect().encode(
                color=alt.Color(
                    "Vehicles:Q", scale=alt.Scale(type="symlog", scheme="greys"), legend=None
                )
            )
            text5 = base5.mark_text(fontSize=12).encode(
                text=alt.Text("Vehicles:Q", format=","),
                color=alt.condition(alt.datum.Vehicles > 500, alt.value("white"), alt.value(INK)),
            )
            st.altair_chart((heat5 + text5).properties(height=240), width="stretch")
            st.caption(
                "Rows: true class. Columns: predicted class. The diagonal is correct; above it "
                "the truck was called in too early (cheap), below it too late (expensive). Hover "
                "a cell to see its cost."
            )

    table = scania_vehicle_table(preds, data["shap_top"], rule)
    with right.container(border=True, key="card-vehicles"):
        st.subheader("Workshop action list")
        st.caption(
            "One row per truck, riskiest first. Risk is the model's estimate that the truck "
            "fails within 48 time steps. Trucks marked Inspect are the ones sent to the "
            "workshop; the last column shows what really happened."
        )
        actions = pd.DataFrame(
            {
                "Vehicle": table["Vehicle"],
                "Risk": table["At-risk probability"] * 100,
                "Action": ["Inspect" if c > 0 else "Monitor" for c in table["Predicted class"]],
                "What happened": [
                    "Failed within 48 steps" if c > 0 else "Stayed healthy"
                    for c in table["True class"]
                ],
            }
        )
        st.dataframe(
            actions,
            hide_index=True,
            width="stretch",
            height=322,
            column_config={
                "Vehicle": st.column_config.NumberColumn(format="%d"),
                "Risk": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0, max_value=100
                ),
            },
        )

    with st.container(border=True, key="card-shap"):
        st.subheader("Why this vehicle? Top risk drivers")
        shap_info = m.get("shap") or {}
        explained = table[table["Explained"]]
        if explained.empty:
            st.info("No SHAP explanations saved. Run `python -m mfp.scania.pipeline`.")
            return
        log_odds = "log-odds" in shap_info.get("output", "")
        unit = "risk score, log-odds" if log_odds else "probability"
        st.caption(
            "Contributions to the **risk score (log-odds)**: the log of the odds that the truck "
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
        st.markdown(
            "**In plain words:** each bar is one reason behind this truck's risk. Red bars "
            "push the risk up, blue bars pull it down, and a longer bar is a stronger reason."
        )
        _shap_chart(shap_bars(data["shap_top"], "vehicle_id", vid))
        st.markdown(
            "**Why is truck age (time step) often the top driver?** Training labels need a "
            "known outcome. For trucks never repaired, the last readouts before the study ended "
            "have no known class and are left out, so late readouts in training come mostly "
            "from trucks that failed. The model learns that older means riskier. Part of this "
            "signal comes from how the data was collected rather than from wear, so read it "
            "with care."
        )
        st.caption(
            'Scania does not publish its sensor names: "Sensor 459, range 13" is Scania\'s '
            "anonymous sensor ID and the histogram range it refers to."
        )
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


def _timeline_chart(tl, alarm, threshold, min_lead, max_lead) -> None:
    failure_cycle = alarm["failure_cycle"]
    window = pd.DataFrame({"start": [failure_cycle - max_lead], "end": [failure_cycle - min_lead]})
    layers = [
        alt.Chart(window)
        .mark_rect(opacity=0.12, color=GREEN)
        .encode(
            x="start:Q",
            x2="end:Q",
            tooltip=[
                alt.Tooltip("start:Q", title="Good-catch window from"),
                alt.Tooltip("end:Q", title="to"),
            ],
        ),
        alt.Chart(tl)
        .mark_line(color=RED, strokeWidth=2.5)
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y(
                "failure_probability:Q",
                title="Failure risk",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=["cycle", "failure_probability", "rul"],
        ),
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(strokeDash=[6, 4], color=MUTED)
        .encode(y="y:Q"),
        alt.Chart(pd.DataFrame({"x": [failure_cycle]})).mark_rule(color=INK).encode(x="x:Q"),
        alt.Chart(pd.DataFrame({"x": [failure_cycle], "y": [1.0], "label": ["failure"]}))
        .mark_point(shape="cross", size=160, color=INK, filled=True)
        .encode(x="x:Q", y="y:Q", tooltip=["label", alt.Tooltip("x:Q", title="cycle")]),
    ]
    if alarm["alarm_cycle"] is not None:
        point = pd.DataFrame(
            {
                "x": [alarm["alarm_cycle"]],
                "y": [alarm["alarm_probability"]],
                "label": [f"Alarm (cycle {alarm['alarm_cycle']})"],
            }
        )
        layers += [
            alt.Chart(point)
            .mark_point(shape="triangle-down", size=320, color=INK, filled=True)
            .encode(x="x:Q", y="y:Q", tooltip=["label"]),
            alt.Chart(point)
            .mark_text(dy=-22, fontSize=13, fontWeight="bold", color=INK)
            .encode(x="x:Q", y="y:Q", text="label:N"),
        ]
    st.altair_chart(alt.layer(*layers).properties(height=340), width="stretch")


def nasa_view(subset: str) -> None:
    """Engine risk timeline, lifecycle alarm and NASA test results for one subset."""
    data = _nasa(subset)
    metrics, timelines, preds = data["metrics"], data["timelines"], data["predictions"]
    settings, ds = metrics["settings"], metrics["dataset"]
    max_lead = settings["alert_max_lead"]
    min_lead_main = settings["alert_min_lead"]
    regimes = ds.get("n_regimes", 1)

    ratio = st.sidebar.slider(
        "Cost of one missed failure, in early alarms",
        min_value=1,
        max_value=50,
        value=int(settings["cost_ratio_main"]),
    )
    options = warning_time_options(data["sweeps"]) or [min_lead_main]
    min_lead = st.sidebar.selectbox(
        "Required warning time (cycles)",
        options,
        index=options.index(min_lead_main) if min_lead_main in options else 0,
    )
    sweep = (
        data["sweep"] if data["sweeps"] is None else sweep_for_min_lead(data["sweeps"], min_lead)
    )
    threshold, confirm, priced = choose_policy(sweep, ratio)
    live = policy_comparison(timelines, threshold, confirm, min_lead, max_lead, ratio)

    _headline(nasa_headline(live.iloc[0], ds["lifecycle_test_engines"]), nasa_takeaway(metrics))

    with st.container(border=True, key="card-timeline"):
        st.subheader("Engine risk timeline")
        table = lifecycle_table(timelines, threshold, confirm, min_lead, max_lead)
        labels = {int(r.unit): f"Engine {int(r.unit)}: {r.outcome}" for r in table.itertuples()}
        unit = st.selectbox("Lifecycle test engine", list(labels), format_func=labels.get)
        tl = engine_timeline(timelines, unit)
        alarm = engine_alarm(tl, threshold, confirm, min_lead, max_lead)
        st.write(alarm["sentence"])
        _timeline_chart(tl, alarm, threshold, min_lead, max_lead)
        st.markdown(
            "**In plain words:** the red line shows how worried the model is about this "
            "engine at each cycle. When it stays above the dashed line long enough, the "
            "alarm rings (triangle). Ringing inside the green band means a good catch."
        )
        st.caption(
            f"Red line: failure risk each cycle. Dashed line: alarm threshold {threshold:.2f}. "
            f"Triangle: where the alarm rings (after {confirm} consecutive cycle(s) at or above "
            f"it). Black line and cross: failure. Green band: the good-catch window ({min_lead} "
            f"to {max_lead} cycles before failure)."
        )
        with st.expander("All lifecycle test engines under this policy"):
            st.dataframe(table, hide_index=True)

    lr = metrics.get("baseline_logistic_regression", {})
    lt_main = metrics["lifecycle_test"]
    gap = lt_main["chosen"]["cost"] - lt_main["fixed_0.5"]["cost"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Maintenance cost (lifecycle)",
        f"{lt_main['chosen']['cost']:.0f}",
        delta=f"{gap:+.0f} vs fixed 0.5" if gap else "same as fixed 0.5",
        delta_color="inverse" if gap else "off",
        help=f"Saved result at {settings['cost_ratio_main']:g}:1 on the lifecycle test engines.",
    )
    c2.metric(
        "Alarm accuracy (PR-AUC)",
        f"{metrics['test_last_cycle']['pr_auc']:.3f}",
        delta=(
            f"{metrics['test_last_cycle']['pr_auc'] - lr['test_last_cycle']['pr_auc']:+.3f}"
            " vs logistic"
        )
        if lr
        else None,
        help=(
            "How well risk scores separate failing from healthy NASA test engines; 1.0 is perfect."
        ),
    )
    c3.metric(
        "Remaining-life error (RMSE, cycles)",
        f"{metrics['rul']['rmse_model']:.1f}",
        delta=f"{metrics['rul']['rmse_model'] - metrics['rul']['rmse_naive_mean']:.1f} vs naive",
        delta_color="inverse",
    )

    with st.expander("How to read this page and method details"):
        st.markdown(
            "- Each **lifecycle test engine** runs to failure. The alarm rings once, when the "
            "failure probability has stayed at or above a threshold for 1, 2 or 3 cycles in a "
            "row.\n"
            f"- **Good catch**: rings {min_lead_main} to {max_lead} cycles before failure. "
            f"**Early alarm**: earlier than {max_lead} cycles (a wasted check, the false alarm). "
            "**Missed**: never, or too late to act.\n"
            "- **Cost** = cost of a miss x misses + early alarms. The 10:1 ratio and the warning "
            "window are assumptions: change them in the left panel.\n"
            "- **Three policies**: the threshold chosen on validation engines, a fixed 0.5 that "
            "rings at once, and a fixed 0.5 that waits the same number of cycles as the chosen "
            "policy. Green marks the cheapest.\n"
            "- **NASA test engines** stop before failure, so they give the standard benchmark: "
            "remaining-life error and, per engine, the SHAP explanation."
        )
        st.write(
            f"At **{ratio}:1** with at least **{min_lead}** cycles of warning (good catch = "
            f"{min_lead} to {max_lead} cycles before failure), the lowest validation cost comes "
            f"from ringing when the failure probability is at least **{threshold:.2f}** for "
            f"**{confirm}** consecutive cycle(s). These costs and windows are assumptions, not "
            "real maintenance figures."
        )
        st.caption(
            f"{subset}: {regimes} operating regime{'s' if regimes != 1 else ''}. Engines: "
            f"{ds['fit_engines']} fit, {ds['validation_engines']} validation, "
            f"{ds['lifecycle_test_engines']} lifecycle test, {ds['test_engines']} NASA test. "
            "Gradient boosting is the headline model; logistic regression is the baseline."
        )

    alarm_tab, test_tab = st.tabs(["Lifecycle alarm", "NASA test engines and SHAP"])

    with alarm_tab:
        st.markdown(
            f"**These settings, on the {ds['lifecycle_test_engines']} lifecycle test engines**"
        )
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
                color=alt.Color(
                    "confirm:N",
                    title="Confirm",
                    scale=alt.Scale(range=["#cbd5e1", "#94a3b8", "#334155"]),
                ),
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
            .mark_rule(strokeDash=[6, 4], color=RED)
            .encode(x="threshold:Q")
        )
        st.altair_chart(cost_chart + chosen_rule, width="stretch")
        st.caption("Validation cost for every threshold; the red dashed line is the chosen one.")

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

    with test_tab:
        st.scatter_chart(
            preds,
            x="true_rul",
            y="predicted_rul",
            x_label="True remaining life (cycles)",
            y_label="Predicted remaining life (cycles)",
            color=MUTED,
        )
        r = metrics["rul"]
        if "nasa_score_model" in r:
            st.caption(
                f"RMSE: gradient boosting {r['rmse_model']:.1f}, ridge {r['rmse_ridge']:.1f}, "
                f"naive {r['rmse_naive_mean']:.1f}. NASA score (lower is better, late "
                f"predictions cost more): gradient boosting {r['nasa_score_model']:,.0f}, ridge "
                f"{r['nasa_score_ridge']:,.0f}."
            )
        if data["shap_top"] is not None:
            st.markdown("**Why this engine? (SHAP)**")
            engine_labels = {
                int(p.unit): f"Engine {int(p.unit)}: failure probability "
                f"{p.failure_probability:.3f}, true RUL {int(p.true_rul)}, predicted "
                f"{p.predicted_rul:.0f}"
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
            _shap_chart(shap_bars(data["shap_top"], "unit", test_unit))
            with st.expander("Features that matter most across NASA test engines"):
                _global_shap_chart(data["shap_global"], "risk score, log-odds")


def comparison_view() -> None:
    """Every model on every dataset."""
    df = _comparison()
    if df is None:
        st.info("No comparison yet. Run the pipelines, or `python -m mfp.report`.")
        return
    st.markdown(
        "**In plain words:** each row is one model on one dataset. Green marks the best "
        "value in each column, so you can see which model wins on which measure."
    )
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
        "- **PR-AUC, precision, recall, false alarms**: Scania uses the at-risk view (classes "
        "1-4 vs 0) on test vehicles, first at the recommended workshop capacity, then with "
        "unlimited capacity; NASA uses the last cycle of each NASA test engine at the model's "
        "validation-chosen threshold.\n"
        "- **Official cost** exists only for Scania (its cost matrix). **Lifecycle cost** needs "
        "engines watched until failure, so it is NASA only. **RMSE and NASA score** measure "
        "remaining life, so they are NASA only; the logistic regression rows show the ridge "
        "baseline there.\n"
        "- Always class 0 has no score (no PR-AUC) and never alarms (no precision), so it has "
        "the fewest false alarms by definition.\n"
        "- Costs cannot be compared across datasets."
    )


st.sidebar.header("Controls")
dataset = st.sidebar.radio("Dataset", datasets, index=0)
view_tab, compare_tab = st.tabs([f"{dataset} results", "Model comparison"])
with view_tab:
    if dataset == SCANIA:
        scania_view()
    else:
        nasa_view(dataset)
with compare_tab:
    comparison_view()
