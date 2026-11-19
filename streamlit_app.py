"""Streamlit dashboard.

Run locally with: streamlit run streamlit_app.py

Reads the saved files in reports/, so it needs no raw data or model file and deploys as-is. The data
logic lives in mfp/dashboard.py, where it is tested.
"""

from pathlib import Path
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st

from mfp.dashboard import (
    SCANIA,
    UNLIMITED,
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
    risk_label,
    scania_headline,
    scania_subline,
    scania_summary,
    scania_takeaway,
    scania_vehicle_table,
    shap_bars,
    sweep_for_min_lead,
    vehicle_label,
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
  font-size: 1.75rem; font-weight: 800; color: {INK}; line-height: 1.25; padding: 0.1rem 0 0.3rem;
}}
[class*="st-key-card"] {{
  background: #ffffff; border: 1px solid {LINE} !important; border-radius: 16px;
  padding: 1.2rem 1.4rem; box-shadow: {CARD};
}}
[class*="st-key-card"] h3 {{color: {INK}; font-weight: 700; font-size: 1.2rem;}}
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
.kpis {{display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem;}}
.kpi {{
  background: #ffffff; border: 1px solid {LINE}; border-radius: 16px; padding: 1.1rem 1.25rem;
  box-shadow: {CARD}; font-family: 'Inter', sans-serif;
}}
.kpi-top {{display: flex; align-items: center; gap: 0.6rem;}}
.kpi-icon {{
  width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
  color: var(--c); background: color-mix(in srgb, var(--c) 12%, white);
}}
.kpi-label {{
  font-size: 0.76rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  color: {MUTED};
}}
.kpi-value {{font-size: 2.3rem; font-weight: 800; color: var(--v); margin: 0.5rem 0 0.35rem;}}
.kpi-bar {{height: 6px; background: #f1f5f9; border-radius: 999px; overflow: hidden;}}
.kpi-bar span {{display: block; height: 100%; background: var(--c); border-radius: 999px;}}
.kpi-note {{font-size: 0.85rem; color: #475569; margin-top: 0.5rem;}}
.action {{
  font-family: 'Inter', sans-serif; border: 1px solid {LINE}; border-radius: 14px;
  padding: 1rem 1.2rem; background: #f8fafc;
}}
.action-label {{
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: {MUTED};
}}
.action-value {{font-size: 1.45rem; font-weight: 800; color: {INK}; margin: 0.3rem 0;}}
.action-note {{font-size: 0.85rem; color: #475569;}}
.st-key-headline-sub p {{font-size: 1.05rem; color: #334155;}}
.brand {{display: flex; align-items: center; gap: 0.7rem; font-family: 'Inter', sans-serif;}}
.brand-icon {{
  width: 40px; height: 40px; border-radius: 12px; background: {INK}; color: #ffffff;
  display: grid; place-items: center;
}}
.brand-name {{font-weight: 800; color: {INK}; font-size: 1rem; line-height: 1.2;}}
.brand-sub {{font-size: 0.8rem; color: {MUTED};}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{display: none;}}
[data-baseweb="tab-list"] {{gap: 0.5rem;}}
[data-testid="stTab"] {{
  background: #ffffff; border: 1px solid {LINE}; border-radius: 999px; padding: 0.35rem 1.1rem;
}}
[data-testid="stTab"][aria-selected="true"] {{background: {INK}; border-color: {INK};}}
[data-testid="stTab"][aria-selected="true"] p {{color: #ffffff;}}
[data-testid="stTab"] p {{font-size: 0.98rem; font-weight: 600;}}
"""


def _truck_svg(stroke: str, opacity: float, x: float = 0, y: float = 0, scale: float = 1) -> str:
    """Side view of a tractor-trailer as line art (64 x 32 units before scaling)."""
    return (
        f'<g transform="translate({x} {y}) scale({scale})" fill="none" stroke="{stroke}" '
        f'stroke-opacity="{opacity}" stroke-width="1.6" stroke-linejoin="round">'
        '<rect x="2" y="6" width="38" height="17" rx="1.5"/><path d="M42 11h11l7 7v5H42z"/>'
        '<path d="M45 13h7l4 5h-11z"/><line x1="2" y1="23" x2="60" y2="23"/>'
        '<circle cx="10" cy="26" r="3"/><circle cx="18" cy="26" r="3"/>'
        '<circle cx="34" cy="26" r="3"/><circle cx="52" cy="26" r="3"/></g>'
    )


def _svg_url(width: int, height: int, body: str) -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{body}</svg>'
    )
    return f'url("data:image/svg+xml;utf8,{quote(svg)}")'


# Truck theme: faint line-art trucks across the white page, and a small road strip under the
# title where a truck drives past. Motion stops for visitors who ask for reduced motion.
PATTERN = _svg_url(
    280, 180, _truck_svg(INK, 0.05, 24, 30, 1.5) + _truck_svg(INK, 0.04, 160, 118, 1.1)
)
TRUCK = _svg_url(64, 32, _truck_svg(INK, 0.85))
TRUCK_FAINT = _svg_url(64, 32, _truck_svg(MUTED, 0.6))
THEME_CSS = (
    f".stApp {{background-color: #ffffff; background-image: {PATTERN}; "
    "background-size: 280px 180px; background-attachment: fixed;}"
    f".road {{position: relative; height: 50px; margin: 0.1rem 0 1rem; overflow: hidden; "
    f"border-radius: 12px; background: #f8fafc; border: 1px solid {LINE};}}"
    '.road::after {content: ""; position: absolute; left: 0; right: 0; bottom: 10px; '
    "height: 2px; background: repeating-linear-gradient(90deg, #cbd5e1 0 18px, transparent "
    "18px 34px); animation: lane 1.1s linear infinite;}"
    ".road-truck {position: absolute; bottom: 12px; left: -90px; width: 64px; height: 32px; "
    f"background: {TRUCK} no-repeat center / contain; animation: drive 18s linear infinite;}}"
    f".road-truck.second {{background-image: {TRUCK_FAINT}; transform: scale(0.8); "
    "animation-delay: -9s;}"
    "@keyframes drive {from {left: -90px;} to {left: 100%;}}"
    "@keyframes lane {from {background-position: 0 0;} to {background-position: -34px 0;}}"
    "@media (prefers-reduced-motion: reduce) {.road::after, .road-truck {animation: none;} "
    ".road-truck {left: 42%;} .road-truck.second {display: none;}}"
)

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.html(f"<style>{CSS}{THEME_CSS}</style>")
st.title("Predictive Maintenance for Trucks and Engines")
st.html(
    '<div class="chips"><span>Scania Component X: real fleet data</span>'
    "<span>NASA C-MAPSS: engine benchmark</span>"
    "<span>Models chosen on validation data</span>"
    "<span>Test data scored once</span></div>"
    '<div class="road"><div class="road-truck"></div><div class="road-truck second"></div></div>'
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
        .mark_bar(color=INK, cornerRadius=3)
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Average influence on risk (higher = matters more)"),
            y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            tooltip=["Feature", alt.Tooltip("mean_abs_shap:Q", format=".4f")],
        )
    )
    st.altair_chart(chart, width="stretch")


# Line icons (Feather style), drawn inline so the page needs no icon files.
ICONS = {
    "saving": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/>'
    '<polyline points="17 18 23 18 23 12"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/>'
    '<circle cx="12" cy="12" r="2"/>',
    "bell": '<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>'
    '<path d="M13.73 21a2 2 0 0 1-3.46 0"/>',
    "tool": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 '
    '7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "truck": '<rect x="1" y="3" width="15" height="13"/>'
    '<polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/>'
    '<circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
}


def _icon_css() -> str:
    """Each icon as a CSS mask, so it takes the colour of its surroundings."""
    rules = [".ico {display: inline-block; background-color: currentColor;}"]
    for name, shapes in ICONS.items():
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            'stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f"{shapes}</svg>"
        )
        url = f'url("data:image/svg+xml;utf8,{quote(svg)}")'
        rules.append(
            f".ico-{name} {{-webkit-mask: {url} center / contain no-repeat; "
            f"mask: {url} center / contain no-repeat;}}"
        )
    return "".join(rules)


st.html(f"<style>{_icon_css()}</style>")


def _icon(name: str, size: int = 18) -> str:
    return f'<i class="ico ico-{name}" style="width:{size}px;height:{size}px"></i>'


def _kpi_cards(cards: list) -> None:
    """Three product-style number cards: icon, big value, thin progress bar and a note."""
    html = []
    for c in cards:
        bar = ""
        if c.get("bar") is not None:
            width = max(0.0, min(1.0, float(c["bar"]))) * 100
            bar = f'<div class="kpi-bar"><span style="width:{width:.1f}%"></span></div>'
        html.append(
            f'<div class="kpi" title="{c.get("help", "")}" '
            f'style="--c:{c["colour"]}; --v:{c.get("value_colour", INK)}">'
            f'<div class="kpi-top"><span class="kpi-icon">{_icon(c["icon"])}</span>'
            f'<span class="kpi-label">{c["label"]}</span></div>'
            f'<div class="kpi-value">{c["value"]}</div>{bar}'
            f'<div class="kpi-note">{c["note"]}</div></div>'
        )
    st.html(f'<div class="kpis">{"".join(html)}</div>')


def _headline(text: str, sub: str, action_label: str, action_value: str, note: str) -> None:
    """Value-first headline with the recommended action beside it."""
    with st.container(key="headline"):
        left, right = st.columns([3, 1.25], gap="large", vertical_alignment="center")
        with left:
            st.header(text, anchor=False)
            with st.container(key="headline-sub"):
                st.markdown(sub)
        right.html(
            f'<div class="action"><div class="action-label">{action_label}</div>'
            f'<div class="action-value">{action_value}</div>'
            f'<div class="action-note">{note}</div></div>'
        )


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

    best = scania_summary(m, preds, UNLIMITED)
    limited = rule != UNLIMITED
    flagged_note = (
        f"Out of {d['test_vehicles']:,} trucks: the riskiest {capacity_label(rule)}."
        if limited
        else f"Out of {d['test_vehicles']:,} trucks: every truck the model flags."
    )
    _headline(
        scania_headline(s, d["test_vehicles"], limited=limited),
        scania_subline(s, best),
        "Recommended this cycle",
        f"Inspect {s['alarms']:,} trucks",
        flagged_note,
    )

    share = s["saving"] / s["always_class_0_cost"] if s["always_class_0_cost"] else 0.0
    _kpi_cards(
        [
            {
                "icon": "saving",
                "label": "Cost saved",
                "value": f"{s['saving']:,.0f} points",
                "bar": share,
                "note": f"{share:.0%} less than never inspecting "
                f"({s['always_class_0_cost']:,.0f} points)",
                "colour": GREEN,
                "value_colour": GREEN,
                "help": "Official Scania cost of always predicting class 0, minus the model cost.",
            },
            {
                "icon": "target",
                "label": "Failing trucks caught",
                "value": f"{s['recall']:.0%}" if s["recall"] is not None else "n/a",
                "bar": s["recall"],
                "note": f"{caught} of the {s['true_at_risk']} trucks that later failed",
                "colour": RED,
                "value_colour": RED,
                "help": "Share of trucks really in classes 1-4 that were flagged (recall).",
            },
            {
                "icon": "bell",
                "label": "False alarms",
                "value": f"{s['false_alarms']:,}",
                "bar": s["healthy_flagged_share"],
                "note": f"{_pct(s['healthy_flagged_share'])} of healthy trucks: cheap checks",
                "colour": "#475569",
                "help": "Healthy (class 0) trucks flagged for a check.",
            },
        ]
    )
    st.caption("Points come from Scania's official scoring, not dollars.")
    st.markdown(
        f"**Why so many false alarms?** Checking a healthy truck is cheap (7 to 10 points); "
        f"missing a failing one is expensive (200 to 500 points). One missed failure costs as much "
        f"as 20 to 70 unnecessary checks, so checking {s['false_alarms']:,} healthy trucks "
        f"to catch {caught} failing ones still saves money."
    )
    with st.expander("How to read this page and method details"):
        st.markdown(
            "- **The task**: for each test truck, predict how soon Component X will need "
            "repair: class 0 is more than 48 time steps away, then 48-24 (1), 24-12 (2), "
            "12-6 (3) and 6-0 (4).\n"
            "- **Cost** uses Scania's official matrix, in points (not dollars): an unneeded "
            "check costs 7 to 10 points, a late or missed failure 200 to 500 points. Lower is "
            'better. **Never inspecting** means "always predict class 0": the baseline to beat.\n'
            "- **Workshop capacity** (left panel): only the riskiest k% of trucks are flagged. "
            "The recommended capacity was chosen on training data, before the test set was "
            "scored.\n"
            "- **At risk** means classes 1-4. **False alarms**: healthy trucks flagged.\n"
            "- **SHAP** (at the bottom) shows which readouts pushed one truck's risk up or down."
        )
        st.caption(scania_takeaway(s))
        reason = m.get("capacity", {}).get("reason")
        if reason and reason.endswith(")"):
            reason = reason[:-1] + " points)"
        st.caption(
            f"Model: {s['model'].lower()}, {s['decision']} decisions, calibrated on training "
            f"data. {d['test_vehicles']:,} test trucks, {s['true_at_risk']} of them in classes "
            "1-4. "
            + (f"Recommended capacity {capacity_label(recommended)}: {reason}." if reason else "")
        )

    left, right = st.columns([1, 1], gap="large")
    with left.container(border=True, key="card-matrix"):
        st.subheader(f":material/grid_view: At risk vs healthy ({capacity_label(rule)} capacity)")
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
                "Classes: 0 = more than 48 steps before failure, 1 = 24 to 48 steps, "
                "2 = 12 to 24 steps, 3 = 6 to 12 steps, 4 = within 6 steps."
            )
            st.caption(
                "Rows: true class. Columns: predicted class. The diagonal is correct; above it "
                "the truck was called in too early (cheap), below it too late (expensive). Hover "
                "a cell to see its cost."
            )

    table = scania_vehicle_table(preds, data["shap_top"], rule)
    with right.container(border=True, key="card-vehicles"):
        st.subheader(":material/build: Workshop action list")
        st.caption(
            "One row per truck, riskiest first. Risk is the model's estimate that the truck "
            "fails within 48 time steps. Trucks marked Inspect are the ones sent to the "
            "workshop; the last column shows what really happened in the next 48 steps."
        )
        st.caption(
            "*High risk doesn't mean certain failure. Some top trucks stayed healthy, and "
            "that's fine: a check costs only 7 to 10 points, while a missed failure costs 200 "
            "to 500 points. Several trucks share the same score because the calibration step "
            "groups similar raw scores into the same risk level; within a level, trucks are "
            "ordered by the raw model score.*"
        )
        actions = pd.DataFrame(
            {
                "Vehicle": table["Vehicle"],
                "Risk": [risk_label(r) for r in table["At-risk probability"]],
                "Risk level": table["At-risk probability"] * 100,
                "Action": ["Inspect" if c > 0 else "Monitor" for c in table["Predicted class"]],
                "Next 48 steps": [
                    "Failed" if c > 0 else "Stayed healthy" for c in table["True class"]
                ],
            }
        )
        st.dataframe(
            actions,
            hide_index=True,
            width="stretch",
            height=322,
            column_config={
                "Vehicle": st.column_config.NumberColumn(format="%d", width="small"),
                "Risk": st.column_config.TextColumn(width="small"),
                "Risk level": st.column_config.ProgressColumn(
                    format=" ", min_value=0, max_value=100, width="small"
                ),
                "Action": st.column_config.TextColumn(width="small"),
                "Next 48 steps": st.column_config.TextColumn(width="medium"),
            },
        )

    with st.expander("Explore cost and workshop capacity"):
        cost_col, capacity_col = st.columns([1, 1], gap="large")
        with cost_col.container(border=True, key="card-cost"):
            st.subheader(":material/payments: Cost impact")
            costs = pd.DataFrame(
                {
                    "Scenario": ["Never inspecting", "With the model", "Saved"],
                    "Cost": [s["always_class_0_cost"], s["test_cost"], s["saving"]],
                }
            )
            bars = alt.Chart(costs).encode(
                y=alt.Y("Scenario:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13)),
                x=alt.X("Cost:Q", title="Official Scania cost (points, test trucks)"),
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

        with capacity_col.container(border=True, key="card-capacity"):
            st.subheader(":material/tune: Savings by workshop capacity")
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
                y=alt.Y("Saved:Q", title="Cost saved (points)"),
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
                "In plain words: each bar is the cost saved (in points) if the workshop checks "
                "that share of "
                "the fleet, with the share of failing trucks it catches on top. The dark green bar "
                "is the setting chosen in the left panel; "
                f"{capacity_label(recommended)} is recommended."
            )

    with st.container(border=True, key="card-shap"):
        st.subheader(":material/insights: Why this vehicle? Top risk drivers")
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
            int(v): vehicle_label(v, p, t, c)
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
            _global_shap_chart(data["shap_global"], unit)
            st.caption(
                "Each bar shows how strongly a feature affects risk on average across all test "
                "trucks, whether it raises or lowers it. For the direction for one truck, see "
                "the red and blue chart above."
            )
            st.caption(
                'Scania does not publish its sensor names: "Sensor 167, range 5" is Scania\'s '
                "anonymous sensor ID and the histogram range it refers to; truck specs are "
                "anonymised in the same way."
            )


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

    lead = live.iloc[0]["Median lead time"]
    _headline(
        nasa_headline(live.iloc[0], ds["lifecycle_test_engines"]),
        f"Alarm rule: service the engine when its failure risk stays above {threshold:.2f} for "
        f"{confirm} cycle(s) in a row.",
        "Median warning before failure",
        "n/a" if lead is None or pd.isna(lead) else f"{lead:.0f} cycles",
        f"Across the {ds['lifecycle_test_engines']} lifecycle test engines.",
    )

    with st.container(border=True, key="card-timeline"):
        st.subheader(":material/show_chart: Engine risk timeline")
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
    pr_auc = metrics["test_last_cycle"]["pr_auc"]
    rul = metrics["rul"]
    _kpi_cards(
        [
            {
                "icon": "tool",
                "label": "Maintenance cost (lifecycle)",
                "value": f"{lt_main['chosen']['cost']:.0f}",
                "note": f"{gap:+.0f} vs a fixed 0.5 threshold"
                if gap
                else "Same as a fixed 0.5 threshold",
                "colour": "#475569",
                "help": f"Saved result at {settings['cost_ratio_main']:g}:1 on the lifecycle "
                "test engines.",
            },
            {
                "icon": "target",
                "label": "Alarm accuracy (PR-AUC)",
                "value": f"{pr_auc:.3f}",
                "bar": pr_auc,
                "note": (
                    f"{pr_auc - lr['test_last_cycle']['pr_auc']:+.3f} vs logistic regression; "
                    "1.0 is perfect"
                    if lr
                    else "1.0 is perfect"
                ),
                "colour": GREEN,
                "help": "How well risk scores separate failing from healthy NASA test engines.",
            },
            {
                "icon": "clock",
                "label": "Remaining-life error",
                "value": f"{rul['rmse_model']:.1f} cycles",
                "note": f"{rul['rmse_model'] - rul['rmse_naive_mean']:.1f} cycles vs a naive guess",
                "colour": "#475569",
                "help": "Root mean squared error of the predicted remaining life.",
            },
        ]
    )

    with st.expander("How to read this page and method details"):
        st.caption(nasa_takeaway(metrics))
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
    points = {"Official cost (Scania)": "Official cost (Scania points)"}
    styles = mask.map(lambda hit: HIGHLIGHT if hit else "").rename(columns=points)
    st.dataframe(
        comparison_display(df).rename(columns=points).style.apply(lambda _: styles, axis=None),
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


st.sidebar.html(
    f'<div class="brand"><span class="brand-icon">{_icon("truck", 22)}</span><div>'
    '<div class="brand-name">Predictive Maintenance</div>'
    '<div class="brand-sub">Fleet risk dashboard</div></div></div>'
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
