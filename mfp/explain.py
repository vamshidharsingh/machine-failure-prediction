"""SHAP explanations, shared by the NASA and Scania pipelines.

SHAP splits one prediction into a contribution per feature: the base value
(the average prediction) plus every feature's contribution equals the model's
output for that row. A positive value pushes the risk up, a negative value
pushes it down.

Both methods start from shap.TreeExplainer, which is exact and fast for
gradient boosting:

- tree_shap: a binary model (NASA, or a single horizon). It explains the
  log-odds of failure within the window directly.
- risk_shap: a five-class model (Scania). TreeExplainer explains each class's
  raw score exactly; the at-risk log-odds, log(sum_k>=1 exp f_k) - f_0, is a
  smooth function of those scores. Each feature's per-class contributions are
  combined with the gradient of that function averaged along the straight
  path from the base scores to the vehicle's scores (Gauss-Legendre
  quadrature). The contributions then add up to the vehicle's at-risk
  log-odds exactly, up to quadrature error, which is recorded. Prior-shift
  class weights enter as constant offsets on the class scores, so they are
  included exactly.

Everything is computed in the pipelines and saved, so the dashboard only reads
files.
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TOP_GLOBAL = 20
QUADRATURE_POINTS = 128
TOP_EACH = 5


def _nasa_name(f: str, window: int) -> str:
    if f == "cycle":
        return "Cycle number"
    m = re.fullmatch(r"s(\d+)(?:_(mean|std|trend))?", f)
    if not m:
        return f
    sensor, kind = m.groups()
    base = f"Sensor {sensor}"
    return {
        None: f"{base}: current value (regime-normalised)",
        "mean": f"{base}: average (last {window} cycles)",
        "std": f"{base}: variability (last {window} cycles)",
        "trend": f"{base}: trend per cycle (last {window} cycles)",
    }[kind]


def _scania_name(f: str, window: int) -> str:
    if f == "time_step":
        return "Time step"
    if f == "n_readouts":
        return "Number of readouts so far"
    m = re.fullmatch(r"Spec_(\d+)=(.+)", f)
    if m:
        return f"Specification {m.group(1)} is {m.group(2)}"
    m = re.fullmatch(r"(\d+)_(value|rate)", f)
    if m:
        cid, kind = m.groups()
        return (
            f"Counter {cid}: value"
            if kind == "value"
            else f"Counter {cid}: rate of change (last {window} readouts)"
        )
    m = re.fullmatch(r"(\d+)_total", f)
    if m:
        return f"Histogram {m.group(1)}: total count"
    m = re.fullmatch(r"(\d+)_bin(\d+)_share(_change)?", f)
    if m:
        hid, b, change = m.groups()
        return (
            f"Histogram {hid}, bin {b}: change in share of total (last {window} readouts)"
            if change
            else f"Histogram {hid}, bin {b}: share of total"
        )
    return f


def readable_name(feature: str, dataset: str, window: int) -> str:
    """Plain-English name, e.g. 'Counter 171: rate of change (last 5 readouts)'."""
    return _scania_name(feature, window) if dataset == "scania" else _nasa_name(feature, window)


def feature_name_map(features, dataset: str, window: int) -> pd.DataFrame:
    """Feature names next to their readable names."""
    return pd.DataFrame(
        {
            "feature": list(features),
            "readable_name": [readable_name(f, dataset, window) for f in features],
        }
    )


def tree_shap(model, X):
    """TreeExplainer on a binary gradient boosting classifier.

    Returns (values n x p, base values n, method info).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X), dtype=float)
    base = np.full(len(X), float(np.ravel(explainer.expected_value)[0]))
    info = {
        "method": "shap.TreeExplainer",
        "output": "log-odds of failure within the window",
        "feature_perturbation": "tree_path_dependent",
        "shap_version": shap.__version__,
    }
    return values, base, info


def at_risk_log_odds(scores) -> np.ndarray:
    """log(sum over classes 1-4 of exp(score)) - score of class 0, per row."""
    z = np.asarray(scores, float)
    top = z[:, 1:].max(axis=1, keepdims=True)
    return np.log(np.exp(z[:, 1:] - top).sum(axis=1)) + top[:, 0] - z[:, 0]


def risk_shap(model, X, class_weights=None, points: int = QUADRATURE_POINTS):
    """SHAP contributions to a five-class model's at-risk log-odds.

    Contributions to the at-risk log-odds of a five-class gradient boosting model, with optional
    prior-shift class weights. Returns (values n x p, base values n, method info, explained output
    n).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    phi = np.asarray(explainer.shap_values(X), dtype=float)  # n x p x 5
    offsets = np.zeros(phi.shape[2]) if class_weights is None else np.log(class_weights)
    start = np.asarray(explainer.expected_value, float) + offsets  # base scores
    end = phi.sum(axis=1) + start  # vehicle scores
    nodes, weights = np.polynomial.legendre.leggauss(points)
    grad = np.zeros_like(end)
    for t, w in zip((nodes + 1) / 2, weights / 2, strict=True):
        z = start + t * (end - start)
        e = np.exp(z[:, 1:] - z[:, 1:].max(axis=1, keepdims=True))
        grad[:, 1:] += w * e / e.sum(axis=1, keepdims=True)
        grad[:, 0] -= w
    values = np.einsum("npk,nk->np", phi, grad)
    base = np.full(len(X), float(at_risk_log_odds(start[None, :])[0]))
    info = {
        "method": "shap.TreeExplainer per class, combined through the softmax",
        "output": "log-odds of being at risk (classes 1-4), after the prior shift",
        "feature_perturbation": "tree_path_dependent",
        "quadrature_points": points,
        "shap_version": shap.__version__,
    }
    return values, base, info, at_risk_log_odds(end)


def global_importance(values, features, names, top: int = TOP_GLOBAL) -> pd.DataFrame:
    """Mean |SHAP| per feature over all explained rows, largest first."""
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs, kind="mergesort")[:top]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "feature": np.asarray(features)[order],
            "readable_name": np.asarray(names)[order],
            "mean_abs_shap": mean_abs[order],
        }
    )


def top_features_per_item(
    values, X, ids, id_name: str, features, names, top: int = TOP_EACH
) -> pd.DataFrame:
    """Each row's top features pushing risk up and down.

    For each explained row: its `top` features pushing risk up (largest positive SHAP) and down
    (most negative), with the feature's value.
    """
    features, names = np.asarray(features), np.asarray(names)
    rows = []
    for i, item in enumerate(ids):
        v = values[i]
        up = [j for j in np.argsort(-v, kind="mergesort")[:top] if v[j] > 0]
        down = [j for j in np.argsort(v, kind="mergesort")[:top] if v[j] < 0]
        for direction, picks in (("up", up), ("down", down)):
            for rank, j in enumerate(picks, start=1):
                x = X[i, j]
                rows.append(
                    {
                        id_name: item,
                        "direction": direction,
                        "rank": rank,
                        "feature": features[j],
                        "readable_name": names[j],
                        "shap_value": float(v[j]),
                        "feature_value": None if np.isnan(x) else float(x),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            id_name,
            "direction",
            "rank",
            "feature",
            "readable_name",
            "shap_value",
            "feature_value",
        ],
    )


def plot_summary(global_df: pd.DataFrame, path, title: str, xlabel: str) -> None:
    """Horizontal bar chart of mean |SHAP| per feature."""
    df = global_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(df) + 1.4))
    ax.barh(df["readable_name"], df["mean_abs_shap"], color="#2b6cb0")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_explanations(
    reports_dir,
    values,
    base,
    info,
    X,
    ids,
    id_name,
    features,
    dataset: str,
    window: int,
    title: str,
    model_output=None,
    flagged=None,
) -> dict:
    """Write feature_names.csv, shap_global.csv, shap_top_features.csv and shap_summary.png.

    Returns the block saved into metrics.json.

    `model_output` is what was explained (one number per row); when given,
    the largest gap between it and base + sum of SHAP values is recorded.
    `flagged` (boolean per row) adds shap_global_flagged.csv: the ranking over
    flagged rows only.
    """
    names_df = feature_name_map(features, dataset, window)
    names = names_df["readable_name"].to_numpy()
    names_df.to_csv(reports_dir / "feature_names.csv", index=False)
    glob = global_importance(values, features, names)
    glob.to_csv(reports_dir / "shap_global.csv", index=False)
    per_item = top_features_per_item(values, X, ids, id_name, features, names)
    per_item.to_csv(reports_dir / "shap_top_features.csv", index=False)
    unit = "log-odds" if "log-odds" in info["output"] else "probability"
    plot_summary(
        glob, reports_dir / "shap_summary.png", title, f"Mean |SHAP value| ({unit} of failure risk)"
    )
    max_gap = (
        None
        if model_output is None
        else float(np.max(np.abs(values.sum(axis=1) + base - np.asarray(model_output, float))))
    )
    out = {
        **info,
        "explained_rows": int(len(ids)),
        "additivity_max_error": max_gap,
        "top_features": glob[["feature", "readable_name", "mean_abs_shap"]]
        .head(10)
        .to_dict(orient="records"),
    }
    if flagged is not None:
        flagged = np.asarray(flagged, bool)
        glob_f = global_importance(values[flagged], features, names)
        glob_f.to_csv(reports_dir / "shap_global_flagged.csv", index=False)
        out["flagged_rows"] = int(flagged.sum())
        out["top_features_flagged"] = (
            glob_f[["feature", "readable_name", "mean_abs_shap"]].head(10).to_dict(orient="records")
        )
    return out
