"""SHAP explanations: readable names, per-item tables and saved outputs."""

import numpy as np

from mfp.explain import feature_name_map, global_importance, readable_name, top_features_per_item


def test_readable_names():
    assert readable_name("171_rate", "scania", 5) == "Counter 171: rate of change (last 5 readouts)"
    assert readable_name("171_value", "scania", 5) == "Counter 171: value"
    assert readable_name("459_bin12_share", "scania", 5) == "Histogram 459, bin 12: share of total"
    assert readable_name("459_bin12_share_change", "scania", 5) == (
        "Histogram 459, bin 12: change in share of total (last 5 readouts)"
    )
    assert readable_name("397_total", "scania", 5) == "Histogram 397: total count"
    assert readable_name("Spec_3=Cat1", "scania", 5) == "Specification 3 is Cat1"
    assert readable_name("n_readouts", "scania", 5) == "Number of readouts so far"
    assert readable_name("s11", "nasa", 10) == "Sensor 11: current value (regime-normalised)"
    assert readable_name("s11_trend", "nasa", 10) == "Sensor 11: trend per cycle (last 10 cycles)"
    assert readable_name("cycle", "nasa", 10) == "Cycle number"
    names = feature_name_map(["s2", "s2_mean", "s2_std"], "nasa", 10)
    assert names["readable_name"].is_unique


def test_top_features_per_item_and_global_importance():
    values = np.array([[0.5, -0.2, 0.1, 0.0], [-0.3, 0.4, -0.1, 0.2]])
    X = np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0]])
    feats = ["a", "b", "c", "d"]
    out = top_features_per_item(
        values,
        X,
        ids=[10, 20],
        id_name="unit",
        features=feats,
        names=[f.upper() for f in feats],
        top=2,
    )
    first = out[out["unit"] == 10]
    assert first[first["direction"] == "up"]["feature"].tolist() == ["a", "c"]
    assert first[first["direction"] == "down"]["feature"].tolist() == ["b"]  # only one negative
    assert first.loc[first["feature"] == "c", "feature_value"].isna().all()
    second = out[out["unit"] == 20]
    assert second[second["direction"] == "up"]["feature"].tolist() == ["b", "d"]
    assert set(out["readable_name"]) <= {"A", "B", "C", "D"}

    glob = global_importance(values, feats, [f.upper() for f in feats], top=3)
    assert glob["feature"].tolist() == ["a", "b", "c"]  # mean |SHAP| 0.4, 0.3, 0.1
    assert np.allclose(glob["mean_abs_shap"], [0.4, 0.3, 0.1])


def test_risk_shap_explains_prior_shifted_at_risk_log_odds_exactly():
    from sklearn.ensemble import HistGradientBoostingClassifier

    from mfp.explain import at_risk_log_odds, risk_shap
    from mfp.scania.evaluate import reweight

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6))
    y = np.digitize(X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 2000), [0.8, 1.2, 1.6, 2.0])
    X[rng.random(X.shape) < 0.05] = np.nan
    model = HistGradientBoostingClassifier(max_iter=40, random_state=0).fit(X, y)
    weights = np.array([1.0, 0.5, 2.0, 1.5, 3.0])
    values, base, info, output = risk_shap(model, X[:50], weights)
    assert values.shape == (50, 6)
    p = reweight(model.predict_proba(X[:50]), weights)
    r = p[:, 1:].sum(axis=1)
    assert np.allclose(output, np.log(r / (1 - r)), atol=1e-8)
    assert np.abs(values.sum(axis=1) + base - output).max() < 1e-8
    assert at_risk_log_odds(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))[0] == np.log(4)
