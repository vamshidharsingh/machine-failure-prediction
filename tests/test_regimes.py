"""Operating-regime normalisation."""

from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
from conftest import CONSTANT_PER_REGIME, REGIME_SETTINGS

from mfp.config import Config
from mfp.data import SENSOR_COLS, load_cmapss
from mfp.features import add_label, add_train_rul, build_features, feature_columns
from mfp.pipeline import run
from mfp.regimes import RegimeNormalizer, informative_normalised
from mfp.split import three_way_split


def _fit_part(data_dir, subset="FD002"):
    train, test, test_rul = load_cmapss(data_dir, subset)
    train = add_label(add_train_rul(train), 30)
    fit, val, life = three_way_split(train, 0.2, 0.2, seed=42)
    return fit, val, life, test


def test_clusters_recover_the_six_operating_conditions(multiregime_dir):
    fit, _, _, test = _fit_part(multiregime_dir)
    norm = RegimeNormalizer(6, seed=42).fit(fit)
    for df in (fit, test):
        regimes = norm.assign(df)
        true = df["op1"].round().to_numpy()
        # Every true condition maps to exactly one cluster, and vice versa.
        pairs = pd.DataFrame({"true": true, "regime": regimes}).drop_duplicates()
        assert len(pairs) == len(REGIME_SETTINGS)
        assert pairs["regime"].nunique() == len(REGIME_SETTINGS)


def test_normalised_sensors_are_z_scores_per_regime(multiregime_dir):
    fit, _, _, _ = _fit_part(multiregime_dir)
    norm = RegimeNormalizer(6, seed=42).fit(fit)
    out = norm.transform(fit)
    varying = [f"s{i}" for i in range(1, 22) if i not in CONSTANT_PER_REGIME]
    by_regime = out.groupby("regime")[varying]
    np.testing.assert_allclose(by_regime.mean().to_numpy(), 0, atol=1e-9)
    np.testing.assert_allclose(by_regime.std().to_numpy(), 1, atol=1e-9)
    # Sensors that are constant within each regime become 0 and are dropped.
    for i in CONSTANT_PER_REGIME:
        assert (out[f"s{i}"] == 0).all()
    used = informative_normalised(out)
    assert used == varying
    assert not any(c.startswith("op") for c in used)


def test_single_regime_is_a_plain_z_score(cmapss_dir):
    train, _, _ = load_cmapss(cmapss_dir)
    norm = RegimeNormalizer(1, seed=42).fit(train)
    out = norm.transform(train)
    assert set(out["regime"]) == {0}
    assert np.isclose(out["s2"].mean(), 0) and np.isclose(out["s2"].std(), 1)
    expected = (train["s3"] - train["s3"].mean()) / train["s3"].std()
    np.testing.assert_allclose(out["s3"], expected)


def test_normalizer_ignores_validation_and_test_data(multiregime_dir):
    """Regime statistics come from fit engines only.

    Clusters and statistics come from fit engines only: changing every other set must not change
    them. (KMeans uses multithreaded sums, so cluster centres are compared to 1e-12, i.e. floating-
    point noise.)
    """
    fit, val, life, test = _fit_part(multiregime_dir)
    a = RegimeNormalizer(6, seed=42).fit(fit)
    # Refit on identical fit rows: the other sets play no part.
    b = RegimeNormalizer(6, seed=42).fit(fit.copy())
    np.testing.assert_allclose(
        a.kmeans_.cluster_centers_, b.kmeans_.cluster_centers_, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(a.means_, b.means_)
    pd.testing.assert_frame_equal(a.stds_, b.stds_)

    # Transforming other data leaves the fitted state untouched.
    centers, means, stds = (a.kmeans_.cluster_centers_.copy(), a.means_.copy(), a.stds_.copy())
    shifted = test.copy()
    shifted[SENSOR_COLS] += 1000
    a.transform(shifted)
    a.transform(val)
    a.transform(life)
    np.testing.assert_array_equal(a.kmeans_.cluster_centers_, centers)
    pd.testing.assert_frame_equal(a.means_, means)
    pd.testing.assert_frame_equal(a.stds_, stds)


def test_pipeline_fits_regimes_on_fit_engines_only(tmp_path, multiregime_dir):
    """The regime normaliser ignores every set except the fit engines.

    End to end: scramble the NASA test file and the validation and lifecycle engines; the saved
    clusters and statistics must not change.
    """

    def cfg(out):
        return replace(
            Config(),
            subset="FD002",
            data_dir=multiregime_dir,
            reports_dir=out / "reports",
            models_dir=out / "models",
        )

    run(cfg(tmp_path / "a"), tune=False, update_docs=False)
    first = joblib.load(tmp_path / "a" / "models" / "model.joblib")

    train, test, _ = load_cmapss(multiregime_dir, "FD002")
    _, val, life = three_way_split(train, 0.2, 0.2, seed=42)
    rng = np.random.default_rng(5)
    other = train["unit"].isin(set(val["unit"]) | set(life["unit"]))
    train.loc[other, SENSOR_COLS] += rng.normal(0, 20, (other.sum(), len(SENSOR_COLS)))
    test[SENSOR_COLS] += rng.normal(0, 20, (len(test), len(SENSOR_COLS)))
    fmt = ["%d", "%d"] + ["%.6f"] * (train.shape[1] - 2)
    np.savetxt(multiregime_dir / "train_FD002.txt", train.to_numpy(), fmt=fmt)
    np.savetxt(multiregime_dir / "test_FD002.txt", test.to_numpy(), fmt=fmt)

    run(cfg(tmp_path / "b"), tune=False, update_docs=False)
    second = joblib.load(tmp_path / "b" / "models" / "model.joblib")

    n1, n2 = first["normalizer"], second["normalizer"]
    np.testing.assert_allclose(
        n1.kmeans_.cluster_centers_, n2.kmeans_.cluster_centers_, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(n1.means_, n2.means_)
    pd.testing.assert_frame_equal(n1.stds_, n2.stds_)
    assert first["sensors"] == second["sensors"]


def test_features_use_no_future_cycles_after_normalisation(multiregime_dir):
    fit, _, _, _ = _fit_part(multiregime_dir)
    norm = RegimeNormalizer(6, seed=42).fit(fit)
    sensors = informative_normalised(norm.transform(fit))
    unit = int(fit["unit"].iloc[0])

    full = build_features(norm.transform(fit), sensors, window=10)
    cut = 60
    truncated = fit[~((fit["unit"] == unit) & (fit["cycle"] > cut))]
    partial = build_features(norm.transform(truncated), sensors, window=10)

    cols = feature_columns(full)
    a = full[(full["unit"] == unit) & (full["cycle_id"] <= cut)][cols].to_numpy()
    b = partial[(partial["unit"] == unit) & (partial["cycle_id"] <= cut)][cols].to_numpy()
    assert len(a) == cut
    np.testing.assert_allclose(a, b)
