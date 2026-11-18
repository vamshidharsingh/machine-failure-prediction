"""Operating-regime normalisation.

FD002 and FD004 engines fly under six operating conditions (altitude, Mach
number, throttle). A sensor reading of 550 may be normal at one condition and
alarming at another, so raw readings mix "where the engine is flying" with
"how worn it is". This module removes the first part:

1. Assign each row an operating regime by clustering the three operating
   settings with KMeans (6 clusters for FD002/FD004, 1 for FD001/FD003).
2. Z-score each sensor with the mean and standard deviation of its regime.
3. Sensors that are constant after normalisation carry no wear signal and are
   dropped.

Everything is fitted on fit engines only, then applied unchanged to
validation, lifecycle test and NASA test data. Each row is normalised using
only its own values and the fitted statistics, so features stay past-only.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import SENSOR_COLS, SETTING_COLS


class RegimeNormalizer:
    """Per-regime z-scores of the sensors, fitted on fit engines."""

    def __init__(self, n_regimes: int, seed: int, tol: float = 1e-6):
        self.n_regimes = int(n_regimes)
        self.seed = seed
        self.tol = tol

    def fit(self, df: pd.DataFrame) -> "RegimeNormalizer":
        """Cluster the operating settings and store per-regime sensor statistics."""
        settings = df[SETTING_COLS].to_numpy(float)
        self.scaler_ = StandardScaler().fit(settings)
        self.kmeans_ = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=self.seed)
        self.kmeans_.fit(self.scaler_.transform(settings))
        regimes = self.kmeans_.labels_
        grouped = df[SENSOR_COLS].groupby(regimes)
        self.means_ = grouped.mean().reindex(range(self.n_regimes))
        self.stds_ = grouped.std().reindex(range(self.n_regimes)).fillna(0.0)
        return self

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Regime label for every row, using the clusters fitted on fit engines."""
        return self.kmeans_.predict(self.scaler_.transform(df[SETTING_COLS].to_numpy(float)))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Copy of df with a `regime` column and sensors z-scored per regime.

        A sensor that is constant within a regime (std ~ 0) becomes 0 there.
        """
        out = df.copy()
        regimes = self.assign(df)
        mu = self.means_.to_numpy()[regimes]
        sd = self.stds_.to_numpy()[regimes]
        raw = df[SENSOR_COLS].to_numpy(float)
        safe = np.where(sd > self.tol, sd, 1.0)
        z = np.where(sd > self.tol, (raw - mu) / safe, 0.0)
        out[SENSOR_COLS] = z
        out["regime"] = regimes
        return out


def informative_normalised(fit_norm: pd.DataFrame, tol: float = 1e-6) -> list:
    """Sensors that still vary after normalisation, decided on fit engines only.

    Operating settings are used only to assign regimes, so they are not
    features.
    """
    stds = fit_norm[SENSOR_COLS].std()
    return [c for c in SENSOR_COLS if stds[c] > tol]
