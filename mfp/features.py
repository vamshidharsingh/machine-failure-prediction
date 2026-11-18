"""Labels and past-only features.

Every feature for cycle t uses only cycles <= t of the same engine, so the
model never sees the future. tests/test_features.py checks this.
"""

import numpy as np
import pandas as pd

from .data import SENSOR_COLS, SETTING_COLS


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Training engines run to failure, so RUL = last cycle - current cycle."""
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last - out["cycle"]
    return out


def add_test_rul(df: pd.DataFrame, test_rul: pd.DataFrame) -> pd.DataFrame:
    """Test engines stop early; the true RUL after the last cycle is given."""
    out = df.merge(test_rul, on="unit", how="left")
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = out["true_rul"] + (last - out["cycle"])
    return out.drop(columns="true_rul")


def add_label(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """label = 1 if the engine fails within `window` cycles."""
    out = df.copy()
    out["label"] = (out["rul"] <= window).astype(int)
    return out


def informative_sensors(train: pd.DataFrame, tol: float = 1e-6) -> list:
    """Sensors and settings that actually vary in the training data.

    Several FD001 sensors are constant; keeping them adds noise only.
    Decided from training data alone.
    """
    candidates = SETTING_COLS + SENSOR_COLS
    stds = train[candidates].std()
    return [c for c in candidates if stds[c] > tol]


def build_features(df: pd.DataFrame, sensors: list, window: int) -> pd.DataFrame:
    """Raw values plus rolling mean, rolling std and trend per sensor.

    Rolling windows look backward only (current and earlier cycles).
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    g = df.groupby("unit", sort=False)
    parts = {"cycle": df["cycle"].astype(float)}
    for col in sensors:
        roll = g[col].rolling(window, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True)
        std = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        # Trend: average change per cycle over the window.
        first = g[col].shift(window - 1)
        first = first.fillna(g[col].transform("first"))
        steps = np.minimum(df["cycle"] - 1, window - 1).replace(0, 1)
        trend = (df[col] - first) / steps
        parts[col] = df[col]
        parts[f"{col}_mean"] = mean
        parts[f"{col}_std"] = std
        parts[f"{col}_trend"] = trend
    feats = pd.DataFrame(parts, index=df.index)
    keep = [c for c in ["unit", "cycle", "rul", "label"] if c in df.columns]
    return pd.concat([df[keep].rename(columns={"cycle": "cycle_id"}), feats], axis=1)


def feature_columns(feats: pd.DataFrame) -> list:
    """Model input columns of a feature frame."""
    return [c for c in feats.columns if c not in {"unit", "cycle_id", "rul", "label"}]


def last_cycle_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its latest cycle."""
    idx = feats.groupby("unit")["cycle_id"].idxmax()
    return feats.loc[idx].reset_index(drop=True)
