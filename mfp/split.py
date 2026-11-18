"""Split by whole engines so no engine leaks between the parts."""

import numpy as np
import pandas as pd


def engine_split(df: pd.DataFrame, val_fraction: float, seed: int):
    """Return (rest, validation), each made of whole engines."""
    units = np.array(sorted(df["unit"].unique()))
    if len(units) < 2:
        raise ValueError("Need at least two engines to split")
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(units) * val_fraction)))
    val_units = set(rng.choice(units, size=n_val, replace=False).tolist())
    is_val = df["unit"].isin(val_units)
    return df[~is_val].copy(), df[is_val].copy()


def three_way_split(df: pd.DataFrame, val_fraction: float, lifecycle_fraction: float, seed: int):
    """Return (fit, validation, lifecycle_test), each made of whole engines.

    Lifecycle test engines run to failure and are never used to fit or choose
    anything; they only show when the alarm would have rung.
    """
    units = np.array(sorted(df["unit"].unique()))
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(units) * val_fraction)))
    n_life = max(1, int(round(len(units) * lifecycle_fraction)))
    if n_val + n_life >= len(units):
        raise ValueError(f"Need more than {n_val + n_life} engines for a three-way split")
    order = rng.permutation(units)
    val_units = set(order[:n_val].tolist())
    life_units = set(order[n_val : n_val + n_life].tolist())
    is_val = df["unit"].isin(val_units)
    is_life = df["unit"].isin(life_units)
    return df[~is_val & ~is_life].copy(), df[is_val].copy(), df[is_life].copy()
