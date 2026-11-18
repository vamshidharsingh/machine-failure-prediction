"""Past-only features for Scania readouts.

The features for a readout use only that vehicle's readouts at or before its
time step, never later ones. tests/test_scania.py checks this by deleting a
vehicle's later readouts and confirming its earlier features do not change.

Per readout:
- time_step and the number of readouts so far
- each counter: its value at this readout, and its rate of change per time
  step over the last `window` readouts (this one and the window - 1 before it;
  fewer if the vehicle has fewer so far, NaN at the first readout)
- each histogram: every bin's share of the histogram total at this readout,
  the total count, and the change in every bin share over the same window
- specifications as one-hot columns (categories learned from training only)

Optional groups (`extras`):
- "intensity": each counter and histogram total divided by the time step
  (how hard the vehicle has been used, not just how long)
- "rate10": change per time step over the last 10 readouts for each counter
  and histogram total (a longer-term rate next to the 5-readout one)
- "fleet": each counter's intensity divided by the median intensity of
  training vehicles with the same Spec_0 (see FleetRelative)

Missing values stay NaN. Gradient boosting handles them directly; the logistic
baseline imputes medians fitted on training data only.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .data import COUNTERS, HIST_COLS, SPEC_COLS


def counter_name(col: str) -> str:
    """ "171" for the counter column "171_0"."""
    return col.rsplit("_", 1)[0]


EXTRA_GROUPS = ("intensity", "rate10", "fleet")
LONG_WINDOW = 10


def _lag_positions(pos: np.ndarray, window: int) -> np.ndarray:
    """Position of the oldest readout in each row's window.

    Row position of the oldest readout in a `window`-readout window that ends at each row (same
    vehicle, never later).
    """
    return np.arange(len(pos)) - np.minimum(pos, window - 1)


def readout_features(readouts: pd.DataFrame, window: int = 5, extras=()) -> pd.DataFrame:
    """One feature row per readout, indexed like `readouts`, in (vehicle, time) order.

    `extras` may hold "intensity" and "rate10"; "fleet" needs training medians
    and is added by FleetRelative.
    """
    df = readouts.sort_values(["vehicle_id", "time_step"], kind="mergesort")
    vid = df["vehicle_id"].to_numpy()
    t = df["time_step"].to_numpy(np.float64)
    pos = df.groupby("vehicle_id", sort=False).cumcount().to_numpy()
    lag = _lag_positions(pos, window)
    has_history = pos > 0
    dt = t - t[lag]

    parts = {"time_step": t.astype(np.float32), "n_readouts": (pos + 1).astype(np.float32)}

    with np.errstate(divide="ignore", invalid="ignore"):
        for col in COUNTERS:
            v = df[col].to_numpy(np.float64)
            name = counter_name(col)
            parts[f"{name}_value"] = v.astype(np.float32)
            rate = np.where(has_history & (dt > 0), (v - v[lag]) / dt, np.nan)
            parts[f"{name}_rate"] = rate.astype(np.float32)

        for hist, cols in HIST_COLS.items():
            x = df[cols].to_numpy(np.float64)
            total = x.sum(axis=1)  # NaN if any bin is missing
            share = np.where(total[:, None] > 0, x / total[:, None], np.nan)
            change = np.where(has_history[:, None], share - share[lag], np.nan)
            for i in range(len(cols)):
                parts[f"{hist}_bin{i}_share"] = share[:, i].astype(np.float32)
            parts[f"{hist}_total"] = total.astype(np.float32)
            for i in range(len(cols)):
                parts[f"{hist}_bin{i}_share_change"] = change[:, i].astype(np.float32)

        totals = {counter_name(c): df[c].to_numpy(np.float64) for c in COUNTERS}
        totals.update(
            {
                f"{h}_total": df[cols].to_numpy(np.float64).sum(axis=1)
                for h, cols in HIST_COLS.items()
            }
        )
        if "intensity" in extras:
            for name, v in totals.items():
                parts[f"{name}_intensity"] = np.where(t > 0, v / t, np.nan).astype(np.float32)
        if "rate10" in extras:
            lag10 = _lag_positions(pos, LONG_WINDOW)
            dt10 = t - t[lag10]
            for name, v in totals.items():
                rate = np.where(has_history & (dt10 > 0), (v - v[lag10]) / dt10, np.nan)
                parts[f"{name}_rate10"] = rate.astype(np.float32)

    out = pd.DataFrame(parts, index=df.index)
    out.insert(0, "vehicle_id", vid)
    return out


class FleetRelative:
    """Counter intensity relative to vehicles with the same Spec_0.

    Counter intensity relative to similar vehicles: intensity / median intensity of training rows
    with the same Spec_0. Medians come from the rows passed to fit (training vehicles only); an
    unseen Spec_0 uses the median over all training rows.
    """

    def fit(self, feats: pd.DataFrame, specs: pd.DataFrame) -> "FleetRelative":
        """Learn median counter intensities per Spec_0 from training rows."""
        cols = [f"{counter_name(c)}_intensity" for c in COUNTERS]
        group = feats["vehicle_id"].map(specs.set_index("vehicle_id")["Spec_0"].astype(str))
        self.medians_ = feats[cols].groupby(group.to_numpy()).median()
        self.overall_ = feats[cols].median()
        return self

    def transform(self, feats: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
        """Copy of feats with one "<counter>_vs_fleet" column per counter."""
        group = feats["vehicle_id"].map(specs.set_index("vehicle_id")["Spec_0"].astype(str))
        out = feats.copy()
        for col in self.medians_.columns:
            med = group.map(self.medians_[col]).fillna(self.overall_[col]).to_numpy(np.float64)
            ratio = np.divide(
                feats[col].to_numpy(np.float64), med, out=np.full(len(med), np.nan), where=med > 0
            )
            out[col.replace("_intensity", "_vs_fleet")] = ratio.astype(np.float32)
        return out


def last_readout_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle: its last readout (feats is in vehicle, time order)."""
    last = ~feats["vehicle_id"].duplicated(keep="last")
    return feats[last.to_numpy()]


class SpecEncoder:
    """One-hot specifications.

    Categories come from training vehicles only; an unseen category in validation or test becomes
    all zeros.
    """

    def fit(self, specs: pd.DataFrame) -> "SpecEncoder":
        """Learn the specification categories from training vehicles."""
        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        self.encoder_.fit(specs[SPEC_COLS].astype(str))
        self.columns_ = [
            f"{col}={cat}"
            for col, cats in zip(SPEC_COLS, self.encoder_.categories_, strict=True)
            for cat in cats
        ]
        return self

    def transform(self, specs: pd.DataFrame) -> pd.DataFrame:
        """One row per vehicle, indexed by vehicle_id."""
        values = self.encoder_.transform(specs[SPEC_COLS].astype(str))
        return pd.DataFrame(values, index=specs["vehicle_id"].to_numpy(), columns=self.columns_)


def feature_names(feats: pd.DataFrame, encoder: SpecEncoder) -> list:
    """Model input names: readout features, then one-hot specifications."""
    return [c for c in feats.columns if c != "vehicle_id"] + list(encoder.columns_)


def assemble(feats: pd.DataFrame, specs: pd.DataFrame, encoder: SpecEncoder) -> np.ndarray:
    """Model matrix (float32): readout features, then the vehicle's one-hot specs."""
    onehot = encoder.transform(specs)
    spec_rows = onehot.reindex(feats["vehicle_id"].to_numpy()).to_numpy(np.float32)
    readout = feats.drop(columns="vehicle_id").to_numpy(np.float32)
    return np.hstack([readout, spec_rows])
