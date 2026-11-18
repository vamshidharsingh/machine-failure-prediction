"""Horizon models: the five classes as four cumulative yes/no questions.

The classes are ordered windows before failure, so "will this vehicle fail
within h time steps?" for h = 6, 12, 24 and 48 answers them all:

    P(class 4) = P(<=6)            P(class 1) = P(<=48) - P(<=24)
    P(class 3) = P(<=12) - P(<=6)  P(class 0) = 1 - P(<=48)
    P(class 2) = P(<=24) - P(<=12)

Each question has its own training labels, which keeps censored readouts
that the five-class labelling has to drop: a vehicle that was not repaired
and whose study ended t steps after a readout is a known negative for every
horizon h < t, even when t <= 48 leaves its class unknown.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .data import N_CLASSES

HORIZONS = (6, 12, 24, 48)


def time_left(readouts: pd.DataFrame, tte: pd.DataFrame) -> tuple:
    """(time left before the study end or repair, repaired flag) per readout."""
    info = tte.set_index("vehicle_id")
    length = readouts["vehicle_id"].map(info["length_of_study_time_step"]).astype(float)
    repaired = readouts["vehicle_id"].map(info["in_study_repair"]).astype(int)
    return (length - readouts["time_step"].astype(float)).to_numpy(), repaired.to_numpy()


def horizon_labels(left: np.ndarray, repaired: np.ndarray, horizon: int) -> np.ndarray:
    """Labels for one horizon: 1, 0 or NaN when unknown.

    1 = repaired within `horizon` steps, 0 = time left known to exceed it, NaN = unknown for this
    horizon (not repaired and the study ended within `horizon` steps) or after the repair.
    """
    left = np.asarray(left, float)
    repaired = np.asarray(repaired, int)
    out = np.full(len(left), np.nan)
    out[left > horizon] = 0.0
    out[(repaired == 1) & (left >= 0) & (left <= horizon)] = 1.0
    return out


def cumulative_to_classes(cumulative) -> np.ndarray:
    """n x 4 probabilities of failing within 6, 12, 24, 48 steps -> n x 5 class probabilities.

    Separately trained horizons can disagree (P(<=12) below P(<=6)), so a running maximum makes them
    non-decreasing first and no class probability is negative.
    """
    c = np.clip(np.asarray(cumulative, float), 0.0, 1.0)
    c = np.maximum.accumulate(c, axis=1)
    c6, c12, c24, c48 = c.T
    return np.column_stack([1 - c48, c48 - c24, c24 - c12, c12 - c6, c6])


def vehicle_weights(vehicle_ids) -> np.ndarray:
    """Row weights that give every vehicle the same total weight.

    Row weights giving every vehicle the same total weight (1 / its rows), rescaled so the mean
    weight is 1.
    """
    ids = pd.Series(np.asarray(vehicle_ids))
    w = 1.0 / ids.map(ids.value_counts()).to_numpy(float)
    return w / w.mean()


def balanced_rows(vehicles, mask, per_vehicle=None, seed: int = 0) -> np.ndarray:
    """Rows that give every vehicle the same number of rows.

    A seeded sample with replacement of `per_vehicle` rows from each vehicle in `mask` (default:
    the median rows per vehicle). In expectation this weights each vehicle equally, like
    vehicle_weights(), but scikit-learn's gradient boosting trains on it far faster than on sample
    weights, whose weighted binning is very slow on hundreds of features.
    """
    vehicles = np.asarray(vehicles)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return idx
    order = idx[np.argsort(vehicles[idx], kind="mergesort")]
    v = vehicles[order]
    starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    counts = np.diff(np.r_[starts, len(v)])
    k = int(np.median(counts)) if per_vehicle is None else int(per_vehicle)
    rng = np.random.default_rng(seed)
    picks = starts[:, None] + (rng.random((len(starts), k)) * counts[:, None]).astype(int)
    return order[picks.ravel()]


def class_shares(classes, weights=None) -> np.ndarray:
    """Share of each class 0-4 (weighted if weights are given)."""
    classes = np.asarray(classes, int)
    w = np.ones(len(classes)) if weights is None else np.asarray(weights, float)
    totals = np.bincount(classes, weights=w, minlength=N_CLASSES)
    return totals / totals.sum()


def fixed_prior_weights(target_shares, training_shares) -> np.ndarray:
    """Prior-shift class weights computed without a search.

    Prior-shift weights with no search: target class share / training class share (1 where the
    training share is zero).
    """
    target = np.asarray(target_shares, float)
    train = np.asarray(training_shares, float)
    return np.where(train > 0, target / np.where(train > 0, train, 1.0), 1.0)


class AtRiskCalibrator:
    """Isotonic calibration of the at-risk probability P(class 1-4).

    The calibrated at-risk probability replaces 1 - P(class 0); classes 1-4
    keep their proportions to one another.
    """

    def fit(self, probs, at_risk) -> "AtRiskCalibrator":
        """Fit the isotonic map from at-risk probability to observed at-risk rate."""
        r = 1.0 - np.asarray(probs, float)[:, 0]
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso_.fit(r, np.asarray(at_risk, float))
        return self

    def transform(self, probs) -> np.ndarray:
        """Calibrated class probabilities."""
        p = np.asarray(probs, float)
        r = 1.0 - p[:, 0]
        r_new = self.iso_.predict(r)
        scale = np.divide(r_new, r, out=np.zeros_like(r), where=r > 0)
        out = p.copy()
        out[:, 1:] = p[:, 1:] * scale[:, None]
        # A vehicle with zero risk mass has no class mix to scale; give its
        # calibrated risk to class 1, the mildest at-risk class.
        out[:, 1] += np.where(r > 0, 0.0, r_new)
        out[:, 0] = 1.0 - out[:, 1:].sum(axis=1)
        return out


class HorizonModel:
    """One binary classifier per horizon.

    One binary classifier per horizon, trained only on rows whose label is known for that horizon.
    """

    def __init__(self, make_model, horizons=HORIZONS):
        self.make_model = make_model
        self.horizons = tuple(horizons)

    def fit(self, X, labels: dict, vehicles=None, seed: int = 0) -> "HorizonModel":
        """`labels` maps each horizon to a 0 / 1 / NaN array aligned with X.

        With `vehicles`, each horizon trains on balanced_rows() so every vehicle carries the same
        weight.
        """
        self.models_ = {}
        for i, h in enumerate(self.horizons):
            keep = ~np.isnan(labels[h])
            rows = (
                np.flatnonzero(keep)
                if vehicles is None
                else balanced_rows(vehicles, keep, seed=seed + i)
            )
            self.models_[h] = self.make_model().fit(X[rows], labels[h][rows].astype(int))
        return self

    def predict_cumulative(self, X) -> np.ndarray:
        """n x 4 probabilities of failing within each horizon."""
        return np.column_stack([self.models_[h].predict_proba(X)[:, 1] for h in self.horizons])

    def predict_proba(self, X) -> np.ndarray:
        """n x 5 class probabilities."""
        return cumulative_to_classes(self.predict_cumulative(X))
