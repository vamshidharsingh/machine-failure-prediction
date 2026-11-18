"""Model selection for Scania on training and validation data only.

The official validation set has about 136 at-risk vehicles, too few to choose
between methods reliably. A simulated validation set is built from training
vehicles the way the official sets were made: each vehicle's readouts are cut
at one seeded random readout, features use only readouts up to that point,
and its class follows the training labelling rules (vehicles whose class is
unknown at that point are skipped). Predictions for it are out-of-fold, from
models trained on the other folds of vehicles.

Every candidate is scored on both the simulated set and the official
validation set. The test set is never loaded here.

    python -m mfp.scania.selection
"""

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ..config import ScaniaConfig
from ..models import make_boosting, make_multiclass_baseline
from .data import COUNTERS, N_CLASSES, load_split
from .evaluate import (
    at_risk_probability,
    decide,
    fit_prior_weights,
    reweight,
    total_cost,
)
from .features import (
    FleetRelative,
    SpecEncoder,
    assemble,
    counter_name,
    feature_names,
    last_readout_rows,
    readout_features,
)
from .horizons import (
    HORIZONS,
    AtRiskCalibrator,
    HorizonModel,
    balanced_rows,
    class_shares,
    fixed_prior_weights,
    horizon_labels,
    time_left,
    vehicle_weights,
)
from .labels import HEALTHY_AFTER, label_training, time_left_to_class

CALIBRATIONS = ("none", "prior_shift", "prior_shift_isotonic", "searched_on_validation")


def simulated_validation(
    feats: pd.DataFrame, left, repaired, n_folds: int, seed: int
) -> pd.DataFrame:
    """One seeded random readout per training vehicle, with its class and fold.

    `feats` is in (vehicle, time) order and `left` / `repaired` are aligned
    with it. Returns columns: row (position in feats), vehicle_id, fold,
    class_label. Vehicles whose class at the chosen readout is unknown (not
    repaired and HEALTHY_AFTER steps or fewer before the study ended) are
    skipped.
    """
    rng = np.random.default_rng(seed)
    vid = feats["vehicle_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, vid[1:] != vid[:-1]])
    counts = np.diff(np.r_[starts, len(vid)])
    rows = starts + rng.integers(0, counts)
    vehicles = vid[starts]
    fold_of = dict(zip(rng.permutation(vehicles), np.arange(len(vehicles)) % n_folds, strict=True))

    left = np.asarray(left, float)[rows]
    repaired = np.asarray(repaired, int)[rows]
    known = ((repaired == 1) & (left >= 0)) | ((repaired == 0) & (left > HEALTHY_AFTER))
    out = pd.DataFrame(
        {
            "row": rows,
            "vehicle_id": vehicles,
            "fold": [fold_of[v] for v in vehicles],
            "class_label": time_left_to_class(left),
        }
    )
    return out[known].reset_index(drop=True)


def evaluate_probs(y, probs) -> dict:
    """Ranking, calibration and official-cost numbers for one set of predictions."""
    y = np.asarray(y, int)
    risk = y > 0
    r = np.clip(at_risk_probability(probs), 0.0, 1.0)  # sums can exceed 1 by rounding
    pred = decide(probs, "lowest_expected_cost")
    flagged = pred > 0
    healthy = ~risk
    return {
        "vehicles": int(len(y)),
        "at_risk": int(risk.sum()),
        "pr_auc": float(average_precision_score(risk, r)),
        "roc_auc": float(roc_auc_score(risk, r)),
        "brier": float(brier_score_loss(risk, r)),
        "cost": total_cost(y, pred),
        "cost_per_vehicle": total_cost(y, pred) / len(y),
        "always_class_0_cost": total_cost(y, np.zeros(len(y), int)),
        "healthy_flagged_share": float(flagged[healthy].mean()),
        "recall": float(flagged[risk].mean()),
        "precision": float(risk[flagged].mean()) if flagged.any() else None,
    }


class Calibration:
    """One way of turning raw class probabilities into calibrated ones.

    - "none": unchanged
    - "prior_shift": fixed class weights = class share in the simulated
      validation set / class share in the training rows (no search)
    - "prior_shift_isotonic": prior shift, then isotonic calibration of the
      at-risk probability fitted on out-of-fold training predictions
    - "searched_on_validation": the class weights that minimise official
      validation cost over a grid (kept only as a comparison: it is fitted
      on the set it is scored on)
    """

    def __init__(self, kind: str):
        if kind not in CALIBRATIONS:
            raise ValueError(f"Unknown calibration: {kind}")
        self.kind = kind

    def fit(
        self, oof, y_sim, folds, train_shares, val_probs=None, y_val=None, grid=None
    ) -> "Calibration":
        """Fit on out-of-fold simulated-validation probabilities.

        `oof_` holds them calibrated out-of-sample (isotonic is cross-fitted by fold).
        """
        oof = np.asarray(oof, float)
        self.weights_ = np.ones(N_CLASSES)
        self.calibrator_ = None
        if self.kind == "searched_on_validation":
            self.weights_, _, _ = fit_prior_weights(val_probs, y_val, "lowest_expected_cost", grid)
        elif self.kind != "none":
            self.weights_ = fixed_prior_weights(class_shares(y_sim), train_shares)
        shifted = reweight(oof, self.weights_)
        self.oof_ = shifted
        if self.kind == "prior_shift_isotonic":
            self.oof_ = shifted.copy()
            for k in np.unique(folds):
                fit_on = folds != k
                cal = AtRiskCalibrator().fit(shifted[fit_on], y_sim[fit_on] > 0)
                self.oof_[~fit_on] = cal.transform(shifted[~fit_on])
            self.calibrator_ = AtRiskCalibrator().fit(shifted, y_sim > 0)
        return self

    def transform(self, probs) -> np.ndarray:
        """Calibrated class probabilities for new predictions."""
        out = reweight(probs, self.weights_)
        return out if self.calibrator_ is None else self.calibrator_.transform(out)

    def info(self) -> dict:
        """Settings saved in metrics.json."""
        return {
            "kind": self.kind,
            "weights": self.weights_.tolist(),
            "isotonic": self.calibrator_ is not None,
        }


def calibrated(probs_sim, y_sim, folds, probs_val, y_val, train_shares, calibration, grid):
    """Apply one calibration to both evaluation sets.

    Apply one calibration to the simulated (out-of-fold) and official validation probabilities.
    Returns (sim, val, info).
    """
    cal = Calibration(calibration).fit(
        probs_sim, y_sim, folds, train_shares, probs_val, y_val, grid
    )
    info = {} if calibration == "none" else {"weights": cal.weights_.tolist()}
    return cal.oof_, cal.transform(probs_val), info


class Data:
    """Training features, labels and the two evaluation sets, built once."""

    def __init__(self, cfg: ScaniaConfig, log=print):
        t0 = time.time()
        train = load_split(cfg.data_dir, "train", cfg.use_cache)
        val = load_split(cfg.data_dir, "validation", cfg.use_cache)
        self.train_specs, self.val_specs = train["specs"], val["specs"]
        extras = ("intensity", "rate10")
        feats = readout_features(train["readouts"], cfg.change_window, extras=extras)
        self.feats = feats.reset_index(drop=True)
        ordered = train["readouts"].loc[feats.index]
        self.left, self.repaired = time_left(ordered, train["tte"])
        self.vehicle = self.feats["vehicle_id"].to_numpy()
        self.horizon = {h: horizon_labels(self.left, self.repaired, h) for h in HORIZONS}
        known = ((self.repaired == 1) & (self.left >= 0)) | (self.left > HEALTHY_AFTER)
        self.five_class = np.where(known, time_left_to_class(self.left), -1)
        self.sim = simulated_validation(
            self.feats, self.left, self.repaired, cfg.selection_folds, cfg.seed
        )
        vf = last_readout_rows(readout_features(val["readouts"], cfg.change_window, extras=extras))
        self.val_feats = vf
        self.y_val = (
            val["labels"].set_index("vehicle_id").loc[vf["vehicle_id"], "class_label"].to_numpy()
        )
        self.encoder = SpecEncoder().fit(self.train_specs)
        _, self.label_report = label_training(train["readouts"], train["tte"])
        self.train_vehicles = int(train["tte"]["vehicle_id"].nunique())
        log(
            f"  data: {len(self.feats):,} training readouts, {len(self.sim):,} simulated "
            f"validation "
            f"vehicles ({int((self.sim['class_label'] > 0).sum())} at risk), {len(self.y_val):,} "
            f"official validation vehicles ({int((self.y_val > 0).sum())} at risk) "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    def columns(self, groups) -> list:
        """Readout feature columns for a set of feature groups.

        Readout feature columns for a set of feature groups (fleet columns are added by matrix()).
        """
        cols = [
            c
            for c in self.feats.columns
            if c != "vehicle_id" and not c.endswith(("_intensity", "_rate10"))
        ]
        for group, suffix in (("intensity", "_intensity"), ("rate10", "_rate10")):
            if group in groups:
                cols += [c for c in self.feats.columns if c.endswith(suffix)]
        return cols

    def fleet_columns(self) -> list:
        """Names of the fleet-relative counter columns."""
        return [f"{counter_name(c)}_vs_fleet" for c in COUNTERS]

    def readout_frame(self, feats, specs, groups, fleet=None) -> pd.DataFrame:
        """Readout features for a set of feature groups, one row per readout."""
        part = feats[["vehicle_id", *self.columns(groups)]]
        if "fleet" in groups:
            rel = fleet.transform(feats[["vehicle_id", *fleet.medians_.columns]], specs)
            part = pd.concat([part, rel[self.fleet_columns()]], axis=1)
        return part

    def matrix(self, feats, specs, groups, fleet=None) -> np.ndarray:
        """Model matrix for a set of feature groups."""
        return assemble(self.readout_frame(feats, specs, groups, fleet), specs, self.encoder)

    def names(self, groups) -> list:
        """Model input names for a set of feature groups."""
        cols = self.columns(groups) + (self.fleet_columns() if "fleet" in groups else [])
        return feature_names(pd.DataFrame(columns=["vehicle_id", *cols]), self.encoder)


def sample_rows(mask, fraction: float, seed: int) -> np.ndarray:
    """A seeded random `fraction` of the rows in `mask`."""
    if fraction >= 1.0:
        return mask
    return mask & (np.random.default_rng(seed).random(len(mask)) < fraction)


def cap_rows(vehicle, mask, max_rows, seed) -> np.ndarray:
    """Keep at most `max_rows` rows per vehicle within `mask` (seeded sample)."""
    if max_rows is None:
        return mask
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    shuffled = idx[rng.permutation(len(idx))]
    rank = pd.Series(vehicle[shuffled]).groupby(vehicle[shuffled]).cumcount().to_numpy()
    out = np.zeros(len(mask), bool)
    out[shuffled[rank < max_rows]] = True
    return out


class FiveClassModel:
    """A multiclass classifier whose probabilities always have 5 columns."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X) -> np.ndarray:
        """n x 5 class probabilities."""
        out = np.zeros((len(X), N_CLASSES))
        out[:, np.asarray(self.model.classes_, int)] = self.model.predict_proba(X)
        return out


class FittedCandidate:
    """A fitted candidate: its model plus the fleet medians its features need."""

    def __init__(self, cand: dict, model, fleet, data: "Data"):
        self.cand, self.model, self.fleet, self.data = cand, model, fleet, data

    @property
    def groups(self) -> tuple:
        """Optional feature groups the candidate uses."""
        return tuple(self.cand["features"])

    def matrix(self, feats, specs) -> np.ndarray:
        """Model matrix for this candidate's features."""
        return self.data.matrix(feats, specs, self.groups, self.fleet)

    def predict_proba(self, feats, specs) -> np.ndarray:
        """n x 5 class probabilities from raw features."""
        return self.model.predict_proba(self.matrix(feats, specs))


class ModelMaker:
    """Factory for fresh, unfitted classifiers.

    Builds a fresh, unfitted classifier; a class (not a closure) so fitted horizon models can be
    saved with joblib.
    """

    def __init__(self, learner: str, seed: int, params: dict):
        self.learner, self.seed, self.params = learner, seed, dict(params)

    def __call__(self):
        """A new classifier with the stored settings."""
        if self.learner == "gradient_boosting":
            return make_boosting(self.seed, **self.params)
        return make_multiclass_baseline(self.seed)


def _fit(data: "Data", cand: dict, cfg: ScaniaConfig, train_mask) -> FittedCandidate:
    groups = tuple(cand["features"])
    fleet = (
        FleetRelative().fit(data.feats[train_mask], data.train_specs) if "fleet" in groups else None
    )
    X = data.matrix(data.feats[train_mask], data.train_specs, groups, fleet)
    vehicles = data.vehicle[train_mask] if cand["weighted"] else None
    make = ModelMaker(cand["learner"], cfg.seed, cfg.boosting_params)
    if cand["target"] == "horizons":
        labels = {h: data.horizon[h][train_mask] for h in HORIZONS}
        model = HorizonModel(make).fit(X, labels, vehicles=vehicles, seed=cfg.seed)
    else:
        y = data.five_class[train_mask]
        keep = y >= 0
        rows = (
            np.flatnonzero(keep)
            if vehicles is None
            else balanced_rows(vehicles, keep, seed=cfg.seed)
        )
        model = FiveClassModel(make().fit(X[rows], y[rows]))
    return FittedCandidate(cand, model, fleet, data)


def _training_mask(data: "Data", base_mask, cfg: ScaniaConfig, row_fraction: float):
    return cap_rows(
        data.vehicle,
        sample_rows(base_mask, row_fraction, cfg.seed),
        cfg.max_rows_per_vehicle,
        cfg.seed,
    )


def fit_candidate(
    data: "Data", cand: dict, cfg: ScaniaConfig, row_fraction: float = 1.0, log=print
) -> tuple:
    """Fit one candidate with out-of-fold predictions.

    Out-of-fold class probabilities on the simulated validation set, and the candidate fitted on all
    training vehicles. Returns (oof_probs, fitted).
    """
    sim = data.sim
    oof = np.zeros((len(sim), N_CLASSES))
    for k in range(cfg.selection_folds):
        t0 = time.time()
        held = np.isin(data.vehicle, sim.loc[sim["fold"] == k, "vehicle_id"])
        fold_model = _fit(data, cand, cfg, _training_mask(data, ~held, cfg, row_fraction))
        rows = sim.index[sim["fold"] == k]
        oof[rows] = fold_model.predict_proba(
            data.feats.iloc[sim.loc[rows, "row"]], data.train_specs
        )
        log(f"    fold {k + 1}/{cfg.selection_folds} [{time.time() - t0:.0f}s]", flush=True)
    t0 = time.time()
    fitted = _fit(
        data, cand, cfg, _training_mask(data, np.ones(len(data.feats), bool), cfg, row_fraction)
    )
    log(f"    fit on all training vehicles [{time.time() - t0:.0f}s]", flush=True)
    return oof, fitted


def candidate_predictions(data: "Data", cand: dict, cfg: ScaniaConfig, log=print):
    """Simulated and official validation probabilities for one candidate.

    Out-of-fold class probabilities on the simulated set and full-fit probabilities on the official
    validation set for one candidate.
    """
    oof, fitted = fit_candidate(data, cand, cfg, cfg.selection_row_fraction, log)
    return oof, fitted.predict_proba(data.val_feats, data.val_specs)


def _name(cand: dict) -> str:
    target = "horizons" if cand["target"] == "horizons" else "5-class"
    learner = "GB" if cand["learner"] == "gradient_boosting" else "LR"
    feats = "+".join(["base", *cand["features"]])
    return (
        f"{target} {learner} · {'vehicle-weighted' if cand['weighted'] else 'unweighted'} · {feats}"
    )


def run_selection(cfg: ScaniaConfig, log=print) -> dict:
    """Compare the candidates and choose the headline model.

    Score every candidate on both evaluation sets, keep feature groups and weighting only when they
    lower the simulated-validation cost, and choose the headline by simulated-validation cost.
    Writes reports/scania/model_selection.csv and selection.json.
    """
    start = time.time()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cache = cfg.models_dir / "selection_cache"
    cache.mkdir(parents=True, exist_ok=True)
    data = Data(cfg, log)
    y_sim, folds = data.sim["class_label"].to_numpy(), data.sim["fold"].to_numpy()
    rows, results = [], {}

    def score(cand):
        """Predictions and evaluation of one candidate (cached on disk)."""
        name = _name(cand)
        if name in results:
            return results[name]
        key = f"{name} rows{cfg.selection_row_fraction:g}"
        path = cache / (
            key.replace(" ", "_").replace("·", "").replace("|", "").replace("+", "-") + ".npz"
        )
        log(f"  {name}", flush=True)
        if path.exists():
            saved = np.load(path)
            ps, pv = saved["sim"], saved["val"]
            log("    loaded from cache", flush=True)
        else:
            ps, pv = candidate_predictions(data, cand, cfg, log)
            np.savez_compressed(path, sim=ps, val=pv)
        train_y = data.five_class
        keep = train_y >= 0
        w = vehicle_weights(data.vehicle)[keep] if cand["weighted"] else None
        shares = class_shares(train_y[keep], w)
        out = {}
        for cal in CALIBRATIONS:
            s, v, info = calibrated(
                ps, y_sim, folds, pv, data.y_val, shares, cal, cfg.prior_weight_grid
            )
            es, ev = evaluate_probs(y_sim, s), evaluate_probs(data.y_val, v)
            out[cal] = {"sim": es, "val": ev, "info": info}
            rows.append(
                {
                    "candidate": name,
                    **{k: v for k, v in cand.items() if k != "features"},
                    "features": "+".join(["base", *cand["features"]]),
                    "calibration": cal,
                    **{f"sim_{k}": x for k, x in es.items()},
                    **{f"val_{k}": x for k, x in ev.items()},
                    "weights": json.dumps(info.get("weights")),
                }
            )
        results[name] = out
        return out

    def sim_cost(cand, cal="prior_shift"):
        """Simulated-validation cost of a candidate under one calibration."""
        return score(cand)[cal]["sim"]["cost"]

    steps = []
    current = {
        "target": "five_class",
        "learner": "gradient_boosting",
        "weighted": False,
        "features": [],
    }
    score(current)
    score({**current, "learner": "logistic_regression"})

    best = {"target": "horizons", "learner": "gradient_boosting", "weighted": False, "features": []}
    weighted = {**best, "weighted": True}
    keep_w = sim_cost(weighted) < sim_cost(best)
    steps.append(
        {
            "step": "vehicle weights",
            "kept": keep_w,
            "sim_cost_without": sim_cost(best),
            "sim_cost_with": sim_cost(weighted),
        }
    )
    if keep_w:
        best = weighted
    for group in cfg.feature_groups:
        trial = {**best, "features": [*best["features"], group]}
        kept = sim_cost(trial) < sim_cost(best)
        steps.append(
            {
                "step": f"features: {group}",
                "kept": kept,
                "sim_cost_without": sim_cost(best),
                "sim_cost_with": sim_cost(trial),
            }
        )
        if kept:
            best = trial
    lr_h = {**best, "learner": "logistic_regression"}
    score(lr_h)

    table = pd.DataFrame(rows).drop_duplicates(["candidate", "calibration"])
    table.to_csv(cfg.reports_dir / "model_selection.csv", index=False)
    eligible = table[table["calibration"] != "searched_on_validation"]
    # The headline is a gradient boosting model; logistic regression is the
    # baseline it is compared with.
    boosting = eligible[eligible["learner"] == "gradient_boosting"]
    winner = boosting.sort_values(["sim_cost", "val_cost"], kind="mergesort").iloc[0]
    val_winner = boosting.sort_values(["val_cost", "sim_cost"], kind="mergesort").iloc[0]
    lr_rows = eligible[eligible["learner"] == "logistic_regression"]
    lr_best = lr_rows.sort_values(["sim_cost", "val_cost"], kind="mergesort").iloc[0]
    choice = {
        "baseline": {
            "candidate": lr_best["candidate"],
            "target": lr_best["target"],
            "learner": lr_best["learner"],
            "weighted": bool(lr_best["weighted"]),
            "features": [f for f in lr_best["features"].split("+") if f != "base"],
            "calibration": lr_best["calibration"],
            "sim_cost": float(lr_best["sim_cost"]),
            "val_cost": float(lr_best["val_cost"]),
        },
        "candidate": winner["candidate"],
        "target": winner["target"],
        "learner": winner["learner"],
        "weighted": bool(winner["weighted"]),
        "features": [f for f in winner["features"].split("+") if f != "base"],
        "calibration": winner["calibration"],
        "sim_cost": float(winner["sim_cost"]),
        "val_cost": float(winner["val_cost"]),
        "official_validation_agrees": val_winner["candidate"] == winner["candidate"]
        and val_winner["calibration"] == winner["calibration"],
        "official_validation_best": {
            "candidate": val_winner["candidate"],
            "calibration": val_winner["calibration"],
            "val_cost": float(val_winner["val_cost"]),
        },
        "steps": steps,
        "row_fraction": cfg.selection_row_fraction,
        "simulated_set": {
            "vehicles": int(len(y_sim)),
            "at_risk": int((y_sim > 0).sum()),
            "class_counts": np.bincount(y_sim, minlength=N_CLASSES).tolist(),
            "folds": cfg.selection_folds,
            "seed": cfg.seed,
        },
        "run_time_seconds": round(time.time() - start, 1),
    }
    with open(cfg.reports_dir / "selection.json", "w") as f:
        json.dump(choice, f, indent=2)
    return {"choice": choice, "table": table}


def main(argv=None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Scania model selection (no test data)")
    parser.add_argument("--max-rows-per-vehicle", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = ScaniaConfig()
    if args.max_rows_per_vehicle is not None:
        cfg = replace(cfg, max_rows_per_vehicle=args.max_rows_per_vehicle)
    out = run_selection(cfg)
    print(json.dumps(out["choice"], indent=2))


if __name__ == "__main__":
    main()
