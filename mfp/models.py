"""Models: a logistic baseline, gradient boosting, and a RUL regressor."""

from itertools import product

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline(seed: int):
    """Logistic regression with scaling and balanced classes (NASA baseline)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )


def make_multiclass_baseline(seed: int, max_iter: int = 1000):
    """Multinomial logistic regression baseline (Scania).

    Multinomial logistic regression with median imputation and scaling, all fitted on training data
    only. No class weighting, so the probabilities keep the training class balance (needed for
    expected-cost decisions).
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=seed),
    )


def make_boosting(seed: int, **params):
    """HistGradientBoostingClassifier with the project defaults, overridden by params."""
    defaults = {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 300}
    defaults.update(params)
    return HistGradientBoostingClassifier(random_state=seed, **defaults)


def make_rul_regressor(seed: int):
    """Gradient boosting regressor for remaining useful life."""
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_leaf_nodes=31, max_iter=300, random_state=seed
    )


def make_rul_baseline():
    """Ridge regression on scaled features: the remaining-life baseline."""
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def nasa_score(true_rul, predicted_rul) -> float:
    """PHM08 / C-MAPSS scoring function (lower is better).

    d = predicted - true. Per engine: exp(-d/13) - 1 if d < 0 (early),
    exp(d/10) - 1 otherwise (late). Late predictions cost more because the
    engine is still running when the model said it had more life left.
    """
    d = np.asarray(predicted_rul, float) - np.asarray(true_rul, float)
    return float(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1).sum())


def pr_auc(y_true, scores) -> float:
    """Average precision; NaN when y_true has a single class."""
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(average_precision_score(y_true, scores))


def tune_boosting(
    X, y, groups, grid: dict, folds: int, seed: int, scorer=None, score_name: str = "cv_pr_auc"
):
    """Small grid search with folds grouped by engine or vehicle.

    Scored by PR-AUC by default; pass scorer(model, X, y) -> float (higher is
    better) for another score. Returns (best_params, results) where results
    lists every combination.
    """
    n_groups = len(np.unique(groups))
    folds = max(2, min(folds, n_groups))
    cv = GroupKFold(n_splits=folds)
    keys = sorted(grid)
    results = []
    for values in product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        scores = []
        for tr, va in cv.split(X, y, groups):
            if y[va].min() == y[va].max():
                continue
            model = make_boosting(seed, **params).fit(X[tr], y[tr])
            if scorer is None:
                scores.append(pr_auc(y[va], model.predict_proba(X[va])[:, 1]))
            else:
                scores.append(float(scorer(model, X[va], y[va])))
        mean = float(np.mean(scores)) if scores else float("nan")
        results.append({**params, score_name: mean})
    valid = [r for r in results if not np.isnan(r[score_name])]
    if not valid:
        return {}, results
    best = max(valid, key=lambda r: r[score_name])
    best_params = {k: best[k] for k in keys}
    return best_params, results
