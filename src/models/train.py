"""
Nested cross-validation engine and external evaluation.

Protocol
--------
Outer loop   repeated, CDR-cluster-grouped, target-stratified K-fold over the
             training cohort.  Hyper-parameters are NEVER chosen using any
             data in the outer test fold.
Inner loop   GroupKFold grid search within the outer training fold, grouped by
             the same CDR clusters so that hyper-parameter selection cannot
             leak through near-duplicates either.
Scoring      inner selection uses negative MAE.  MAE rather than RMSE because
             the HIC distribution has a long right tail (median 9.9 min, max
             25 min) and RMSE would let a handful of extreme antibodies decide
             the hyper-parameters.  We then report MAE, RMSE, R-squared and
             Spearman, so the reader can see all four regardless.
Aggregation  out-of-fold predictions are pooled within each repeat and the
             metric computed once on the full pooled vector; the spread across
             repeats is reported as the stability of the estimate.

External sets are never touched during any of this.  They are scored once, at
the end, by a model refit on the entire training cohort with hyper-parameters
chosen by one inner grid search on that cohort.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

from src import config
from src.models import metrics as M
from src.models import zoo as Z

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------
# One outer fold
# --------------------------------------------------------------------------

def _fit_one_fold(estimator, grid, X, y, groups, train_idx, test_idx,
                  scoring: str, inner_splits: int):
    """Grid-search inside the training fold, then predict the test fold."""
    Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
    Xte = X[test_idx]

    n_groups = len(np.unique(gtr))
    k = int(min(inner_splits, n_groups))
    if k < 2 or Z.grid_size(grid) == 1:
        best = clone(estimator)
        if Z.grid_size(grid) > 1:
            best.set_params(**{p: v[0] for p, v in grid.items()})
        best.fit(Xtr, ytr)
        return test_idx, best.predict(Xte), {}

    search = GridSearchCV(
        clone(estimator), grid, scoring=scoring,
        cv=GroupKFold(n_splits=k), n_jobs=1, refit=True, error_score="raise",
    )
    search.fit(Xtr, ytr, groups=gtr)
    return test_idx, search.best_estimator_.predict(Xte), search.best_params_


def _predict_scores(estimator, X):
    """Continuous score for a classifier: probability of the positive class."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def _fit_one_fold_clf(estimator, grid, X, y, groups, train_idx, test_idx,
                      scoring: str, inner_splits: int):
    Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
    Xte = X[test_idx]

    if len(np.unique(ytr)) < 2:
        return test_idx, np.full(len(test_idx), float(ytr.mean())), {}

    n_groups = len(np.unique(gtr))
    n_pos = int(ytr.sum())
    # The positive class is the unfavourable tail, so it is ~10-20% of the data.
    # An inner fold with no positives makes average_precision undefined, which
    # poisons the whole grid search, so we cap the inner fold count by the
    # number of positives and stratify the inner split on the label as well as
    # grouping it by cluster.
    k = int(min(inner_splits, n_groups, n_pos, len(ytr) - n_pos))
    if k < 2 or Z.grid_size(grid) == 1:
        best = clone(estimator)
        if Z.grid_size(grid) > 1:
            best.set_params(**{p: v[0] for p, v in grid.items()})
        best.fit(Xtr, ytr)
        return test_idx, _predict_scores(best, Xte), {}

    from sklearn.model_selection import StratifiedGroupKFold
    search = GridSearchCV(
        clone(estimator), grid, scoring=scoring,
        cv=StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=config.SEED),
        n_jobs=1, refit=True, error_score=np.nan,
    )
    search.fit(Xtr, ytr, groups=gtr)
    if not np.isfinite(search.best_score_):
        # Every candidate scored NaN: fall back to the first grid point rather
        # than letting GridSearchCV's arbitrary tie-break choose for us.
        best = clone(estimator)
        best.set_params(**{p: v[0] for p, v in grid.items()})
        best.fit(Xtr, ytr)
        return test_idx, _predict_scores(best, Xte), {}
    return test_idx, _predict_scores(search.best_estimator_, Xte), search.best_params_


# --------------------------------------------------------------------------
# Nested CV over all repeats
# --------------------------------------------------------------------------

@dataclass
class CVResult:
    feature_set: str
    model: str
    task: str
    n: int
    n_features: int
    oof_per_repeat: list[np.ndarray] = field(default_factory=list)
    metrics_per_repeat: list[dict] = field(default_factory=list)
    chosen_params: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def representative_oof(self, repeat: int = 0) -> np.ndarray:
        """
        One repeat's complete out-of-fold prediction vector.

        **This is the vector to use for any reported metric.**  Each repeat
        already covers every antibody exactly once, so a single repeat is a
        complete, honest nested-CV prediction: every value came from a model
        that never saw that antibody.
        """
        return self.oof_per_repeat[repeat]

    def ensemble_mean_oof(self) -> np.ndarray:
        """
        Mean prediction per antibody across repeats -- an ENSEMBLE, not a
        nested-CV prediction.

        Averaging the repeats bags five models trained on five different ~110
        antibody subsets, which cancels repeat-to-repeat variance and makes the
        predictor strictly better than any single fitted model.  Scoring that
        vector and calling it "the cross-validated performance of model X"
        overstates what model X would do on new data, and calibrating
        prediction intervals on its residuals makes them too narrow.

        It is kept only for residual plots, where the lower noise is helpful
        and no performance claim is attached.
        """
        return np.mean(np.vstack(self.oof_per_repeat), axis=0)


def run_nested_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  folds_per_repeat: list[list[tuple[np.ndarray, np.ndarray]]],
                  estimator, grid: dict, feature_set: str, model_name: str,
                  task: str = "regression", n_jobs: int = -1) -> CVResult:
    t0 = time.time()
    scoring = "neg_mean_absolute_error" if task == "regression" else "average_precision"
    fold_fn = _fit_one_fold if task == "regression" else _fit_one_fold_clf
    metric_fn = M.regression_metrics if task == "regression" else M.classification_metrics

    res = CVResult(feature_set=feature_set, model=model_name, task=task,
                   n=len(y), n_features=X.shape[1])

    # Flatten every (repeat, fold) pair into one task list before dispatching.
    # Parallelising per repeat would cap concurrency at n_splits (5), leaving
    # most cores idle on a 12-core machine; flattening gives n_repeats*n_splits
    # (50) independent tasks and keeps every worker busy.
    tasks = [(r, tr, te)
             for r, folds in enumerate(folds_per_repeat)
             for tr, te in folds]

    outputs = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(fold_fn)(estimator, grid, X, y, groups, tr, te,
                         scoring, config.INNER_CV_N_SPLITS)
        for _, tr, te in tasks
    )

    n_repeats = len(folds_per_repeat)
    oofs = [np.full(len(y), np.nan, dtype=float) for _ in range(n_repeats)]
    for (r, _, _), (test_idx, preds, params) in zip(tasks, outputs):
        oofs[r][test_idx] = preds
        if params:
            res.chosen_params.append(params)

    for oof in oofs:
        res.oof_per_repeat.append(oof)
        res.metrics_per_repeat.append(metric_fn(y, oof))

    res.seconds = time.time() - t0
    return res


def summarise_cv(res: CVResult) -> dict:
    """Pooled metric plus the across-repeat spread."""
    # Only numeric metrics can be averaged; some entries (e.g. the provenance
    # of the classification threshold) are strings and are carried through
    # unchanged rather than summarised.
    keys = [k for k, v in res.metrics_per_repeat[0].items()
            if k not in ("n", "n_positive") and isinstance(v, (int, float, np.floating))]
    out: dict = {
        "feature_set": res.feature_set, "model": res.model, "task": res.task,
        "n": res.n, "n_features": res.n_features,
        "n_repeats": len(res.oof_per_repeat), "seconds": round(res.seconds, 1),
    }
    for k in keys:
        vals = np.array([m.get(k, np.nan) for m in res.metrics_per_repeat], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[f"{k}_mean"] = float(vals.mean())
            out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


# --------------------------------------------------------------------------
# External evaluation
# --------------------------------------------------------------------------

def fit_full_and_predict(X_train: np.ndarray, y_train: np.ndarray,
                         groups: np.ndarray, estimator, grid: dict,
                         X_external: dict[str, np.ndarray],
                         task: str = "regression") -> tuple[object, dict, dict]:
    """
    Refit on the whole training cohort, then score each external cohort.

    Hyper-parameters come from one grouped grid search on the training cohort.
    That search sees only training data, so the external evaluation stays
    clean; what it does mean is that the external numbers use a model whose
    hyper-parameters were tuned on all 137 antibodies rather than on 110 of
    them, which is the normal deployment situation.
    """
    scoring = "neg_mean_absolute_error" if task == "regression" else "average_precision"
    n_groups = len(np.unique(groups))
    k = int(min(config.INNER_CV_N_SPLITS, n_groups))

    if k >= 2 and Z.grid_size(grid) > 1:
        search = GridSearchCV(clone(estimator), grid, scoring=scoring,
                              cv=GroupKFold(n_splits=k), n_jobs=-1, refit=True)
        search.fit(X_train, y_train, groups=groups)
        model, best = search.best_estimator_, search.best_params_
    else:
        model = clone(estimator)
        if Z.grid_size(grid) > 1:
            model.set_params(**{p: v[0] for p, v in grid.items()})
        model.fit(X_train, y_train)
        best = {}

    preds = {}
    for name, Xe in X_external.items():
        preds[name] = (model.predict(Xe) if task == "regression"
                       else _predict_scores(model, Xe))
    return model, best, preds


# --------------------------------------------------------------------------
# Full-refit permutation test (the expensive, honest one)
# --------------------------------------------------------------------------

def refit_permutation_test(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           folds: list[tuple[np.ndarray, np.ndarray]],
                           estimator, grid: dict, observed_spearman: float,
                           n_perm: int = 200, seed: int = config.SEED,
                           n_jobs: int = -1) -> dict:
    """
    Permute y, re-run the whole nested CV, recompute Spearman. Repeat.

    This tests the entire pipeline -- feature set, model family, hyper-parameter
    search and all -- rather than just the final correlation, so it also
    catches selection bias introduced by the grid search. It is expensive, so
    it is run only for the single headline configuration.
    """
    rng = np.random.default_rng(seed)
    scoring = "neg_mean_absolute_error"

    def one_perm(i: int) -> float:
        r = np.random.default_rng(seed + 1000 + i)
        yp = r.permutation(y)
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            _, preds, _ = _fit_one_fold(estimator, grid, X, yp, groups, tr, te,
                                        scoring, config.INNER_CV_N_SPLITS)
            oof[te] = preds
        from scipy import stats
        return float(stats.spearmanr(yp, oof).statistic)

    null = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(one_perm)(i) for i in range(n_perm)
    )
    null = np.array([v for v in null if np.isfinite(v)], dtype=float)
    p = float((np.sum(null >= observed_spearman) + 1) / (len(null) + 1))
    return {
        "observed_spearman": float(observed_spearman),
        "p_value": p,
        "n_permutations": int(len(null)),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_q95": float(np.percentile(null, 95)),
    }
