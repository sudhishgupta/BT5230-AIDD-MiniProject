"""
The model zoo and their hyper-parameter grids.

Every estimator is wrapped in a Pipeline that standardises features first.
That matters for two reasons: the descriptor block mixes units that differ by
orders of magnitude (molecular weight in the tens of thousands next to amino
acid fractions in [0,1]), and the penalised linear models are not
scale-invariant.

Grids are deliberately small.  They sit inside a nested cross-validation, so
every extra grid point multiplies the total fit count by the inner fold count;
a sprawling grid would buy a little tuning accuracy at the cost of making the
whole study impractical to re-run.  The chosen ranges bracket the regime that
matters for n~137 with p from 4 to 1000: heavy regularisation.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC

from src import config


class PLSWrapper(PLSRegression):
    """
    PLSRegression that plays nicely inside a shared grid search.

    Two adjustments:

    * ``predict`` returns a 1-D vector like every other estimator, instead of
      the (n, 1) column PLSRegression emits.
    * ``n_components`` is clamped to what the data can actually support. The
      same grid is applied to every feature set, and the trivial baseline has
      only four columns, so a grid point asking for 5 or 8 components is valid
      for the 480-dimensional embeddings and impossible for the baseline.
      Clamping lets one grid serve all feature sets; the alternative -- a
      different grid per feature set -- would make the comparison between
      feature sets less like-for-like.
    """

    def fit(self, X, y=None, **kwargs):
        import numpy as _np
        X = _np.asarray(X)
        max_components = max(1, min(X.shape[0] - 1, X.shape[1]))
        if self.n_components > max_components:
            self.n_components = max_components
        return super().fit(X, y, **kwargs)

    def predict(self, X, copy=True):
        return super().predict(X, copy=copy).ravel()


def _base_steps() -> list:
    # VarianceThreshold removes constant columns, which appear whenever a
    # liability motif is absent from every antibody in a given fold.  Leaving
    # them in makes the scaler emit NaNs.
    return [("var", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler())]


# --------------------------------------------------------------------------
# Regressors
# --------------------------------------------------------------------------

def regression_zoo(seed: int = config.SEED) -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, param_grid)"""
    zoo: dict[str, tuple[Pipeline, dict]] = {}

    zoo["ridge"] = (
        Pipeline(_base_steps() + [("model", Ridge(random_state=None))]),
        {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
    )

    zoo["elasticnet"] = (
        Pipeline(_base_steps() + [("model", ElasticNet(max_iter=20000, random_state=seed))]),
        {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.9]},
    )

    zoo["pls"] = (
        Pipeline(_base_steps() + [("model", PLSWrapper())]),
        {"model__n_components": [1, 2, 3, 5, 8]},
    )

    zoo["svr_rbf"] = (
        Pipeline(_base_steps() + [("model", SVR(kernel="rbf"))]),
        {"model__C": [1.0, 10.0, 100.0], "model__gamma": ["scale", 0.01],
         "model__epsilon": [0.1]},
    )

    zoo["knn"] = (
        Pipeline(_base_steps() + [("model", KNeighborsRegressor())]),
        {"model__n_neighbors": [3, 5, 10, 20], "model__weights": ["distance"]},
    )

    # The two tree ensembles dominate the runtime of the whole study, because
    # every grid point is refitted for each inner fold of each outer fold of
    # each repeat.  Their grids are kept to four points apiece: the leaf-size
    # and feature-subsample axes are the ones that actually matter at n~137,
    # and depth is better controlled through min_samples_leaf than through a
    # hard cap.
    zoo["random_forest"] = (
        Pipeline(_base_steps() + [
            ("model", RandomForestRegressor(n_estimators=200, random_state=seed,
                                            n_jobs=1))]),
        {"model__min_samples_leaf": [1, 5],
         "model__max_features": ["sqrt", 0.3]},
    )

    try:
        from lightgbm import LGBMRegressor
        zoo["lightgbm"] = (
            Pipeline(_base_steps() + [
                ("model", LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1,
                                        n_estimators=300, colsample_bytree=0.6))]),
            {"model__learning_rate": [0.03, 0.1],
             "model__num_leaves": [7, 15]},
        )
    except ImportError:
        pass

    return zoo


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------

def classification_zoo(seed: int = config.SEED) -> dict[str, tuple[Pipeline, dict]]:
    zoo: dict[str, tuple[Pipeline, dict]] = {}

    # class_weight='balanced' throughout: the positive class is ~10-20% by
    # construction (it is the unfavourable tail), and without reweighting the
    # models simply predict the majority class.
    zoo["logistic"] = (
        Pipeline(_base_steps() + [
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced",
                                         random_state=seed))]),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    )

    # probability=False on purpose: the ranking metrics (ROC-AUC, PR-AUC) only
    # need a monotone score, and decision_function supplies one directly. Platt
    # scaling would add an internal cross-validation to every fit, and it is
    # deprecated in scikit-learn 1.9 anyway.  _predict_scores falls back to
    # decision_function automatically.
    zoo["svc_rbf"] = (
        Pipeline(_base_steps() + [
            ("model", SVC(kernel="rbf", probability=False,
                          class_weight="balanced", random_state=seed))]),
        {"model__C": [1.0, 10.0, 100.0], "model__gamma": ["scale", 0.01]},
    )

    zoo["random_forest"] = (
        Pipeline(_base_steps() + [
            ("model", RandomForestClassifier(n_estimators=200, random_state=seed,
                                             class_weight="balanced_subsample",
                                             n_jobs=1))]),
        {"model__min_samples_leaf": [1, 3],
         "model__max_features": ["sqrt", 0.3]},
    )

    try:
        from lightgbm import LGBMClassifier
        zoo["lightgbm"] = (
            Pipeline(_base_steps() + [
                ("model", LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1,
                                         n_estimators=200, min_child_samples=5,
                                         class_weight="balanced"))]),
            {"model__learning_rate": [0.03, 0.1], "model__num_leaves": [7, 15]},
        )
    except ImportError:
        pass

    return zoo


def grid_size(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n
