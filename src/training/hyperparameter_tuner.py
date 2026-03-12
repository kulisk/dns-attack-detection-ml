"""
Hyperparameter tuning using scikit-learn RandomizedSearchCV / GridSearchCV.
Supports Isolation Forest, Random Forest, XGBoost, and SVM.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.utils import get_logger

logger = get_logger(__name__)

# Per-model search spaces
_SEARCH_SPACES: dict[str, dict] = {
    "random_forest": {
        "n_estimators": randint(100, 500),
        "max_depth": [10, 15, 20, 30, None],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
    },
    "xgboost": {
        "n_estimators": randint(100, 500),
        "max_depth": randint(4, 12),
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    },
    "svm": {
        "C": loguniform(0.01, 100),
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "poly"],
    },
}


class HyperparameterTuner:
    """Wraps sklearn random/grid search for DNS attack detection models.

    Args:
        method: ``"random_search"`` (default) or ``"grid_search"``.
        cv_folds: Number of cross-validation folds.
        n_iter: Iterations for random search.
        scoring: Optimisation metric (default ``"f1_weighted"``).
        n_jobs: CPU parallelism.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        method: str = "random_search",
        cv_folds: int = 5,
        n_iter: int = 50,
        scoring: str = "f1_weighted",
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> None:
        self.method = method
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def tune(
        self,
        model_name: str,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[dict] = None,
    ) -> dict:
        """Run hyperparameter search.

        Args:
            model_name: Key for the default parameter space.
            estimator: scikit-learn compatible estimator.
            X_train: Training features.
            y_train: Training labels.
            param_grid: Override the default search space.

        Returns:
            Dict with ``best_params``, ``best_score``, and ``cv_results``.
        """
        space = param_grid or _SEARCH_SPACES.get(model_name, {})
        if not space:
            logger.warning(
                "No search space defined for model – using defaults",
                extra={"model": model_name},
            )
            return {"best_params": {}, "best_score": 0.0, "cv_results": {}}

        logger.info(
            "Starting hyperparameter tuning",
            extra={"model": model_name, "method": self.method, "n_iter": self.n_iter},
        )

        if self.method == "grid_search":
            search = GridSearchCV(
                estimator,
                param_grid=space,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        else:
            search = RandomizedSearchCV(
                estimator,
                param_distributions=space,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=42,
            )

        search.fit(X_train, y_train)

        result = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "cv_results": search.cv_results_,
        }
        logger.info(
            "Hyperparameter tuning complete",
            extra={"best_score": round(result["best_score"], 4), "best_params": result["best_params"]},
        )
        return result

    @staticmethod
    def get_search_space(model_name: str) -> dict:
        """Return the default search space for *model_name*."""
        return _SEARCH_SPACES.get(model_name, {})
