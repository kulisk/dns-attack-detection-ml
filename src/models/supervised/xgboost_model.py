"""
XGBoost detector for DNS attack classification.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from xgboost import XGBClassifier

from src.models.base_detector import BaseDetector
from src.utils import get_logger

logger = get_logger(__name__)


class XGBoostDetector(BaseDetector):
    """XGBoost multi-class DNS attack detector.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage (eta).
        subsample: Fraction of samples per tree.
        colsample_bytree: Fraction of features per tree.
        scale_pos_weight: Balances positive/negative weights (binary mode).
        n_jobs: CPU parallelism.
        random_state: Seed.
        model_dir: Persistence directory.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="xgboost", model_dir=model_dir)
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
        )
        self._model = XGBClassifier(**self._params)

    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostDetector":
        logger.info(
            "Training XGBoost",
            extra={"n_samples": len(X_train), "n_estimators": self._params["n_estimators"]},
        )
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self._is_fitted = True
        train_acc = self._model.score(X_train, y_train)
        logger.info("XGBoost training complete", extra={"train_acc": round(train_acc, 4)})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return self._params

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_
