"""
Random Forest detector for DNS attack classification.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base_detector import BaseDetector
from src.utils import get_logger

logger = get_logger(__name__)


class RandomForestDetector(BaseDetector):
    """Random Forest multi-class DNS attack detector.

    Args:
        n_estimators: Number of decision trees (default 200).
        max_depth: Maximum tree depth (default 20, ``None`` = unlimited).
        min_samples_split: Minimum samples to split an internal node.
        min_samples_leaf: Minimum samples in a leaf node.
        class_weight: ``"balanced"`` compensates for class imbalance.
        n_jobs: CPU parallelism (-1 = all cores).
        random_state: Reproducibility seed.
        model_dir: Directory for model persistence.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        n_jobs: int = -1,
        random_state: int = 42,
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="random_forest", model_dir=model_dir)
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._model = RandomForestClassifier(**self._params)

    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RandomForestDetector":
        """Train the Random Forest.

        Validation data is used for scoring only (no gradient updates).
        """
        logger.info(
            "Training RandomForest",
            extra={"n_samples": len(X_train), "n_estimators": self._params["n_estimators"]},
        )
        self._model.fit(X_train, y_train)
        self._is_fitted = True

        train_acc = self._model.score(X_train, y_train)
        logger.info("RandomForest training complete", extra={"train_acc": round(train_acc, 4)})
        if X_val is not None and y_val is not None:
            val_acc = self._model.score(X_val, y_val)
            logger.info("RandomForest validation", extra={"val_acc": round(val_acc, 4)})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return self._params

    @property
    def feature_importances(self) -> np.ndarray:
        """Gini importance for each feature."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet.")
        return self._model.feature_importances_
