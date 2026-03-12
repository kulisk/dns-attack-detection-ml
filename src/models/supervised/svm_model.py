"""
SVM detector for DNS attack classification.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.svm import SVC

from src.models.base_detector import BaseDetector
from src.utils import get_logger

logger = get_logger(__name__)


class SVMDetector(BaseDetector):
    """Support Vector Machine multi-class DNS attack detector.

    Args:
        kernel: SVM kernel type (``"rbf"``, ``"linear"``, etc.).
        C: Regularisation parameter.
        gamma: Kernel coefficient.
        class_weight: ``"balanced"`` adjusts for imbalanced classes.
        probability: Enable probability estimates (requires cross-validation;
            slower training but enables :meth:`predict_proba`).
        max_iter: Hard limit on training iterations.
        model_dir: Persistence directory.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        class_weight: str = "balanced",
        probability: bool = True,
        max_iter: int = 1000,
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="svm", model_dir=model_dir)
        self._params = dict(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            max_iter=max_iter,
        )
        self._model = SVC(**self._params)

    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "SVMDetector":
        logger.info(
            "Training SVM",
            extra={
                "n_samples": len(X_train),
                "kernel": self._params["kernel"],
                "C": self._params["C"],
            },
        )
        self._model.fit(X_train, y_train)
        self._is_fitted = True
        train_acc = self._model.score(X_train, y_train)
        logger.info("SVM training complete", extra={"train_acc": round(train_acc, 4)})
        if X_val is not None and y_val is not None:
            val_acc = self._model.score(X_val, y_val)
            logger.info("SVM validation", extra={"val_acc": round(val_acc, 4)})
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._params.get("probability", False):
            raise RuntimeError("SVM was initialised with probability=False.")
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return self._params
