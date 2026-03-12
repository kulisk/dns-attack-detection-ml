"""
One-Class SVM anomaly detector for DNS traffic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.svm import OneClassSVM

from src.models.base_detector import BaseDetector
from src.utils import get_logger

logger = get_logger(__name__)


class OneClassSVMDetector(BaseDetector):
    """One-Class SVM unsupervised anomaly detector.

    Args:
        kernel: Kernel type used in SVM.
        nu: Upper bound on the fraction of training errors (0 < nu < 1).
        gamma: Kernel coefficient.
        model_dir: Persistence directory.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str = "scale",
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="one_class_svm", model_dir=model_dir)
        self._params = dict(kernel=kernel, nu=nu, gamma=gamma)
        self._model = OneClassSVM(**self._params)

    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "OneClassSVMDetector":
        logger.info("Training OneClassSVM", extra={"n_samples": len(X_train)})
        self._model.fit(X_train)
        self._is_fitted = True
        logger.info("OneClassSVM training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Remap sklearn's 1/-1 to 0/1 (0=benign, 1=attack)."""
        raw = self._model.predict(X)
        return np.where(raw == 1, 0, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._model.decision_function(X)
        normalised = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        prob_attack = 1.0 - normalised
        return np.column_stack([1.0 - prob_attack, prob_attack])

    def get_params(self) -> dict:
        return self._params
