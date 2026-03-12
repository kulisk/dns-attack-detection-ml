"""
Isolation Forest anomaly detector for DNS traffic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.base_detector import BaseDetector
from src.utils import get_logger

logger = get_logger(__name__)


class IsolationForestDetector(BaseDetector):
    """Isolation Forest unsupervised anomaly detector.

    Trained on *benign* traffic only; assigns negative anomaly scores
    to suspicious samples.

    Args:
        n_estimators: Number of isolation trees.
        contamination: Expected fraction of anomalies in training data.
        max_samples: Subsampling size per tree.
        random_state: Seed.
        n_jobs: CPU parallelism.
        model_dir: Persistence directory.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        max_samples: str | int = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="isolation_forest", model_dir=model_dir)
        self._params = dict(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._model = IsolationForest(**self._params)

    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "IsolationForestDetector":
        logger.info(
            "Training IsolationForest",
            extra={"n_samples": len(X_train), "contamination": self._params["contamination"]},
        )
        self._model.fit(X_train)
        self._is_fitted = True
        logger.info("IsolationForest training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (normal) or -1 (anomaly) per sklearn convention.
        We remap to 0 (benign) and 1 (attack) for consistency.
        """
        raw = self._model.predict(X)
        return np.where(raw == 1, 0, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores normalised to [0, 1].

        Higher value → more anomalous.
        """
        scores = self._model.decision_function(X)
        # Shift and scale to [0, 1]: low decision_function = more anomalous
        normalised = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        prob_attack = 1.0 - normalised
        return np.column_stack([1.0 - prob_attack, prob_attack])

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw isolation scores (negative = more anomalous)."""
        return self._model.decision_function(X)

    def get_params(self) -> dict:
        return self._params
