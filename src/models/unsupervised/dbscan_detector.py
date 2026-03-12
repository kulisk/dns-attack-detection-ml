"""
DBSCAN-based anomaly detector for DNS traffic.

DBSCAN classifies points as core, border, or noise (outlier).
Noise points are treated as anomalies (attacks).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN

from src.models.base_detector import BaseDetector
from src.utils import get_logger

logger = get_logger(__name__)


class DBSCANDetector(BaseDetector):
    """DBSCAN clustering-based anomaly detector.

    Points assigned label ``-1`` (noise) by DBSCAN are flagged as attacks.

    Args:
        eps: Maximum distance between two samples in the same neighbourhood.
        min_samples: Minimum samples in a core-point neighbourhood.
        algorithm: Nearest-neighbour algorithm.
        n_jobs: CPU parallelism.
        model_dir: Persistence directory.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        algorithm: str = "auto",
        n_jobs: int = -1,
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="dbscan", model_dir=model_dir)
        self._params = dict(
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            n_jobs=n_jobs,
        )
        self._model = DBSCAN(**self._params)
        self._train_X: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "DBSCANDetector":
        """Fit DBSCAN and memorise training embeddings for inference.

        Note: DBSCAN is transductive, so we keep training data to
        extend cluster membership at prediction time using a nearest-
        neighbour heuristic.
        """
        logger.info("Fitting DBSCAN", extra={"n_samples": len(X_train)})
        self._model.fit(X_train)
        self._train_X = X_train.copy()
        self._train_labels = self._model.labels_.copy()
        self._is_fitted = True
        n_noise = int(np.sum(self._train_labels == -1))
        n_clusters = len(set(self._train_labels)) - (1 if -1 in self._train_labels else 0)
        logger.info(
            "DBSCAN fitting complete",
            extra={"n_clusters": n_clusters, "noise_points": n_noise},
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each test point to the nearest training cluster label.

        Points without a close-enough neighbour are noise (labelled -1).
        Returns 0 for benign (cluster-member) and 1 for attack (noise).
        """
        from sklearn.metrics import pairwise_distances_argmin_min

        if self._train_X is None:
            raise RuntimeError("DBSCANDetector not fitted yet.")

        closest_idx, distances = pairwise_distances_argmin_min(X, self._train_X)
        assigned = self._train_labels[closest_idx]
        assigned = np.where(distances > self._model.eps, -1, assigned)
        # Remap: -1 (noise) → 1 (attack), else → 0 (benign)
        return np.where(assigned == -1, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return binary probability: prob_attack based on nearest distance."""
        from sklearn.metrics import pairwise_distances_argmin_min
        if self._train_X is None:
            raise RuntimeError("DBSCANDetector not fitted yet.")
        _, distances = pairwise_distances_argmin_min(X, self._train_X)
        # Normalise distances to [0, 1] probability of being attack
        max_d = self._model.eps * 3 + 1e-9
        prob_attack = np.clip(distances / max_d, 0.0, 1.0)
        return np.column_stack([1.0 - prob_attack, prob_attack])

    def get_params(self) -> dict:
        return self._params
