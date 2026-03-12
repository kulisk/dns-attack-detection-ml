"""
DBSCAN-based anomaly detector for DNS traffic.

DBSCAN classifies points as core, border, or noise (outlier).
Noise points are treated as anomalies (attacks).
"""
from __future__ import annotations

from typing import Any, Literal, Optional, cast

import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

from src.models.base_detector import BaseDetector
from src.utils import get_logger

logger = get_logger(__name__)

AlgorithmName = Literal["auto", "ball_tree", "kd_tree", "brute"]


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
        self._params: dict[str, float | int | str] = dict(
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            n_jobs=n_jobs,
        )
        self._model = DBSCAN(
            eps=self._eps(),
            min_samples=self._min_samples(),
            algorithm=self._algorithm_name(),
            n_jobs=self._n_jobs(),
        )
        self._reference_X: Optional[np.ndarray] = None
        self._nn_index: Optional[NearestNeighbors] = None

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
        if X_val is not None and y_val is not None:
            self._tune_params(X_train, X_val, y_val)

        self._model = DBSCAN(
            eps=self._eps(),
            min_samples=self._min_samples(),
            algorithm=self._algorithm_name(),
            n_jobs=self._n_jobs(),
        )
        self._model.fit(X_train)
        self._train_labels = self._model.labels_.copy()
        non_noise_mask = self._train_labels != -1
        if not np.any(non_noise_mask):
            logger.warning(
                "DBSCAN produced no clusters; falling back to full benign reference set",
                extra={"eps": self._params["eps"], "min_samples": self._params["min_samples"]},
            )
            self._reference_X = X_train.copy()
        else:
            self._reference_X = X_train[non_noise_mask].copy()
        self._build_index()
        self._is_fitted = True
        n_noise = int(np.sum(self._train_labels == -1))
        n_clusters = len(set(self._train_labels)) - (1 if -1 in self._train_labels else 0)
        reference_points = len(self._reference_X)
        logger.info(
            "DBSCAN fitting complete",
            extra={
                "n_clusters": n_clusters,
                "noise_points": n_noise,
                "reference_points": int(reference_points),
                "eps": self._params["eps"],
                "min_samples": self._params["min_samples"],
            },
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each test point to the nearest training cluster label.

        Points without a close-enough neighbour are noise (labelled -1).
        Returns 0 for benign (cluster-member) and 1 for attack (noise).
        """
        if self._reference_X is None:
            raise RuntimeError("DBSCANDetector not fitted yet.")
        distances, _ = self._ensure_index().kneighbors(X)
        return (distances[:, 0] > self._params["eps"]).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return binary probability: prob_attack based on nearest distance."""
        if self._reference_X is None:
            raise RuntimeError("DBSCANDetector not fitted yet.")
        distances, _ = self._ensure_index().kneighbors(X)
        distances = distances[:, 0]
        # Normalise distances to [0, 1] probability of being attack
        max_d = self._model.eps * 3 + 1e-9
        prob_attack = np.clip(distances / max_d, 0.0, 1.0)
        return np.column_stack([1.0 - prob_attack, prob_attack])

    def get_params(self) -> dict[str, float | int | str]:
        return self._params

    def save(self, filename: Optional[str] = None) -> str:
        if not self._is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before saving.")
        path = self.model_dir / (filename or f"{self.name}.joblib")
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self._feature_names,
                "params": self._params,
                "reference_X": self._reference_X,
            },
            path,
            compress=3,
        )
        logger.info("Model saved", extra={"model": self.name, "path": str(path)})
        return str(path)

    def load(self, filename: Optional[str] = None) -> "DBSCANDetector":
        path = self.model_dir / (filename or f"{self.name}.joblib")
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = cast(dict[str, Any], joblib.load(path))
        self._model = payload["model"]
        self._feature_names = payload.get("feature_names", [])
        self._params = cast(dict[str, float | int | str], payload.get("params", self._model.get_params()))
        self._reference_X = payload.get("reference_X")
        self._build_index()
        self._is_fitted = True
        logger.info("Model loaded", extra={"model": self.name, "path": str(path)})
        return self

    def _build_index(self) -> None:
        if self._reference_X is None or len(self._reference_X) == 0:
            self._nn_index = None
            return
        self._nn_index = NearestNeighbors(
            n_neighbors=1,
            algorithm=self._algorithm_name(),
            n_jobs=self._n_jobs(),
        )
        self._nn_index.fit(self._reference_X)

    def _ensure_index(self) -> NearestNeighbors:
        if self._nn_index is None:
            self._build_index()
        if self._nn_index is None:
            raise RuntimeError("DBSCANDetector nearest-neighbor index is not available.")
        return self._nn_index

    def _tune_params(self, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Tune DBSCAN using binary benign-vs-attack validation performance."""
        y_val_binary = (np.asarray(y_val) != 0).astype(int)
        min_candidates = sorted({3, 5, 10, 20, int(self._params.get("min_samples", 5))})
        max_k = min(max(min_candidates) + 1, len(X_train))
        if max_k < 2:
            return

        nn = NearestNeighbors(
            n_neighbors=max_k,
            algorithm=self._algorithm_name(),
            n_jobs=self._n_jobs(),
        )
        nn.fit(X_train)
        distances, _ = nn.kneighbors(X_train)

        candidate_eps: set[float] = {float(self._params.get("eps", 0.5))}
        for min_samples in min_candidates:
            kth_distances = distances[:, min(min_samples, distances.shape[1] - 1)]
            for quantile in (0.75, 0.85, 0.9, 0.95, 0.97, 0.99):
                candidate_eps.add(round(float(np.quantile(kth_distances, quantile)), 4))

        best: Optional[tuple[float, float, int, int, float, int]] = None
        for eps in sorted(candidate_eps):
            if eps <= 0:
                continue
            for min_samples in min_candidates:
                candidate = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    algorithm=self._algorithm_name(),
                    n_jobs=self._n_jobs(),
                )
                labels = candidate.fit_predict(X_train)
                reference_mask = labels != -1
                if not np.any(reference_mask):
                    continue
                candidate_nn = NearestNeighbors(
                    n_neighbors=1,
                    algorithm=self._algorithm_name(),
                    n_jobs=self._n_jobs(),
                )
                candidate_nn.fit(X_train[reference_mask])
                val_distances, _ = candidate_nn.kneighbors(X_val)
                y_pred = (val_distances[:, 0] > eps).astype(int)
                f1 = float(f1_score(y_val_binary, y_pred, average="weighted", zero_division=0))
                acc = float(accuracy_score(y_val_binary, y_pred))
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = int(np.sum(labels == -1))
                score = (f1, acc, -n_noise, n_clusters, eps, min_samples)
                if best is None or score > best:
                    best = score

        if best is None:
            logger.warning("DBSCAN tuning found no usable parameter combination")
            return

        best_f1, best_acc, _, best_clusters, best_eps, best_min_samples = best
        self._params.update(eps=float(best_eps), min_samples=int(best_min_samples))
        logger.info(
            "DBSCAN parameters tuned",
            extra={
                "best_eps": float(best_eps),
                "best_min_samples": int(best_min_samples),
                "val_f1_weighted": round(best_f1, 4),
                "val_accuracy": round(best_acc, 4),
                "val_clusters": int(best_clusters),
            },
        )

    def _eps(self) -> float:
        return float(self._params.get("eps", 0.5))

    def _min_samples(self) -> int:
        return int(self._params.get("min_samples", 5))

    def _algorithm_name(self) -> AlgorithmName:
        value = str(self._params.get("algorithm", "auto"))
        if value not in {"auto", "ball_tree", "kd_tree", "brute"}:
            value = "auto"
        return cast(AlgorithmName, value)

    def _n_jobs(self) -> int:
        return int(self._params.get("n_jobs", -1))
