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
        max_train_samples: If set, stratified-downsample the training set to
            this many rows before fitting (SVM scales as O(n²)–O(n³)).
        random_state: Seed for reproducible downsampling.
        model_dir: Persistence directory.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        class_weight: str = "balanced",
        probability: bool = True,
        max_iter: int = 5000,
        max_train_samples: Optional[int] = None,
        random_state: int = 42,
        model_dir: str = "models",
    ) -> None:
        super().__init__(name="svm", model_dir=model_dir)
        self.max_train_samples = max_train_samples
        self.random_state = random_state
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
        if self.max_train_samples is not None and len(X_train) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            # Stratified downsampling: keep class proportions
            classes, counts = np.unique(y_train, return_counts=True)
            keep_idx: list[np.ndarray] = []
            for cls, cnt in zip(classes, counts):
                cls_idx = np.where(y_train == cls)[0]
                n_keep = max(1, int(round(self.max_train_samples * cnt / len(y_train))))
                keep_idx.append(rng.choice(cls_idx, size=min(n_keep, len(cls_idx)), replace=False))
            idx = np.concatenate(keep_idx)
            X_train, y_train = X_train[idx], y_train[idx]
            logger.info(
                "SVM training set downsampled",
                extra={"original": len(idx), "downsampled_to": len(X_train)},
            )

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
