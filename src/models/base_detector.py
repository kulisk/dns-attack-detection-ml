"""
Abstract base class for all DNS Attack Detection models.
Enforces a common interface for training, prediction, persistence,
and hyperparameter tuning.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class BaseDetector(ABC):
    """Shared interface for all supervised and unsupervised detectors.

    Subclasses must implement :meth:`fit`, :meth:`predict`,
    :meth:`predict_proba`, and :meth:`get_params`.
    """

    def __init__(self, name: str, model_dir: str = "models") -> None:
        self.name = name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any = None
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Abstract methods                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseDetector":
        """Train the model on the provided data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (shape: [n_samples, n_classes])."""

    @abstractmethod
    def get_params(self) -> dict:
        """Return the model hyper-parameters."""

    # ------------------------------------------------------------------ #
    # Concrete methods                                                    #
    # ------------------------------------------------------------------ #

    def save(self, filename: Optional[str] = None) -> str:
        """Persist the model to disk with joblib.

        Args:
            filename: Override the default ``<name>.joblib`` filename.

        Returns:
            Absolute path of the saved file.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before saving.")
        path = self.model_dir / (filename or f"{self.name}.joblib")
        joblib.dump(
            {"model": self._model, "feature_names": self._feature_names},
            path,
            compress=3,
        )
        logger.info("Model saved", extra={"model": self.name, "path": str(path)})
        return str(path)

    def load(self, filename: Optional[str] = None) -> "BaseDetector":
        """Load a persisted model from disk.

        Args:
            filename: Override the default ``<name>.joblib`` filename.

        Returns:
            Self (supports chaining).
        """
        path = self.model_dir / (filename or f"{self.name}.joblib")
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        self._model = payload["model"]
        self._feature_names = payload.get("feature_names", [])
        self._is_fitted = True
        logger.info("Model loaded", extra={"model": self.name, "path": str(path)})
        return self

    def set_feature_names(self, names: list[str]) -> None:
        """Store feature names for later reference / importance plots."""
        self._feature_names = names

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self._is_fitted})"
