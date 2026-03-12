"""
Feature scaler wrapper supporting StandardScaler, MinMaxScaler,
and RobustScaler from scikit-learn.  Fits *only* on training data
and applies the same transformation to val/test sets (no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils import get_logger

logger = get_logger(__name__)

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class DNSScaler:
    """Fit-once, transform-many feature scaler for DNS data.

    Args:
        method: Scaling strategy – ``"standard"`` (default), ``"minmax"``,
            or ``"robust"``.
        exclude_cols: Column names to leave unscaled (e.g., label, binary flags).
    """

    def __init__(
        self,
        method: str = "standard",
        exclude_cols: list[str] | None = None,
    ) -> None:
        if method not in _SCALERS:
            raise ValueError(f"Unknown scaler '{method}'. Use one of {list(_SCALERS)}")
        self.method = method
        self.exclude_cols = set(exclude_cols or [])
        self._scaler = _SCALERS[method]()
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "DNSScaler":
        """Fit scaler on *df* (training set only).

        Args:
            df: Training DataFrame.

        Returns:
            Fitted instance.
        """
        self._feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in self.exclude_cols
        ]
        self._scaler.fit(df[self._feature_cols])
        logger.info(
            "DNSScaler fitted",
            extra={"method": self.method, "n_features": len(self._feature_cols)},
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale *df* using fitted statistics.

        Args:
            df: DataFrame to transform.

        Returns:
            New DataFrame with scaled numeric columns.
        """
        df = df.copy()
        present = [c for c in self._feature_cols if c in df.columns]
        df[present] = self._scaler.transform(df[present])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit then transform (training set convenience method)."""
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse the scaling transformation."""
        df = df.copy()
        present = [c for c in self._feature_cols if c in df.columns]
        df[present] = self._scaler.inverse_transform(df[present])
        return df

    @property
    def feature_names(self) -> list[str]:
        """Names of the scaled features."""
        return self._feature_cols
