"""
Data cleaner module: handles missing values, outliers, duplicates,
and infinite values in DNS feature DataFrames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Clean raw DNS feature DataFrames.

    Args:
        missing_strategy: How to impute missing values –
            ``"median"`` (default), ``"mean"``, or ``"drop"``.
        outlier_method: ``"iqr"`` (default), ``"zscore"``, or ``"none"``.
        outlier_threshold: Z-score cutoff when *outlier_method* = ``"zscore"``.
        drop_duplicates: Whether to remove exact duplicate rows.
    """

    def __init__(
        self,
        missing_strategy: str = "median",
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        drop_duplicates: bool = True,
    ) -> None:
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.drop_duplicates = drop_duplicates
        # Learned statistics (fitted on training set only)
        self._medians: pd.Series | None = None
        self._means: pd.Series | None = None
        self._iqr_bounds: pd.DataFrame | None = None
        self._zscore_stats: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> "DataCleaner":
        """Compute imputation and outlier statistics from the training set.

        Args:
            df: Training DataFrame (no test/val data included).
            label_col: Column name of the target – excluded from stats.

        Returns:
            Fitted instance (supports chaining).
        """
        num_cols = self._numeric_cols(df, exclude=label_col)
        self._medians = df[num_cols].median()
        self._means = df[num_cols].mean()

        if self.outlier_method == "iqr":
            q1 = df[num_cols].quantile(0.25)
            q3 = df[num_cols].quantile(0.75)
            iqr = q3 - q1
            self._iqr_bounds = pd.DataFrame(
                {"lower": q1 - 1.5 * iqr, "upper": q3 + 1.5 * iqr}
            )
        elif self.outlier_method == "zscore":
            self._zscore_stats = pd.DataFrame(
                {"mean": df[num_cols].mean(), "std": df[num_cols].std()}
            )
        logger.info("DataCleaner fitted", extra={"num_features": len(num_cols)})
        return self

    def transform(self, df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """Apply cleaning to a DataFrame using fitted statistics.

        This method must be called *after* :meth:`fit`.  Applies the
        same transformations to train, validation, and test sets
        preventing data leakage.

        Args:
            df: DataFrame to clean.
            label_col: Target column (preserved unchanged).

        Returns:
            Cleaned copy of *df*.
        """
        df = df.copy()
        num_cols = self._numeric_cols(df, exclude=label_col)

        # 1. Replace ±inf with NaN
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

        # 2. Impute missing values
        df = self._impute(df, num_cols)

        # 3. Clip outliers
        df = self._clip_outliers(df, num_cols)

        # 4. Remove duplicates
        if self.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            if removed > 0:
                logger.debug("Duplicates removed", extra={"count": removed})

        return df.reset_index(drop=True)

    def fit_transform(
        self, df: pd.DataFrame, label_col: str = "label"
    ) -> pd.DataFrame:
        """Convenience wrapper: fit then transform."""
        return self.fit(df, label_col).transform(df, label_col)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _impute(self, df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
        missing = df[num_cols].isna().sum()
        total_missing = missing.sum()
        if total_missing == 0:
            return df
        logger.debug("Imputing missing values", extra={"total_missing": int(total_missing)})

        if self.missing_strategy == "drop":
            df = df.dropna(subset=num_cols)
        elif self.missing_strategy == "mean" and self._means is not None:
            for col in num_cols:
                df[col] = df[col].fillna(self._means.get(col, 0))
        else:  # median (default)
            if self._medians is not None:
                for col in num_cols:
                    df[col] = df[col].fillna(self._medians.get(col, 0))
        return df

    def _clip_outliers(self, df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
        if self.outlier_method == "iqr" and self._iqr_bounds is not None:
            for col in num_cols:
                lo = self._iqr_bounds.loc[col, "lower"]
                hi = self._iqr_bounds.loc[col, "upper"]
                df[col] = df[col].clip(lower=lo, upper=hi)
        elif self.outlier_method == "zscore" and self._zscore_stats is not None:
            for col in num_cols:
                mean = self._zscore_stats.loc[col, "mean"]
                std = self._zscore_stats.loc[col, "std"]
                if std > 0:
                    df[col] = df[col].clip(
                        lower=mean - self.outlier_threshold * std,
                        upper=mean + self.outlier_threshold * std,
                    )
        return df

    @staticmethod
    def _numeric_cols(df: pd.DataFrame, exclude: str = "label") -> list[str]:
        return [
            c for c in df.select_dtypes(include=[np.number]).columns if c != exclude
        ]
