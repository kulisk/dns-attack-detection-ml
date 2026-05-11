"""
End-to-end ML training pipeline for DNS attack detection.

Orchestrates data loading → preprocessing → feature engineering →
imbalance handling → model training → evaluation → persistence.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from imblearn.over_sampling import SMOTE

from src.data_collection import DatasetLoader
from src.feature_engineering import DNSFeatureExtractor
from src.models.base_detector import BaseDetector
from src.preprocessing import DataCleaner, DNSScaler, LabelEncoder
from src.utils import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)

NON_FEATURE_COLS = {"label", "is_attack"}


class ModelTrainer:
    """Full training pipeline that fits a given detector.

    Args:
        config_path: Path to the main YAML configuration file.
        model_dir: Directory for saving trained models.
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        model_dir: str = "models",
    ) -> None:
        self.cfg = ConfigLoader(config_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = DataCleaner(
            missing_strategy=self.cfg.get("preprocessing.handle_missing", "median"),
            outlier_method=self.cfg.get("preprocessing.outlier_method", "iqr"),
            outlier_threshold=float(self.cfg.get("preprocessing.outlier_threshold", 3.0)),
        )
        self.scaler = DNSScaler(
            method=self.cfg.get("preprocessing.scaler", "standard"),
            exclude_cols=list(NON_FEATURE_COLS),
        )
        self.extractor = DNSFeatureExtractor()
        dataset_source = str(self.cfg.get("dataset.source", "synthetic")).lower()
        known_classes = self.cfg.get("attack_types") if dataset_source == "synthetic" else None
        self.label_encoder = LabelEncoder(known_classes=known_classes)
        self._categorical_maps: dict[str, dict[str, int]] = {}

        maps_path = self.model_dir / "categorical_maps.joblib"
        if maps_path.exists():
            try:
                import joblib  # pyright: ignore[reportMissingTypeStubs]
                self._categorical_maps = cast(
                    dict[str, dict[str, int]],
                    joblib.load(maps_path),  # pyright: ignore[reportUnknownMemberType]
                )
            except Exception:
                self._categorical_maps = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        model: BaseDetector,
        use_smote: bool = True,
    ) -> dict[str, Any]:
        """Execute the full training + evaluation pipeline.

        Args:
            model: Detector instance to train.
            use_smote: Apply SMOTE oversampling on the training set.

        Returns:
            Dict with keys ``"model"``, ``"metrics"``, ``"model_path"``.
        """
        # ── 1. Load data ──────────────────────────────────────────────
        logger.info("Loading dataset …")
        loader = DatasetLoader(
            dataset_dir=self.cfg.get("paths.datasets", "datasets"),
            source=self.cfg.get("dataset.source", "synthetic"),
            test_size=float(self.cfg.get("dataset.test_size", 0.15)),
            val_size=float(self.cfg.get("dataset.val_size", 0.15)),
            random_state=int(self.cfg.get("dataset.random_state", 42)),
            label_col=self.cfg.get("label_column", "label"),
        )
        df = loader.load()
        train_df, val_df, test_df = loader.split(df)

        # ── 2. Feature engineering ────────────────────────────────────
        logger.info("Extracting features …")
        train_df = self.extractor.transform(train_df)
        val_df = self.extractor.transform(val_df)
        test_df = self.extractor.transform(test_df)

        # ── 3. Clean (fit on train only) ──────────────────────────────
        logger.info("Cleaning data …")
        label_col = self.cfg.get("label_column", "label")
        train_df = self.cleaner.fit_transform(train_df, label_col)
        val_df = self.cleaner.transform(val_df, label_col)
        test_df = self.cleaner.transform(test_df, label_col)

        # ── 4. Encode labels ──────────────────────────────────────────
        y_train = self.label_encoder.fit_transform(train_df[label_col])
        y_val = self.label_encoder.transform(val_df[label_col])
        y_test = self.label_encoder.transform(test_df[label_col])

        feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
        train_df, val_df, test_df = self._encode_categorical_features(
            train_df,
            val_df,
            test_df,
            feature_cols,
        )

        # ── 5. Scale (fit on train only) ──────────────────────────────
        logger.info("Scaling features …")
        train_df = self.scaler.fit(train_df[feature_cols]).transform(train_df[feature_cols])
        val_df_s = self.scaler.transform(val_df[feature_cols])
        test_df_s = self.scaler.transform(test_df[feature_cols])

        X_train = train_df.values
        X_val = val_df_s.values
        X_test = test_df_s.values

        # ── 6. Imbalance handling (SMOTE on training only) ────────────
        unsupervised_model_names = {
            "isolation_forest",
            "one_class_svm",
            "dbscan",
            "autoencoder",
        }
        is_unsupervised = model.name in unsupervised_model_names

        if is_unsupervised:
            benign_label_idx = int(self.label_encoder.classes.index("benign"))
            benign_mask = y_train == benign_label_idx
            benign_count = int(benign_mask.sum())
            if benign_count == 0:
                raise ValueError("Unsupervised training requires benign samples, but none were found.")
            X_train = X_train[benign_mask]
            y_train = y_train[benign_mask]
            logger.info(
                "Unsupervised training uses benign-only subset",
                extra={"benign_samples": benign_count},
            )

        if (not is_unsupervised) and use_smote and self.cfg.get("imbalance.strategy", "smote") == "smote":
            logger.info("Applying SMOTE …")
            X_train, y_train = self._apply_smote(X_train, y_train)

        # ── 7. Train ──────────────────────────────────────────────────
        model.set_feature_names(list(train_df.columns))
        logger.info(f"Training model: {model.name}")
        t0 = time.perf_counter()
        model.fit(X_train, y_train, X_val, y_val)
        elapsed = time.perf_counter() - t0
        logger.info(f"Training done in {elapsed:.2f}s")

        # ── 8. Evaluate ───────────────────────────────────────────────
        from src.evaluation import Evaluator
        evaluator = Evaluator(
            class_names=self.label_encoder.classes,
            output_dir="reports",
        )
        metrics = evaluator.evaluate(model, X_test, y_test, split="test")

        # ── 9. Save model ─────────────────────────────────────────────
        model_path = model.save()

        # Also persist preprocessing objects
        self._save_preprocessors()

        return {"model": model, "metrics": metrics, "model_path": model_path}

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform a raw DataFrame into scaled features and integer labels.

        This is used for inference and evaluation outside of the full pipeline.
        """
        label_col = self.cfg.get("label_column", "label")
        df = self.extractor.transform(df)
        df = self.cleaner.transform(df, label_col)
        y = self.label_encoder.transform(df[label_col])
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        for col in feature_cols:
            if col not in df.columns:
                continue
            if col in self._categorical_maps:
                mapping = self._categorical_maps[col]
                df[col] = (
                    df[col]
                    .astype(str)
                    .fillna("__nan__")
                    .map(mapping)
                    .fillna(-1)
                    .astype(float)
                )
        df_s = self.scaler.transform(df[feature_cols])
        return df_s.values, y

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        k = int(self.cfg.get("imbalance.k_neighbors", 5))
        _, counts = np.unique(y, return_counts=True)
        # Only SMOTE if every class has at least k+1 samples
        min_count = counts.min()
        if min_count <= k:
            logger.warning(
                "SMOTE skipped – some class has too few samples",
                extra={"min_count": int(min_count), "k_neighbors": k},
            )
            return X, y
        smote = SMOTE(k_neighbors=k, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(
            "SMOTE applied",
            extra={"before": len(y), "after": len(y_res)},
        )
        return X_res, y_res

    def _save_preprocessors(self) -> None:
        """Persist the fitted cleaner and scaler for inference use."""
        import joblib  # pyright: ignore[reportMissingTypeStubs]
        joblib.dump(self.cleaner, self.model_dir / "cleaner.joblib")  # pyright: ignore[reportUnknownMemberType]
        joblib.dump(self.scaler, self.model_dir / "scaler.joblib")  # pyright: ignore[reportUnknownMemberType]
        joblib.dump(self.label_encoder, self.model_dir / "label_encoder.joblib")  # pyright: ignore[reportUnknownMemberType]
        joblib.dump(self._categorical_maps, self.model_dir / "categorical_maps.joblib")  # pyright: ignore[reportUnknownMemberType]
        logger.info("Preprocessors saved", extra={"dir": str(self.model_dir)})

    def _encode_categorical_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Encode string-like feature columns to numeric values."""
        self._categorical_maps = {}
        for col in feature_cols:
            if col not in train_df.columns:
                continue
            dtype = train_df[col].dtype
            if (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, CategoricalDtype)
                or pd.api.types.is_bool_dtype(dtype)
            ):
                train_values = train_df[col].astype(str).fillna("__nan__")
                unique_values = list(pd.unique(train_values))
                mapping = {value: idx for idx, value in enumerate(unique_values)}
                self._categorical_maps[col] = mapping
                train_df[col] = train_values.map(mapping).astype(float)
                val_df[col] = val_df[col].astype(str).fillna("__nan__").map(mapping).fillna(-1).astype(float)
                test_df[col] = test_df[col].astype(str).fillna("__nan__").map(mapping).fillna(-1).astype(float)
        if self._categorical_maps:
            logger.info("Categorical features encoded", extra={"n_categorical": len(self._categorical_maps)})
        return train_df, val_df, test_df
