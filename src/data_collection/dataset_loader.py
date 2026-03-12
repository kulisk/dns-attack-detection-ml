"""
Dataset loader supporting CIC-DNS, CIRA-CIC-DoHBrw, UNSW-NB15,
and any custom CSV-based DNS dataset.

The loader enforces strict train/validation/test splits *before*
any normalisation so that statistics are computed only on the
training fold (no data leakage).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Column maps for known public datasets
# ──────────────────────────────────────────────────────────────────────

_CIC_DNS_LABEL_MAP: dict[str, str] = {
    "BENIGN": "benign",
    "DNS DDoS": "dns_ddos",
    "Amplification": "dns_amplification",
    "DNS Tunneling": "dns_tunneling",
    "Cache Poisoning": "cache_poisoning",
    "NXDOMAIN": "nxdomain_attack",
    "Data Exfiltration": "data_exfiltration",
    "Botnet": "botnet_dns",
}

_UNSW_LABEL_MAP: dict[int, str] = {
    0: "benign",
    1: "dns_ddos",
}

SUPPORTED_DATASETS = ("cic_dns", "cira_doh", "unsw_nb15", "custom", "synthetic")


class DatasetLoader:
    """Load, merge, and split DNS security datasets.

    Args:
        dataset_dir: Directory containing raw CSV files.
        source: Dataset type from :data:`SUPPORTED_DATASETS`.
        test_size: Fraction of data reserved for test set (default 0.15).
        val_size: Fraction of data reserved for validation set (default 0.15).
        random_state: Reproducibility seed (default 42).
        label_col: Name of the target column.
        binary: If *True*, produce a binary (attack / benign) label column.
    """

    def __init__(
        self,
        dataset_dir: str = "datasets",
        source: str = "synthetic",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        label_col: str = "label",
        binary: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.source = source
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.label_col = label_col
        self.binary = binary

        logger.info(
            "DatasetLoader initialised",
            extra={"source": source, "dataset_dir": str(self.dataset_dir)},
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> pd.DataFrame:
        """Load raw data from disk and return a unified DataFrame.

        Returns:
            Raw DataFrame with a normalised ``label`` column.

        Raises:
            ValueError: If the dataset source is not supported.
            FileNotFoundError: If no CSV files are found.
        """
        if self.source == "synthetic":
            from src.data_collection.synthetic_generator import SyntheticDNSGenerator
            logger.info("Generating synthetic DNS dataset …")
            gen = SyntheticDNSGenerator(random_state=self.random_state)
            df = gen.generate(n_samples=50_000)
        elif self.source == "cic_dns":
            df = self._load_csv_dir()
            df = self._normalise_cic_labels(df)
        elif self.source == "cira_doh":
            df = self._load_csv_dir()
            df = self._normalise_cira_labels(df)
        elif self.source == "unsw_nb15":
            df = self._load_csv_dir()
            df = self._normalise_unsw_labels(df)
        elif self.source == "custom":
            df = self._load_csv_dir()
        else:
            raise ValueError(
                f"Unknown dataset source '{self.source}'. "
                f"Supported: {SUPPORTED_DATASETS}"
            )

        if self.binary:
            df["is_attack"] = (df[self.label_col] != "benign").astype(int)

        logger.info(
            "Dataset loaded",
            extra={
                "rows": len(df),
                "columns": len(df.columns),
                "label_distribution": df[self.label_col].value_counts().to_dict(),
            },
        )
        return df

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into train / validation / test sets.

        Splitting is stratified on the label column.  The validation
        fraction is computed *relative to the training remainder* so
        that test_size and val_size are both fractions of the full
        dataset (no data leakage between sets).

        Args:
            df: Full dataset returned by :meth:`load`.

        Returns:
            Tuple ``(train_df, val_df, test_df)``.
        """
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df[self.label_col],
            random_state=self.random_state,
        )
        relative_val = self.val_size / (1.0 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            stratify=train_val_df[self.label_col],
            random_state=self.random_state,
        )
        logger.info(
            "Dataset split complete",
            extra={
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
            },
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_csv_dir(self) -> pd.DataFrame:
        """Load and concatenate all CSV files found in ``dataset_dir``."""
        csv_files = list(self.dataset_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in '{self.dataset_dir}'. "
                "Place your dataset files there or use source='synthetic'."
            )
        dfs = []
        for fp in csv_files:
            logger.info("Reading CSV", extra={"file": fp.name})
            chunk = pd.read_csv(fp, low_memory=False)
            chunk.columns = chunk.columns.str.strip().str.lower().str.replace(" ", "_")
            dfs.append(chunk)
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def _normalise_cic_labels(df: pd.DataFrame) -> pd.DataFrame:
        label_candidates = ["label", "Label", "attack_type"]
        col = next((c for c in label_candidates if c in df.columns), None)
        if col is None:
            raise ValueError("No label column found in CIC-DNS dataset")
        df = df.rename(columns={col: "label"})
        df["label"] = df["label"].map(
            lambda x: _CIC_DNS_LABEL_MAP.get(str(x).strip(), "benign")
        )
        return df

    @staticmethod
    def _normalise_cira_labels(df: pd.DataFrame) -> pd.DataFrame:
        label_candidates = ["label", "Label", "type"]
        col = next((c for c in label_candidates if c in df.columns), None)
        if col is None:
            raise ValueError("No label column found in CIRA dataset")
        df = df.rename(columns={col: "label"})
        df["label"] = df["label"].str.lower().str.replace(" ", "_")
        return df

    @staticmethod
    def _normalise_unsw_labels(df: pd.DataFrame) -> pd.DataFrame:
        label_candidates = ["attack_cat", "label", "Label"]
        col = next((c for c in label_candidates if c in df.columns), None)
        if col is None:
            raise ValueError("No label column found in UNSW-NB15 dataset")
        df = df.rename(columns={col: "label"})
        df["label"] = df["label"].fillna("benign")
        df["label"] = df["label"].str.lower().str.strip().str.replace(" ", "_")
        # Map generic attack labels to known attack types
        attack_map = {
            "generic": "dns_ddos",
            "exploits": "dns_amplification",
            "fuzzers": "botnet_dns",
            "dos": "dns_ddos",
        }
        df["label"] = df["label"].map(lambda x: attack_map.get(x, x))
        return df
