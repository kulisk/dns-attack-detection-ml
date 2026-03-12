"""
DNS-specific feature extractor.

Transforms raw DNS records (either from a DataFrame of parsed packet
fields or from scapy DNS layers) into the numeric feature vector used
by the ML models.  All transformations are stateless and determinis-
tic to avoid leakage.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from src.utils import get_logger
from src.utils.helpers import compute_entropy, extract_domain_features

logger = get_logger(__name__)

# Columns expected when transforming a pre-built DNS DataFrame
EXPECTED_COLUMNS = [
    "query_length", "entropy", "num_subdomains", "num_labels",
    "max_label_length", "avg_label_length", "digit_ratio", "hyphen_ratio",
    "consonant_ratio", "query_frequency", "nxdomain_ratio", "ttl_mean",
    "ttl_std", "ttl_min", "ttl_max", "packet_size", "packet_size_std",
    "req_resp_ratio", "unique_src_ips", "unique_dst_ips", "query_rate_10s",
    "query_rate_30s", "query_rate_60s", "is_any_query", "is_tcp",
    "response_code", "answer_count", "authority_count", "has_valid_tld",
]


class DNSFeatureExtractor:
    """Extract and augment DNS features from a raw DataFrame.

    The extractor accepts a DataFrame that already contains the expected
    columns (as produced by :class:`~src.data_collection.SyntheticDNSGenerator`
    or after parsing real packet captures) and adds derived features.

    Args:
        add_interaction_features: Compute cross-feature interactions
            (e.g. ``entropy × query_length``).
        add_statistical_fingerprint: Add higher-order stats derived from
            window aggregations if available.
    """

    def __init__(
        self,
        add_interaction_features: bool = True,
        add_statistical_fingerprint: bool = True,
    ) -> None:
        self.add_interaction_features = add_interaction_features
        self.add_statistical_fingerprint = add_statistical_fingerprint

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich the feature DataFrame with derived DNS features.

        Args:
            df: DataFrame with at least a subset of :data:`EXPECTED_COLUMNS`.

        Returns:
            Enriched copy with additional computed columns.
        """
        df = df.copy()

        # --- Fill any missing expected columns with safe defaults ---
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        # --- Derived features ---
        df = self._add_ratio_features(df)
        df = self._add_domain_profile_features(df)

        if self.add_interaction_features:
            df = self._add_interaction_features(df)

        if self.add_statistical_fingerprint:
            df = self._add_fingerprint_features(df)

        logger.debug(
            "Features extracted",
            extra={"original_cols": len(EXPECTED_COLUMNS), "total_cols": len(df.columns)},
        )
        return df

    def extract_from_packet(self, packet_dict: dict) -> dict:
        """Extract features from a single parsed packet dictionary.

        Accepts the raw parsed fields produced by the real-time
        capture pipeline and returns a flat feature dict ready for
        model inference.

        Args:
            packet_dict: Dict with keys such as ``qname``, ``src_ip``,
                ``dst_ip``, ``packet_size``, ``rcode``, ``ttl``, etc.

        Returns:
            Feature dictionary aligned with :data:`EXPECTED_COLUMNS`.
        """
        qname = str(packet_dict.get("qname", "")).rstrip(".")
        domain_feats = extract_domain_features(qname)

        features = {
            "query_length": domain_feats["query_length"],
            "entropy": domain_feats["entropy"],
            "num_subdomains": domain_feats["num_subdomains"],
            "num_labels": domain_feats["num_labels"],
            "max_label_length": domain_feats["max_label_length"],
            "avg_label_length": domain_feats["avg_label_length"],
            "digit_ratio": domain_feats["digit_ratio"],
            "hyphen_ratio": domain_feats["hyphen_ratio"],
            "consonant_ratio": domain_feats["consonant_ratio"],
            "query_frequency": float(packet_dict.get("query_frequency", 1)),
            "nxdomain_ratio": float(packet_dict.get("nxdomain_ratio", 0.0)),
            "ttl_mean": float(packet_dict.get("ttl", 300)),
            "ttl_std": 0.0,
            "ttl_min": float(packet_dict.get("ttl", 300)),
            "ttl_max": float(packet_dict.get("ttl", 300)),
            "packet_size": float(packet_dict.get("packet_size", 60)),
            "packet_size_std": 0.0,
            "req_resp_ratio": float(packet_dict.get("req_resp_ratio", 1.0)),
            "unique_src_ips": float(packet_dict.get("unique_src_ips", 1)),
            "unique_dst_ips": float(packet_dict.get("unique_dst_ips", 1)),
            "query_rate_10s": float(packet_dict.get("query_rate_10s", 0.0)),
            "query_rate_30s": float(packet_dict.get("query_rate_30s", 0.0)),
            "query_rate_60s": float(packet_dict.get("query_rate_60s", 0.0)),
            "is_any_query": int(packet_dict.get("qtype", 1) == 255),
            "is_tcp": int(packet_dict.get("proto", "udp") == "tcp"),
            "response_code": int(packet_dict.get("rcode", 0)),
            "answer_count": int(packet_dict.get("answer_count", 0)),
            "authority_count": int(packet_dict.get("authority_count", 0)),
            "has_valid_tld": domain_feats.get("is_ip", 0) ^ 1,
        }
        # Add derived features from a single-row DataFrame
        row_df = pd.DataFrame([features])
        row_df = self.transform(row_df)
        return row_df.iloc[0].to_dict()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add normalised and ratio features."""
        eps = 1e-9
        df["entropy_per_label"] = df["entropy"] / (df["num_labels"] + eps)
        df["length_entropy_product"] = df["query_length"] * df["entropy"]
        df["subdomain_density"] = df["num_subdomains"] / (df["num_labels"] + eps)
        df["ttl_range"] = df["ttl_max"] - df["ttl_min"]
        df["ttl_cv"] = df["ttl_std"] / (df["ttl_mean"] + eps)  # coefficient of variation
        df["nxdomain_x_freqency"] = df["nxdomain_ratio"] * df["query_frequency"]
        df["amplification_ratio"] = df["packet_size"] * df["is_any_query"]
        return df

    @staticmethod
    def _add_domain_profile_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-structure profile indicators."""
        df["is_high_entropy"] = (df["entropy"] > 3.5).astype(int)
        df["is_long_query"] = (df["query_length"] > 50).astype(int)
        df["is_deep_subdomain"] = (df["num_subdomains"] >= 4).astype(int)
        df["is_high_nxdomain"] = (df["nxdomain_ratio"] > 0.5).astype(int)
        df["is_low_ttl"] = (df["ttl_mean"] < 60).astype(int)
        df["is_high_query_rate"] = (df["query_rate_60s"] > 20.0).astype(int)
        df["is_large_packet"] = (df["packet_size"] > 512).astype(int)
        return df

    @staticmethod
    def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Pairwise interaction terms for key features."""
        df["entropy_x_nxdomain"] = df["entropy"] * df["nxdomain_ratio"]
        df["entropy_x_subdomain"] = df["entropy"] * df["num_subdomains"]
        df["freq_x_unique_src"] = df["query_frequency"] * df["unique_src_ips"]
        df["rate60_x_packet_size"] = df["query_rate_60s"] * df["packet_size"]
        df["ttl_cv_x_entropy"] = df["ttl_cv"] * df["entropy"]
        return df

    @staticmethod
    def _add_fingerprint_features(df: pd.DataFrame) -> pd.DataFrame:
        """Higher-order statistical profile."""
        num_cols = ["query_rate_10s", "query_rate_30s", "query_rate_60s"]
        for col in num_cols:
            if col in df.columns:
                df[f"log1p_{col}"] = np.log1p(df[col])
        df["packet_size_log"] = np.log1p(df["packet_size"])
        df["query_freq_log"] = np.log1p(df["query_frequency"])
        return df
