"""
Synthetic DNS traffic generator.

Produces a realistic DataFrame with DNS-specific features for all
supported attack types.  Useful for unit tests, demos, and
development without real packet captures.
"""
from __future__ import annotations

import random
import string
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

ATTACK_TYPES = [
    "benign",
    "dns_ddos",
    "dns_amplification",
    "dns_tunneling",
    "cache_poisoning",
    "nxdomain_attack",
    "data_exfiltration",
    "botnet_dns",
]

# Target class distribution (approx)
_CLASS_WEIGHTS = {
    "benign": 0.50,
    "dns_ddos": 0.12,
    "dns_amplification": 0.10,
    "dns_tunneling": 0.08,
    "cache_poisoning": 0.05,
    "nxdomain_attack": 0.07,
    "data_exfiltration": 0.04,
    "botnet_dns": 0.04,
}


class SyntheticDNSGenerator:
    """Generate labelled synthetic DNS traffic for training / testing.

    Args:
        random_state: NumPy / Python random seed for reproducibility.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.rng = np.random.default_rng(random_state)
        random.seed(random_state)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(self, n_samples: int = 50_000) -> pd.DataFrame:
        """Generate *n_samples* labelled DNS records.

        Args:
            n_samples: Total number of samples to generate.

        Returns:
            DataFrame with feature columns and a ``label`` column.
        """
        logger.info("Generating synthetic DNS data", extra={"n_samples": n_samples})
        labels = self._sample_labels(n_samples)
        records = [self._generate_record(lbl) for lbl in labels]
        df = pd.DataFrame(records)
        df["label"] = labels
        logger.info("Synthetic data generated", extra={"shape": df.shape})
        return df.sample(frac=1, random_state=int(self.rng.integers(0, 10_000))).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _sample_labels(self, n: int) -> list[str]:
        weights = list(_CLASS_WEIGHTS.values())
        classes = list(_CLASS_WEIGHTS.keys())
        counts = self.rng.multinomial(n, weights)
        labels: list[str] = []
        for cls, cnt in zip(classes, counts):
            labels.extend([cls] * cnt)
        return labels

    def _generate_record(self, label: str) -> dict:
        """Dispatch to the appropriate per-class record generator."""
        _dispatch = {
            "benign": self._benign,
            "dns_ddos": self._dns_ddos,
            "dns_amplification": self._dns_amplification,
            "dns_tunneling": self._dns_tunneling,
            "cache_poisoning": self._cache_poisoning,
            "nxdomain_attack": self._nxdomain,
            "data_exfiltration": self._data_exfiltration,
            "botnet_dns": self._botnet,
        }
        return _dispatch[label]()

    # ──────────────────────── Per-class generators ────────────────────────

    def _benign(self) -> dict:
        domain = self._normal_domain()
        return {
            "query_length": len(domain),
            "entropy": self._entropy(domain),
            "num_subdomains": self.rng.integers(0, 3).item(),
            "num_labels": domain.count(".") + 1,
            "max_label_length": max(len(p) for p in domain.split(".")),
            "avg_label_length": np.mean([len(p) for p in domain.split(".")]),
            "digit_ratio": sum(c.isdigit() for c in domain) / len(domain),
            "hyphen_ratio": domain.count("-") / len(domain),
            "consonant_ratio": self.rng.uniform(0.5, 0.7).item(),
            "query_frequency": self.rng.integers(1, 20).item(),
            "nxdomain_ratio": self.rng.uniform(0.0, 0.05).item(),
            "ttl_mean": self.rng.integers(300, 86400).item(),
            "ttl_std": self.rng.uniform(0, 500).item(),
            "ttl_min": self.rng.integers(60, 300).item(),
            "ttl_max": self.rng.integers(3600, 86400).item(),
            "packet_size": self.rng.integers(40, 200).item(),
            "packet_size_std": self.rng.uniform(5, 30).item(),
            "req_resp_ratio": self.rng.uniform(0.8, 1.2).item(),
            "unique_src_ips": self.rng.integers(1, 10).item(),
            "unique_dst_ips": self.rng.integers(1, 5).item(),
            "query_rate_10s": self.rng.uniform(0.1, 2.0).item(),
            "query_rate_30s": self.rng.uniform(0.1, 3.0).item(),
            "query_rate_60s": self.rng.uniform(0.1, 5.0).item(),
            "is_any_query": 0,
            "is_tcp": 0,
            "response_code": 0,  # NOERROR
            "answer_count": self.rng.integers(1, 5).item(),
            "authority_count": 0,
            "has_valid_tld": 1,
        }

    def _dns_ddos(self) -> dict:
        rec = self._benign()
        rec.update({
            "query_frequency": self.rng.integers(500, 5000).item(),
            "query_rate_10s": self.rng.uniform(50.0, 500.0).item(),
            "query_rate_30s": self.rng.uniform(100.0, 1000.0).item(),
            "query_rate_60s": self.rng.uniform(200.0, 2000.0).item(),
            "unique_src_ips": self.rng.integers(100, 10000).item(),
            "packet_size": self.rng.integers(40, 60).item(),
            "packet_size_std": self.rng.uniform(1, 5).item(),
            "req_resp_ratio": self.rng.uniform(5.0, 50.0).item(),
        })
        return rec

    def _dns_amplification(self) -> dict:
        rec = self._benign()
        rec.update({
            "is_any_query": 1,
            "packet_size": self.rng.integers(500, 4096).item(),
            "packet_size_std": self.rng.uniform(100, 500).item(),
            "req_resp_ratio": self.rng.uniform(0.01, 0.1).item(),  # many large responses
            "answer_count": self.rng.integers(10, 50).item(),
            "query_frequency": self.rng.integers(100, 1000).item(),
            "ttl_mean": self.rng.integers(0, 60).item(),
        })
        return rec

    def _dns_tunneling(self) -> dict:
        domain = self._tunneling_domain()
        rec = self._benign()
        rec.update({
            "query_length": len(domain),
            "entropy": self._entropy(domain),
            "num_subdomains": self.rng.integers(3, 8).item(),
            "max_label_length": self.rng.integers(40, 63).item(),
            "avg_label_length": self.rng.uniform(30, 60).item(),
            "digit_ratio": self.rng.uniform(0.3, 0.6).item(),
            "consonant_ratio": self.rng.uniform(0.4, 0.6).item(),
            "is_tcp": int(self.rng.integers(0, 2)),
            "packet_size": self.rng.integers(200, 512).item(),
            "query_frequency": self.rng.integers(20, 200).item(),
        })
        return rec

    def _cache_poisoning(self) -> dict:
        rec = self._benign()
        rec.update({
            "response_code": 0,
            "answer_count": self.rng.integers(1, 3).item(),
            "authority_count": self.rng.integers(1, 5).item(),
            "unique_src_ips": self.rng.integers(10, 500).item(),
            "unique_dst_ips": self.rng.integers(1, 3).item(),
            "ttl_mean": self.rng.integers(0, 30).item(),  # very low TTL
            "query_frequency": self.rng.integers(50, 500).item(),
            "req_resp_ratio": self.rng.uniform(0.5, 2.0).item(),
        })
        return rec

    def _nxdomain(self) -> dict:
        rec = self._benign()
        rec.update({
            "nxdomain_ratio": self.rng.uniform(0.7, 1.0).item(),
            "response_code": 3,  # NXDOMAIN
            "query_frequency": self.rng.integers(100, 2000).item(),
            "has_valid_tld": int(self.rng.integers(0, 2)),
            "entropy": self.rng.uniform(3.5, 4.5).item(),
            "query_rate_10s": self.rng.uniform(5.0, 50.0).item(),
        })
        return rec

    def _data_exfiltration(self) -> dict:
        domain = self._exfiltration_domain()
        rec = self._benign()
        rec.update({
            "query_length": len(domain),
            "entropy": self._entropy(domain),
            "num_subdomains": self.rng.integers(2, 6).item(),
            "max_label_length": self.rng.integers(30, 63).item(),
            "digit_ratio": self.rng.uniform(0.2, 0.5).item(),
            "query_frequency": self.rng.integers(10, 100).item(),
            "packet_size": self.rng.integers(150, 300).item(),
            "has_valid_tld": 1,
        })
        return rec

    def _botnet(self) -> dict:
        rec = self._benign()
        rec.update({
            "query_frequency": self.rng.integers(5, 30).item(),
            "unique_src_ips": self.rng.integers(50, 1000).item(),
            "ttl_mean": self.rng.integers(30, 120).item(),
            "ttl_std": self.rng.uniform(0, 5).item(),  # very consistent
            "nxdomain_ratio": self.rng.uniform(0.1, 0.4).item(),
            "req_resp_ratio": self.rng.uniform(1.0, 3.0).item(),
            "query_rate_60s": self.rng.uniform(1.0, 10.0).item(),
        })
        return rec

    # ──────────────────────────── Domain helpers ──────────────────────────

    def _normal_domain(self) -> str:
        tlds = ["com", "net", "org", "io", "co.uk", "de", "fr"]
        words = ["google", "youtube", "facebook", "amazon", "microsoft",
                 "twitter", "reddit", "netflix", "github", "stackoverflow"]
        sld = random.choice(words) + str(self.rng.integers(0, 100).item())
        tld = random.choice(tlds)
        if self.rng.random() < 0.3:
            sub = "".join(random.choices(string.ascii_lowercase, k=self.rng.integers(3, 8).item()))
            return f"{sub}.{sld}.{tld}"
        return f"{sld}.{tld}"

    def _tunneling_domain(self) -> str:
        payload = "".join(
            random.choices(string.ascii_lowercase + string.digits,
                           k=self.rng.integers(30, 55).item())
        )
        return f"{payload}.tunnel.example.com"

    def _exfiltration_domain(self) -> str:
        b64chars = string.ascii_letters + string.digits + "+/"
        chunk = "".join(random.choices(b64chars, k=self.rng.integers(20, 50).item()))
        return f"{chunk}.data.exfil.net"

    @staticmethod
    def _entropy(text: str) -> float:
        from collections import Counter
        import math
        if not text:
            return 0.0
        counts = Counter(text.lower())
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())
