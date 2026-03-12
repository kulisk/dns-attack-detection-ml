"""
General-purpose helper functions for feature computation,
domain analysis, and data validation.
"""
import math
import re
import socket
from collections import Counter
from typing import Optional

import numpy as np


# ─────────────────────────── Entropy ───────────────────────────


def compute_entropy(text: str) -> float:
    """Compute Shannon entropy of a string (useful for DGA / tunneling detection).

    Args:
        text: Input string (domain name, label, etc.).

    Returns:
        Entropy value in bits.  Returns 0.0 for empty string.
    """
    if not text:
        return 0.0
    counts = Counter(text.lower())
    length = len(text)
    return -sum(
        (c / length) * math.log2(c / length) for c in counts.values()
    )


# ─────────────────────────── Domain Features ───────────────────────────


def extract_domain_features(domain: str) -> dict:
    """Extract numeric features from a DNS domain name.

    Args:
        domain: Fully-qualified domain name (e.g. ``sub.example.com``).

    Returns:
        Dictionary with computed features:
        - ``query_length``: Total character count.
        - ``num_labels``: Number of labels (dot-separated parts).
        - ``num_subdomains``: Labels minus TLD and SLD.
        - ``max_label_length``: Longest individual label.
        - ``avg_label_length``: Average label length.
        - ``digit_ratio``: Fraction of characters that are digits.
        - ``hyphen_ratio``: Fraction of characters that are hyphens.
        - ``consonant_ratio``: Fraction of letters that are consonants.
        - ``entropy``: Shannon entropy of the full domain.
        - ``tld``: Top-level domain string.
        - ``is_ip``: 1 if domain looks like an IP address, else 0.
    """
    domain = domain.rstrip(".").lower()
    labels = domain.split(".")

    is_ip = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)))
    entropy = compute_entropy(domain.replace(".", ""))
    lengths = [len(lbl) for lbl in labels]
    digits = sum(c.isdigit() for c in domain)
    hyphens = domain.count("-")
    letters = [c for c in domain if c.isalpha()]
    consonants = sum(
        c not in "aeiou" for c in letters
    )

    return {
        "query_length": len(domain),
        "num_labels": len(labels),
        "num_subdomains": max(len(labels) - 2, 0),
        "max_label_length": max(lengths, default=0),
        "avg_label_length": float(np.mean(lengths)) if lengths else 0.0,
        "digit_ratio": digits / len(domain) if domain else 0.0,
        "hyphen_ratio": hyphens / len(domain) if domain else 0.0,
        "consonant_ratio": consonants / len(letters) if letters else 0.0,
        "entropy": entropy,
        "tld": labels[-1] if labels else "",
        "is_ip": is_ip,
    }


# ─────────────────────────── Validation ────────────────────────


def validate_ip(ip: str) -> bool:
    """Return True if *ip* is a valid IPv4 or IPv6 address."""
    try:
        socket.inet_pton(socket.AF_INET, ip)
        return True
    except OSError:
        pass
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return True
    except OSError:
        return False


def is_valid_domain(domain: str) -> bool:
    """Return True if *domain* looks like a valid FQDN."""
    pattern = re.compile(
        r"^(?!-)[A-Za-z0-9\-]{1,63}(?<!-)"
        r"(?:\.(?!-)[A-Za-z0-9\-]{1,63}(?<!-))*"
        r"\.[A-Za-z]{2,}$"
    )
    return bool(pattern.match(domain.rstrip(".")))


# ─────────────────────────── Statistical Helpers ───────────────────────────


def safe_log2(x: float) -> float:
    """Return log2(x) or 0 when x <= 0."""
    return math.log2(x) if x > 0 else 0.0


def sliding_window_stats(values: list[float], window: int) -> dict:
    """Compute rolling mean/std/min/max for a list of values.

    Args:
        values: Numeric sequence.
        window: Window size in samples.

    Returns:
        Dict with ``mean``, ``std``, ``min``, ``max`` of the last *window* items.
    """
    arr = np.array(values[-window:], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
