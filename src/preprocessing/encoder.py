"""
Label encoder for multi-class DNS attack classification.
Wraps scikit-learn's LabelEncoder and adds convenience methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

from src.utils import get_logger

logger = get_logger(__name__)

ATTACK_CLASSES = [
    "benign",
    "dns_ddos",
    "dns_amplification",
    "dns_tunneling",
    "cache_poisoning",
    "nxdomain_attack",
    "data_exfiltration",
    "botnet_dns",
]


class LabelEncoder:
    """Encode string attack labels to integers and back.

    Fitting is optional when ``known_classes`` is provided (avoids leakage
    if the test set is missing some rare classes).

    Args:
        known_classes: Fixed ordered list of class labels.  If *None*,
            classes are inferred from the training data at fit time.
    """

    def __init__(self, known_classes: list[str] | None = None) -> None:
        self._encoder = SKLabelEncoder()
        if known_classes:
            self._encoder.classes_ = np.array(known_classes)
        self.known_classes = known_classes

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, labels: pd.Series | list[str]) -> "LabelEncoder":
        """Fit encoder on the training labels.

        Args:
            labels: Iterable of string class labels.

        Returns:
            Fitted instance.
        """
        if self.known_classes is None:
            self._encoder.fit(labels)
        logger.info(
            "LabelEncoder fitted",
            extra={"classes": list(self._encoder.classes_)},
        )
        return self

    def transform(self, labels: pd.Series | list[str]) -> np.ndarray:
        """Encode string labels to integer indices.

        Args:
            labels: Labels to encode.

        Returns:
            Integer-encoded array.
        """
        return self._encoder.transform(labels)

    def inverse_transform(self, indices: np.ndarray | list[int]) -> np.ndarray:
        """Decode integer indices back to string labels."""
        return self._encoder.inverse_transform(indices)

    def fit_transform(self, labels: pd.Series | list[str]) -> np.ndarray:
        """Fit then encode labels."""
        return self.fit(labels).transform(labels)

    @property
    def classes(self) -> list[str]:
        """Ordered list of class labels."""
        return list(self._encoder.classes_)

    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(self._encoder.classes_)

    def label_to_index(self, label: str) -> int:
        """Return the integer index for a single label string."""
        return int(self.transform([label])[0])

    def index_to_label(self, index: int) -> str:
        """Return the label string for a single integer index."""
        return str(self.inverse_transform([index])[0])
