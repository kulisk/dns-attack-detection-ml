"""
Time-window aggregation for real-time DNS feature computation.

Maintains per-source rolling windows and computes aggregate statistics
(query rate, NXDOMAIN ratio, etc.) over configurable time horizons.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class _WindowBuffer:
    """Ring buffer of timestamped DNS events for one source IP."""

    timestamps: deque = field(default_factory=lambda: deque(maxlen=10_000))
    nxdomain_flags: deque = field(default_factory=lambda: deque(maxlen=10_000))
    packet_sizes: deque = field(default_factory=lambda: deque(maxlen=10_000))
    response_codes: deque = field(default_factory=lambda: deque(maxlen=10_000))


class WindowAggregator:
    """Compute windowed DNS statistics per source IP for real-time detection.

    Args:
        windows: List of window sizes in seconds (default: [10, 30, 60, 300]).
        max_sources: Maximum number of tracked source IPs before LRU eviction.
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        max_sources: int = 50_000,
    ) -> None:
        self.windows = windows or [10, 30, 60, 300]
        self.max_sources = max_sources
        self._buffers: dict[str, _WindowBuffer] = defaultdict(_WindowBuffer)
        self._access_order: deque = deque(maxlen=max_sources)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        src_ip: str,
        is_nxdomain: bool,
        packet_size: int,
        rcode: int,
        ts: Optional[float] = None,
    ) -> None:
        """Record a new DNS event for a source IP.

        Args:
            src_ip: Source IP address string.
            is_nxdomain: Whether the response was NXDOMAIN.
            packet_size: Packet payload size in bytes.
            rcode: DNS response code.
            ts: Unix timestamp; defaults to ``time.time()``.
        """
        if ts is None:
            ts = time.time()

        # LRU eviction
        if src_ip not in self._buffers and len(self._buffers) >= self.max_sources:
            oldest = self._access_order.popleft()
            self._buffers.pop(oldest, None)

        buf = self._buffers[src_ip]
        buf.timestamps.append(ts)
        buf.nxdomain_flags.append(int(is_nxdomain))
        buf.packet_sizes.append(packet_size)
        buf.response_codes.append(rcode)
        self._access_order.append(src_ip)

    def get_features(self, src_ip: str, ts: Optional[float] = None) -> dict:
        """Compute window-aggregated features for *src_ip*.

        Args:
            src_ip: Source IP to query.
            ts: Reference timestamp (defaults to now).

        Returns:
            Dict with keys like ``query_rate_10s``, ``nxdomain_ratio_60s``,
            ``avg_packet_size_30s``, etc.
        """
        if ts is None:
            ts = time.time()

        if src_ip not in self._buffers:
            return self._empty_features()

        buf = self._buffers[src_ip]
        features: dict = {}

        for w in self.windows:
            cutoff = ts - w
            indices = [
                i for i, t in enumerate(buf.timestamps) if t >= cutoff
            ]
            n = len(indices)
            features[f"query_rate_{w}s"] = n / w if w > 0 else 0.0

            if n > 0:
                nx_count = sum(buf.nxdomain_flags[i] for i in indices)
                features[f"nxdomain_ratio_{w}s"] = nx_count / n
                sizes = [buf.packet_sizes[i] for i in indices]
                features[f"avg_packet_size_{w}s"] = sum(sizes) / n
            else:
                features[f"nxdomain_ratio_{w}s"] = 0.0
                features[f"avg_packet_size_{w}s"] = 0.0

        return features

    def reset(self, src_ip: Optional[str] = None) -> None:
        """Clear buffers for one or all source IPs."""
        if src_ip:
            self._buffers.pop(src_ip, None)
        else:
            self._buffers.clear()
            self._access_order.clear()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _empty_features(self) -> dict:
        features: dict = {}
        for w in self.windows:
            features[f"query_rate_{w}s"] = 0.0
            features[f"nxdomain_ratio_{w}s"] = 0.0
            features[f"avg_packet_size_{w}s"] = 0.0
        return features
