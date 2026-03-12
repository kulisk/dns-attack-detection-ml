"""
Alert manager for real-time DNS attack detection.

Receives alert dictionaries from the inference engine and:
  - Logs structured alerts via the central logger.
  - Optionally forwards alerts to a webhook URL.
  - Maintains an in-memory alert ring-buffer for the FastAPI endpoint.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


class AlertManager:
    """Manages and routes DNS attack alerts.

    Args:
        webhook_url: Optional HTTP endpoint to POST JSON alerts.
        max_history: Maximum number of alerts kept in memory.
        deduplicate_window: Seconds within which duplicate alerts
            (same src_ip + attack_type) are suppressed.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        max_history: int = 1000,
        deduplicate_window: float = 30.0,
    ) -> None:
        self.webhook_url = webhook_url
        self.max_history = max_history
        self.deduplicate_window = deduplicate_window
        self._history: deque = deque(maxlen=max_history)
        self._seen: dict[str, float] = {}  # key → last_seen_ts

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def handle(self, alert: dict) -> None:
        """Process an alert: deduplicate, log, store, and forward.

        Args:
            alert: Alert dict with keys ``src_ip``, ``attack_type``,
                ``confidence``, ``domain``, ``timestamp``, etc.
        """
        # Deduplication
        key = f"{alert.get('src_ip')}::{alert.get('attack_type')}"
        now = alert.get("timestamp", 0.0)
        last = self._seen.get(key, 0.0)
        if now - last < self.deduplicate_window:
            return  # Suppress duplicate within window
        self._seen[key] = float(now)

        # Enrich with human-readable timestamp
        alert["datetime"] = datetime.fromtimestamp(
            float(now), tz=timezone.utc
        ).isoformat()

        # Store and log
        self._history.append(alert)
        logger.warning(
            "DNS ATTACK ALERT",
            extra={
                "src_ip": alert.get("src_ip"),
                "domain": alert.get("domain"),
                "attack_type": alert.get("attack_type"),
                "confidence": alert.get("confidence"),
                "model": alert.get("model"),
            },
        )

        # Forward to webhook
        if self.webhook_url:
            asyncio.create_task(self._post_webhook(alert))

    def get_recent_alerts(self, n: int = 50) -> list[dict]:
        """Return the *n* most recent alerts (newest first)."""
        return list(reversed(list(self._history)))[:n]

    def clear(self) -> None:
        """Clear the alert history and deduplication state."""
        self._history.clear()
        self._seen.clear()

    @property
    def total_alerts(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _post_webhook(self, alert: dict) -> None:
        if not _HTTPX_AVAILABLE:
            logger.debug("httpx not installed – webhook skipped")
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    self.webhook_url,
                    json=alert,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code not in (200, 201, 202):
                    logger.warning(
                        "Webhook delivery failed",
                        extra={"status": resp.status_code, "url": self.webhook_url},
                    )
        except Exception as exc:
            logger.error("Webhook error", extra={"error": str(exc)})
