"""
Async inference engine for real-time DNS attack detection.

Consumes packets from the capture queue, extracts features, feeds
them to a trained model, and emits alerts via the AlertManager.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.feature_engineering import DNSFeatureExtractor, WindowAggregator
from src.models.base_detector import BaseDetector
from src.utils import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class InferenceEngine:
    """Pull packets from a queue and classify them in real time.

    Args:
        model: Fitted :class:`~src.models.base_detector.BaseDetector` instance.
        packet_queue: asyncio queue populated by :class:`PacketCapture`.
        alert_callback: Async callable invoked when an attack is detected::

            async def my_callback(alert: dict) -> None: ...

        alert_threshold: Minimum attack probability to trigger an alert.
        model_dir: Directory with persisted preprocessor files.
        config_path: YAML config path.
    """

    def __init__(
        self,
        model: BaseDetector,
        packet_queue: asyncio.Queue,
        alert_callback: Optional[object] = None,
        alert_threshold: float = 0.75,
        model_dir: str = "models",
        config_path: str = "configs/config.yaml",
    ) -> None:
        self.model = model
        self.packet_queue = packet_queue
        self.alert_callback = alert_callback
        self.alert_threshold = alert_threshold
        self.model_dir = Path(model_dir)
        self.cfg = ConfigLoader(config_path)

        self.extractor = DNSFeatureExtractor()
        self.aggregator = WindowAggregator(
            windows=self.cfg.get("feature_engineering.aggregation_windows", [10, 30, 60, 300])
        )

        # Load preprocessing objects if available
        self._scaler = self._load_preprocessor("scaler.joblib")
        self._encoder = self._load_preprocessor("label_encoder.joblib")

        self._attack_types: list[str] = self.cfg.get("attack_types", ["benign", "attack"])
        self._processed = 0
        self._alerts = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Consume the packet queue indefinitely until cancelled."""
        logger.info(
            "InferenceEngine started",
            extra={"model": self.model.name, "threshold": self.alert_threshold},
        )
        while True:
            try:
                pkt_dict = await asyncio.wait_for(
                    self.packet_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                await self._process_packet(pkt_dict)
            except Exception as exc:
                logger.error("Inference failed", extra={"error": str(exc)})
            finally:
                self.packet_queue.task_done()

    async def run_for(self, seconds: float) -> None:
        """Run inference for a fixed duration (useful for testing)."""
        try:
            await asyncio.wait_for(self.run(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
        logger.info(
            "Inference finished",
            extra={"processed": self._processed, "alerts": self._alerts},
        )

    @property
    def stats(self) -> dict:
        return {"processed": self._processed, "alerts": self._alerts}

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _process_packet(self, pkt_dict: dict) -> None:
        src_ip = pkt_dict.get("src_ip", "0.0.0.0")
        rcode = int(pkt_dict.get("rcode", 0))
        packet_size = int(pkt_dict.get("packet_size", 60))
        ts = float(pkt_dict.get("timestamp", time.time()))

        # Update rolling-window state
        self.aggregator.update(
            src_ip=src_ip,
            is_nxdomain=(rcode == 3),
            packet_size=packet_size,
            rcode=rcode,
            ts=ts,
        )

        # Merge window stats into packet dict
        window_feats = self.aggregator.get_features(src_ip, ts=ts)
        pkt_dict.update(window_feats)

        # Feature extraction
        features = self.extractor.extract_from_packet(pkt_dict)
        X = np.array([[v for v in features.values()]], dtype=np.float32)

        # Apply scaler if available
        if self._scaler is not None:
            try:
                import pandas as pd
                feat_df = pd.DataFrame([features])
                feat_df = self._scaler.transform(feat_df)
                X = feat_df.values.astype(np.float32)
            except Exception:
                pass

        # Inference
        proba = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        attack_prob = float(1.0 - proba[0]) if len(proba) > 1 else float(proba[0])

        self._processed += 1

        if attack_prob >= self.alert_threshold:
            attack_label = (
                self._attack_types[pred_idx]
                if pred_idx < len(self._attack_types)
                else "unknown"
            )
            alert = {
                "timestamp": ts,
                "src_ip": src_ip,
                "domain": pkt_dict.get("qname", ""),
                "attack_type": attack_label,
                "confidence": round(attack_prob, 4),
                "model": self.model.name,
            }
            self._alerts += 1
            logger.warning("ATTACK DETECTED", extra=alert)

            if self.alert_callback is not None:
                await self.alert_callback(alert)  # type: ignore[operator]

    def _load_preprocessor(self, filename: str):
        path = self.model_dir / filename
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as exc:
                logger.warning(f"Could not load {filename}", extra={"error": str(exc)})
        return None
