"""Real-time DNS detection package.

Keep imports lazy so lightweight components (e.g., AlertManager) do not
implicitly load Scapy/WinPcap when importing this package.
"""

__all__ = ["PacketCapture", "InferenceEngine", "AlertManager"]


def __getattr__(name: str):
    if name == "PacketCapture":
        from .packet_capture import PacketCapture

        return PacketCapture
    if name == "InferenceEngine":
        from .inference_engine import InferenceEngine

        return InferenceEngine
    if name == "AlertManager":
        from .alert_manager import AlertManager

        return AlertManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
