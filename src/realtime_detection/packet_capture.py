"""
Real-time DNS packet capture using Scapy.

Captures UDP/TCP packets on port 53, parses DNS layers, and
pushes extracted packet dictionaries to an asyncio queue for
downstream processing by the inference engine.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable, Optional

from src.utils import get_logger

logger = get_logger(__name__)

try:
    from scapy.all import DNS, DNSQR, DNSRR, IP, TCP, UDP, sniff
    _SCAPY_AVAILABLE = True
except ImportError:
    _SCAPY_AVAILABLE = False
    logger.warning(
        "scapy not available – PacketCapture will use mock mode. "
        "Install with: pip install scapy"
    )


class PacketCapture:
    """Asynchronous DNS packet sniffer.

    Runs Scapy sniff in a background thread and pushes parsed
    packet dictionaries into *packet_queue*.

    Args:
        interface: Network interface name (e.g. ``"eth0"``, ``"\\Device\\NPF_...``).
        packet_queue: asyncio queue for passing packets to the inference engine.
        bpf_filter: Berkeley Packet Filter expression.
        max_queue_size: Drop packets if queue exceeds this size.
    """

    def __init__(
        self,
        interface: str = "eth0",
        packet_queue: Optional[asyncio.Queue] = None,
        bpf_filter: str = "udp port 53 or tcp port 53",
        max_queue_size: int = 1000,
    ) -> None:
        self.interface = interface
        self.packet_queue = packet_queue or asyncio.Queue(maxsize=max_queue_size)
        self.bpf_filter = bpf_filter
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stats = {"captured": 0, "parsed": 0, "dropped": 0}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start packet capture in a background thread.

        Args:
            loop: Event loop for thread-safe queue puts.
        """
        self._loop = loop or asyncio.get_event_loop()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="PacketCaptureThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "PacketCapture started",
            extra={"interface": self.interface, "filter": self.bpf_filter},
        )

    def stop(self) -> None:
        """Signal the capture loop to stop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("PacketCapture stopped", extra={"stats": self._stats})

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _capture_loop(self) -> None:
        if not _SCAPY_AVAILABLE:
            self._mock_capture_loop()
            return

        try:
            sniff(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._process_packet,
                store=False,
                stop_filter=lambda _: not self._running,
            )
        except Exception as exc:
            logger.error("Packet capture error", extra={"error": str(exc)})

    def _mock_capture_loop(self) -> None:
        """Generate synthetic DNS packets for testing without root/scapy."""
        import random
        import math
        domains = [
            "google.com", "facebook.com", "evil.tunnel.example.com",
            "A" * 40 + ".exfil.net", "update.microsoft.com",
        ]
        logger.info("Running in MOCK capture mode (no scapy / no privileged interface)")
        while self._running:
            domain = random.choice(domains)
            pkt_dict = {
                "src_ip": f"10.0.0.{random.randint(1, 254)}",
                "dst_ip": "8.8.8.8",
                "qname": domain,
                "qtype": random.choice([1, 28, 255]),
                "rcode": random.choice([0, 0, 0, 3]),
                "ttl": random.randint(30, 86400),
                "packet_size": random.randint(40, 512),
                "proto": "udp",
                "timestamp": time.time(),
                "answer_count": random.randint(0, 3),
                "authority_count": 0,
            }
            self._push(pkt_dict)
            time.sleep(0.1)

    def _process_packet(self, pkt) -> None:
        """Parse a scapy packet into a serialisable dict."""
        self._stats["captured"] += 1
        try:
            if not pkt.haslayer(DNS):
                return

            dns = pkt[DNS]
            ip = pkt[IP] if pkt.haslayer(IP) else None
            proto = "tcp" if pkt.haslayer(TCP) else "udp"

            # Query name
            qname = ""
            qtype = 1
            if dns.qdcount > 0 and dns.qd:
                qname = dns.qd.qname.decode("utf-8", errors="replace").rstrip(".")
                qtype = int(dns.qd.qtype)

            # TTL from answer records
            ttls = []
            an = dns.an
            answer_count = 0
            while an:
                answer_count += 1
                if hasattr(an, "ttl"):
                    ttls.append(an.ttl)
                an = an.payload if hasattr(an, "payload") else None

            pkt_dict = {
                "src_ip": ip.src if ip else "0.0.0.0",
                "dst_ip": ip.dst if ip else "0.0.0.0",
                "qname": qname,
                "qtype": qtype,
                "rcode": int(dns.rcode),
                "ttl": int(sum(ttls) / len(ttls)) if ttls else 300,
                "packet_size": len(pkt),
                "proto": proto,
                "timestamp": time.time(),
                "answer_count": answer_count,
                "authority_count": int(dns.arcount) if hasattr(dns, "arcount") else 0,
            }
            self._push(pkt_dict)
            self._stats["parsed"] += 1
        except Exception as exc:
            logger.debug("Failed to parse packet", extra={"error": str(exc)})

    def _push(self, pkt_dict: dict) -> None:
        """Thread-safe push to asyncio queue."""
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._async_put(pkt_dict), self._loop
            )
        except RuntimeError:
            self._stats["dropped"] += 1

    async def _async_put(self, pkt_dict: dict) -> None:
        try:
            self.packet_queue.put_nowait(pkt_dict)
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
