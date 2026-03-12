"""
Optional FastAPI microservice for DNS attack alerting.

Exposes REST endpoints to:
  - GET /health        – health check
  - GET /alerts        – recent alert list
  - GET /stats         – detection statistics
  - POST /alerts/clear – clear alert history

Start with: uvicorn src.realtime_detection.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.realtime_detection.alert_manager import AlertManager

app = FastAPI(
    title="DNS Attack Detection API",
    description="Real-time DNS attack detection alert service",
    version="1.0.0",
)

# Singleton alert manager shared with the inference engine
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# ─────────────────────────── Schemas ──────────────────────────


class AlertOut(BaseModel):
    timestamp: float
    datetime: Optional[str] = None
    src_ip: str
    domain: str
    attack_type: str
    confidence: float
    model: str


class StatsOut(BaseModel):
    total_alerts: int


# ─────────────────────────── Endpoints ────────────────────────


@app.get("/health", tags=["system"])
async def health() -> dict:
    """Service health check."""
    return {"status": "ok"}


@app.get("/alerts", response_model=list[AlertOut], tags=["detection"])
async def get_alerts(n: int = 50) -> list[dict]:
    """Return the *n* most recent detection alerts."""
    mgr = get_alert_manager()
    return mgr.get_recent_alerts(n=n)


@app.get("/stats", response_model=StatsOut, tags=["detection"])
async def get_stats() -> dict:
    """Return detection statistics."""
    mgr = get_alert_manager()
    return {"total_alerts": mgr.total_alerts}


@app.post("/alerts/clear", tags=["system"])
async def clear_alerts() -> dict:
    """Clear the in-memory alert history."""
    get_alert_manager().clear()
    return {"message": "Alert history cleared."}


@app.post("/alerts/ingest", tags=["detection"])
async def ingest_alert(alert: dict) -> dict:
    """Accept an alert from external sources (e.g. the inference engine)."""
    mgr = get_alert_manager()
    await mgr.handle(alert)
    return {"status": "received"}
