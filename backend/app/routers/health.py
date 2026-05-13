"""Endpoint /health + /health/cache (observabilidad del cache de precios)."""
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.decorators import log_latency
from app.models.db_models import Price
from app.models.schemas import HealthOut
from app.services.data import CACHE_STATS

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Estado del servicio")
@log_latency("GET /health")
async def health(s: Settings = Depends(get_settings)) -> HealthOut:
    return HealthOut(status="ok", env=s.env, app_name=s.app_name)


@router.get(
    "/health/cache",
    summary="Estado y stats del cache de precios (HIT/MISS/STALE por ticker)",
)
async def health_cache(db: Session = Depends(get_db)) -> dict:
    total_rows = db.scalar(select(func.count()).select_from(Price)) or 0
    return {
        "tickers": {t: dict(s) for t, s in CACHE_STATS.items()},
        "total_rows": int(total_rows),
    }
