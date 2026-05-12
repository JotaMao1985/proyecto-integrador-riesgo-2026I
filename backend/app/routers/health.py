"""Endpoint /health."""
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.decorators import log_latency
from app.models.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Estado del servicio")
@log_latency("GET /health")
async def health(s: Settings = Depends(get_settings)) -> HealthOut:
    return HealthOut(status="ok", env=s.env, app_name=s.app_name)
