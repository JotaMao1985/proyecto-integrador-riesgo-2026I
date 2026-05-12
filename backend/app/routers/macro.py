"""Endpoint /macro: tasa libre y otros indicadores."""
from fastapi import APIRouter

from app.models.schemas import MacroOut
from app.services.macro import get_rf_annual, now_iso

router = APIRouter(tags=["datos"])


@router.get("/macro", response_model=MacroOut)
async def macro() -> MacroOut:
    rf, source = get_rf_annual()
    return MacroOut(rf=rf, rf_source=source, fetched_at=now_iso())
