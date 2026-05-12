"""Endpoint /alertas: senales activas."""
from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import SignalsOut
from app.services.data import get_prices, list_assets
from app.services.signals import detect_signals

router = APIRouter(tags=["analisis"])


@router.get("/alertas", response_model=SignalsOut)
async def alertas(db: Session = Depends(get_db)) -> SignalsOut:
    all_signals = []
    for asset in list_assets(db):
        df = get_prices(db, asset.ticker)
        if df.empty:
            continue
        all_signals.extend(detect_signals(asset.ticker, df["close"]))
    return SignalsOut(as_of=date.today(), signals=all_signals)
