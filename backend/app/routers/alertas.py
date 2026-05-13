"""Endpoint /alertas: senales activas con umbrales configurables.

Spec CIII (criterio 1): umbrales configurables (rsi_overbought, rsi_oversold,
bb_k) validados con Pydantic; persistencia en `signals_log`.
"""
from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import SignalLog
from app.models.schemas import SignalsOut
from app.services.data import get_prices, list_assets
from app.services.signals import SignalThresholds, detect_signals

router = APIRouter(tags=["analisis"])


@router.get("/alertas", response_model=SignalsOut)
async def alertas(
    db: Session = Depends(get_db),
    rsi_overbought: float = Query(default=70.0, ge=50.0, le=100.0),
    rsi_oversold: float = Query(default=30.0, ge=0.0, le=50.0),
    bb_k: float = Query(default=2.0, gt=0.0, le=5.0),
) -> SignalsOut:
    th = SignalThresholds(
        rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold, bb_k=bb_k
    )
    all_signals = []
    for asset in list_assets(db):
        df = get_prices(db, asset.ticker, validate_ticker=False)
        if df.empty:
            continue
        sigs = detect_signals(asset.ticker, df["close"], thresholds=th)
        all_signals.extend(sigs)
        for s in sigs:
            db.add(
                SignalLog(ticker=s.ticker, rule=s.rule, value=float(s.strength))
            )
    if all_signals:
        db.commit()
    return SignalsOut(as_of=date.today(), signals=all_signals)
