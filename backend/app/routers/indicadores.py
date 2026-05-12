"""Endpoint /indicadores/{ticker}."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import IndicatorsOut
from app.services.data import get_prices
from app.services.indicators import all_indicators

router = APIRouter(tags=["analisis"])


def _none_list(s) -> list:
    return [None if x != x else float(x) for x in s.tolist()]


@router.get("/indicadores/{ticker}", response_model=IndicatorsOut)
async def indicadores(ticker: str, db: Session = Depends(get_db)) -> IndicatorsOut:
    df = get_prices(db, ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos para {ticker}")
    close = df["close"]
    ind = all_indicators(close)
    return IndicatorsOut(
        ticker=ticker.upper(),
        dates=[d for d in close.index.tolist()],
        close=[float(x) for x in close.tolist()],
        sma_20=_none_list(ind["sma_20"]),
        ema_20=_none_list(ind["ema_20"]),
        rsi_14=_none_list(ind["rsi_14"]),
        macd=_none_list(ind["macd"]),
        macd_signal=_none_list(ind["macd_signal"]),
        bb_upper=_none_list(ind["bb_upper"]),
        bb_lower=_none_list(ind["bb_lower"]),
        stoch_k=_none_list(ind["stoch_k"]),
    )
