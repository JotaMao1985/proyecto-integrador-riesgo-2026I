"""Endpoint /rendimientos/{ticker}."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import ReturnsOut
from app.services.data import get_prices
from app.services.returns import descriptive_stats, log_returns, simple_returns

router = APIRouter(tags=["analisis"])


@router.get("/rendimientos/{ticker}", response_model=ReturnsOut)
async def rendimientos(ticker: str, db: Session = Depends(get_db)) -> ReturnsOut:
    df = get_prices(db, ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos para {ticker}")
    s = df["close"]
    sr = simple_returns(s)
    lr = log_returns(s)
    return ReturnsOut(
        ticker=ticker.upper(),
        simple=[float(x) for x in sr.tolist()],
        log=[float(x) for x in lr.tolist()],
        dates=[d for d in lr.index.tolist()],
        stats=descriptive_stats(lr),
    )
