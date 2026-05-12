"""Endpoint /volatilidad/{ticker} - EWMA + GARCH."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import GarchModelResult, VolatilityOut
from app.services.data import get_prices
from app.services.returns import log_returns
from app.services.volatility import best_garch, ewma_volatility, fit_garch_family

router = APIRouter(tags=["analisis"])


@router.get("/volatilidad/{ticker}", response_model=VolatilityOut)
async def volatilidad(
    ticker: str,
    ewma_lambda: float = Query(0.94, gt=0.5, lt=1.0),
    db: Session = Depends(get_db),
) -> VolatilityOut:
    df = get_prices(db, ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos para {ticker}")
    lr = log_returns(df["close"])
    ewma = ewma_volatility(lr, lam=ewma_lambda)
    garch = fit_garch_family(lr)
    return VolatilityOut(
        ticker=ticker.upper(),
        ewma_lambda=ewma_lambda,
        ewma_sigma=[float(x) for x in ewma.tolist()],
        garch_results=[GarchModelResult(**g) for g in garch],
        best_model=best_garch(garch),
        dates=[d for d in lr.index.tolist()],
    )
