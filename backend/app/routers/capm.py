"""Endpoint /capm."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import CapmOut, CapmResult
from app.services.capm import alpha_jensen, beta, capm_expected_return
from app.services.data import BENCHMARK_TICKER, get_prices, list_assets
from app.services.macro import get_rf_daily
from app.services.returns import log_returns

router = APIRouter(tags=["riesgo"])


@router.get("/capm", response_model=CapmOut)
async def capm(db: Session = Depends(get_db)) -> CapmOut:
    mkt = get_prices(db, BENCHMARK_TICKER, validate_ticker=False)
    if mkt.empty:
        raise HTTPException(status_code=503, detail=f"Benchmark {BENCHMARK_TICKER} sin datos")
    rm = log_returns(mkt["close"])
    rf_daily = get_rf_daily()

    results: list[CapmResult] = []
    for asset in list_assets(db):
        df = get_prices(db, asset.ticker, validate_ticker=False)
        if df.empty:
            continue
        ra = log_returns(df["close"])
        b = beta(ra, rm)
        a = alpha_jensen(ra, rm, rf_daily)
        e = capm_expected_return(b, rf_daily, float(rm.mean()))
        results.append(
            CapmResult(
                ticker=asset.ticker,
                beta=b,
                alpha=a,
                expected_return=e,
                rf=rf_daily,
                market_return=float(rm.mean()),
            )
        )
    return CapmOut(benchmark=BENCHMARK_TICKER, rf=rf_daily, results=results)
