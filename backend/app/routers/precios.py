"""Endpoint /precios/{ticker} con cache transparente y filtros de fecha."""
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import PricePoint, PricesOut
from app.services.data import get_prices

router = APIRouter(tags=["datos"])


@router.get(
    "/precios/{ticker}",
    response_model=PricesOut,
    summary="Precios historicos (con cache SQLite transparente)",
)
async def precios(
    ticker: str,
    db: Session = Depends(get_db),
    start: date | None = Query(default=None, description="Fecha inicial (inclusiva)"),
    end: date | None = Query(default=None, description="Fecha final (inclusiva)"),
) -> PricesOut:
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=422, detail="start debe ser <= end")
    df = get_prices(db, ticker, start=start, end=end)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos para ticker={ticker}")
    points = [
        PricePoint(date=ix, close=float(row.close), volume=float(row.volume))
        for ix, row in df.iterrows()
    ]
    return PricesOut(ticker=ticker.upper(), points=points)
