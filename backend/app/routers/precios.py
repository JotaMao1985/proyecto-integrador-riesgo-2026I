"""Endpoint /precios/{ticker} con cache transparente."""
from fastapi import APIRouter, Depends, HTTPException
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
async def precios(ticker: str, db: Session = Depends(get_db)) -> PricesOut:
    df = get_prices(db, ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos para ticker={ticker}")
    points = [PricePoint(date=ix, close=float(row.close), volume=float(row.volume)) for ix, row in df.iterrows()]
    return PricesOut(ticker=ticker.upper(), points=points)
