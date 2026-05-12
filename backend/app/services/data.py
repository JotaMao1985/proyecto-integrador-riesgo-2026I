"""Servicio de datos: yfinance + cache transparente en SQLite."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.db_models import Asset, Price

logger = logging.getLogger(__name__)


SEED_ASSETS: list[dict[str, str]] = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials"},
    {"ticker": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"ticker": "KO", "name": "The Coca-Cola Company", "sector": "Consumer Staples"},
]

BENCHMARK_TICKER = "SPY"


def seed_assets_if_empty(db: Session) -> int:
    """Inserta los 5 activos seed si la tabla esta vacia. Idempotente."""
    existing = db.scalar(select(Asset).limit(1))
    if existing is not None:
        return 0

    inserted = 0
    for spec in SEED_ASSETS:
        db.add(Asset(ticker=spec["ticker"], name=spec["name"], sector=spec["sector"]))
        inserted += 1
    db.commit()
    logger.info("seed_assets_if_empty inserted=%d", inserted)
    return inserted


def list_assets(db: Session) -> list[Asset]:
    return list(db.scalars(select(Asset).order_by(Asset.ticker)))


def get_prices(
    db: Session,
    ticker: str,
    *,
    start: date | None = None,
    end: date | None = None,
    auto_fetch: bool = True,
) -> pd.DataFrame:
    """Devuelve precios desde BD; rellena con yfinance si el cache es stale.

    Cache transparente: si la fecha mas reciente en BD es anterior al TTL,
    invoca yfinance para refrescar y persiste los nuevos puntos.
    """
    ticker = ticker.upper()
    end = end or date.today()
    start = start or (end - timedelta(days=365 * 2 + 30))

    if auto_fetch and _is_cache_stale(db, ticker, end):
        _refresh_from_yfinance(db, ticker, start, end)

    return _read_prices_df(db, ticker, start, end)


def _is_cache_stale(db: Session, ticker: str, end: date) -> bool:
    last = db.scalar(
        select(Price.date).where(Price.ticker == ticker).order_by(Price.date.desc())
    )
    if last is None:
        return True
    ttl = timedelta(minutes=settings.cache_ttl_minutes)
    return (datetime.utcnow() - datetime.combine(last, datetime.min.time())) > ttl


def _refresh_from_yfinance(db: Session, ticker: str, start: date, end: date) -> int:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance no instalado; cache no se refresca")
        return 0

    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:  # red caida, ticker invalido, rate limit
        logger.warning("yfinance fallo ticker=%s err=%s", ticker, exc)
        return 0

    if df is None or df.empty:
        return 0

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    existing_dates = {
        d for d in db.scalars(select(Price.date).where(Price.ticker == ticker))
    }

    new_count = 0
    for ix, row in df.iterrows():
        d = ix.date() if hasattr(ix, "date") else ix
        if d in existing_dates:
            continue
        close = float(row["Close"])
        vol = float(row.get("Volume", 0) or 0)
        db.add(Price(ticker=ticker, date=d, close=close, volume=vol))
        new_count += 1

    if new_count:
        db.commit()
        logger.info("yfinance refreshed ticker=%s new=%d", ticker, new_count)
    return new_count


def _read_prices_df(db: Session, ticker: str, start: date, end: date) -> pd.DataFrame:
    rows = list(
        db.scalars(
            select(Price)
            .where(Price.ticker == ticker, Price.date >= start, Price.date <= end)
            .order_by(Price.date)
        )
    )
    if not rows:
        return pd.DataFrame(columns=["date", "close", "volume"]).set_index("date")
    df = pd.DataFrame(
        [{"date": r.date, "close": r.close, "volume": r.volume} for r in rows]
    ).set_index("date")
    return df


def insert_synthetic_prices(
    db: Session, ticker: str, dates: Iterable[date], closes: Iterable[float]
) -> int:
    """Helper para fixtures de tests: inserta precios sin tocar yfinance."""
    count = 0
    for d, c in zip(dates, closes):
        db.add(Price(ticker=ticker, date=d, close=float(c), volume=0.0))
        count += 1
    db.commit()
    return count
