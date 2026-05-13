"""Servicio de datos: yfinance + cache transparente en SQLite.

Resiliencia (T1.2): yfinance.download tiene retry exponencial (3 intentos)
y un circuit breaker en memoria que se abre tras 3 invocaciones con retries
agotados por ticker y se mantiene abierto durante 5 minutos. Una invocacion
fallida ya implica 3 reintentos absorbidos por tenacity, asi que el "fallo"
contado contra el umbral es post-retry.

El estado del circuit (`_circuit_state`) es modulo-level y no usa lock. Es
aceptable bajo Render free-tier (1 worker uvicorn, threadpool por defecto):
el peor caso de carrera read-modify-write es perder un incremento. Si en
algun momento se escala a multi-worker o gunicorn, mover el estado a Redis
o anadir `threading.Lock`.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

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


_circuit_state: dict[str, tuple[int, datetime]] = {}
_CIRCUIT_THRESHOLD = 3
_CIRCUIT_COOLDOWN = timedelta(minutes=5)


def _circuit_open(ticker: str) -> bool:
    """True si el ticker acumulo >= _CIRCUIT_THRESHOLD fallos en los ultimos 5 min."""
    state = _circuit_state.get(ticker)
    if state is None:
        return False
    count, last_failure_at = state
    if count < _CIRCUIT_THRESHOLD:
        return False
    if datetime.now(timezone.utc) - last_failure_at > _CIRCUIT_COOLDOWN:
        _circuit_state.pop(ticker, None)
        return False
    return True


def _circuit_record_failure(ticker: str) -> None:
    count, _ = _circuit_state.get(ticker, (0, None))
    _circuit_state[ticker] = (count + 1, datetime.now(timezone.utc))


def _circuit_reset(ticker: str) -> None:
    _circuit_state.pop(ticker, None)


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
    last_dt = datetime.combine(last, datetime.min.time(), tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last_dt) > ttl


def _yfinance_download_raw(
    ticker: str, start: date, end: date
) -> "pd.DataFrame | None":
    """Llamada cruda a yfinance.download. Aislada para mock en tests.

    Retorna None solo si yfinance no esta instalado (entorno mocheado).
    Lanza RuntimeError si la respuesta esta vacia (caso tipico de rate-limit
    silencioso). Cualquier otra excepcion se propaga para que tenacity reintente.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance no instalado; cache no se refresca")
        return None

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty for ticker={ticker}")
    return df


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _yfinance_download_with_retry(
    ticker: str, start: date, end: date
) -> "pd.DataFrame | None":
    """Wrapper con backoff exponencial (1s, 2s, 4s)."""
    return _yfinance_download_raw(ticker, start, end)


def _refresh_from_yfinance(db: Session, ticker: str, start: date, end: date) -> int:
    if _circuit_open(ticker):
        logger.info("circuit OPEN ticker=%s skip fetch", ticker)
        return 0

    try:
        df = _yfinance_download_with_retry(ticker, start, end)
    except Exception as exc:
        _circuit_record_failure(ticker)
        logger.warning("yfinance fallo tras retries ticker=%s err=%s", ticker, exc)
        return 0

    if df is None:
        # yfinance no instalado: no es fallo, no contamos contra el circuit
        return 0

    _circuit_reset(ticker)

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
