"""CLI para bootstrap manual del historico de precios (T1.5).

Uso:
    python -m app.scripts.seed_history
    python -m app.scripts.seed_history --tickers AAPL,JPM --years 1

Descarga `--years` anos de precios via yfinance para cada ticker. Idempotente:
re-ejecutar no duplica filas (delegado a `_refresh_from_yfinance`, que filtra
por `existing_dates`).

Exit codes:
  0  al menos un ticker se proceso con exito
  1  todos los tickers fallaron
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from app.database import Base, SessionLocal, engine
from app.models.db_models import Asset, Price
from app.services.data import (
    BENCHMARK_TICKER,
    SEED_ASSETS,
    _refresh_from_yfinance,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _default_tickers() -> list[str]:
    return [a["ticker"] for a in SEED_ASSETS] + [BENCHMARK_TICKER]


def _ensure_asset(db: "Session", ticker: str) -> None:
    """Asegura que el ticker exista en `assets`. Si no existe lo crea con
    `sector="Unknown"` y emite un log INFO distinguible (un typo via CLI
    dejaria un Asset huerfano: el log permite detectarlo en post-mortem)."""
    if db.scalar(select(Asset.ticker).where(Asset.ticker == ticker)) is None:
        db.add(Asset(ticker=ticker, name=ticker, sector="Unknown"))
        db.commit()
        logger.info("seed_history: created new asset ticker=%s sector=Unknown", ticker)


def run(
    tickers: list[str] | None = None,
    years: int = 2,
    db: "Session | None" = None,
) -> dict[str, int | dict[str, int]]:
    """Descarga `years` anos de precios para cada ticker. Idempotente.

    Si `db` es None, abre y cierra una sesion propia (uso CLI). Si se pasa,
    asume que el caller maneja el ciclo de vida (uso desde lifespan/tests).

    Retorna `{ok, failed, total_rows_added, by_ticker}`.
    """
    tickers = [t.upper() for t in (tickers or _default_tickers())]
    end = date.today()
    start = end - timedelta(days=365 * years + 30)

    own_db = db is None
    if own_db:
        # En invocacion CLI: asegurar schema antes de la primera escritura
        # (FastAPI normalmente lo hace en lifespan, pero el CLI corre solo).
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()

    results: dict = {"ok": 0, "failed": 0, "total_rows_added": 0, "by_ticker": {}}
    try:
        for ticker in tickers:
            _ensure_asset(db, ticker)
            try:
                added = _refresh_from_yfinance(
                    db, ticker, start, end, raise_on_error=True
                )
                total = (
                    db.scalar(
                        select(func.count())
                        .select_from(Price)
                        .where(Price.ticker == ticker)
                    )
                    or 0
                )
                results["ok"] += 1
                results["total_rows_added"] += added
                results["by_ticker"][ticker] = {"added": added, "total": int(total)}
                print(f"seeding {ticker} [{added} new, {total} total]")
            except Exception as exc:
                results["failed"] += 1
                results["by_ticker"][ticker] = {"error": str(exc)}
                logger.warning("seed failed ticker=%s err=%s", ticker, exc)
                print(f"FAILED {ticker}: {exc}")
    finally:
        if own_db:
            db.close()

    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m app.scripts.seed_history",
        description="Bootstrap del historico de precios desde yfinance.",
    )
    p.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="CSV de tickers (default: 5 seed + SPY)",
    )
    p.add_argument("--years", type=int, default=2, help="anos de historico (default: 2)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args = _parse_args(argv)

    tickers: list[str] | None = None
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    results = run(tickers=tickers, years=args.years)
    print(
        f"summary: ok={results['ok']} failed={results['failed']} "
        f"total_added={results['total_rows_added']}"
    )

    # Exit 1 solo si TODOS los tickers fallaron.
    if results["ok"] == 0 and results["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
