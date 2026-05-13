"""Datos macro desde FRED: Rf y curva de tesoros."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from app.config import settings

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Serie de Rf: T-Bill 3m (DGS3MO). Fallback estatico si FRED falla.
RF_SERIES = "DGS3MO"
RF_FALLBACK = 0.045  # 4.5% anual

# Puntos de curva de rendimiento.
CURVE_SERIES: list[tuple[float, str]] = [
    (0.25, "DGS3MO"),
    (1.0, "DGS1"),
    (2.0, "DGS2"),
    (5.0, "DGS5"),
    (10.0, "DGS10"),
    (30.0, "DGS30"),
]


def fetch_fred_latest(series_id: str) -> float | None:
    """Obtiene el ultimo valor numerico no vacio de una serie de FRED."""
    if not settings.fred_api_key:
        logger.info("FRED_API_KEY vacia; uso fallback")
        return None
    params = {
        "series_id": series_id,
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 30,
    }
    try:
        resp = requests.get(FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("FRED fallo series=%s err=%s", series_id, exc)
        return None
    for obs in data.get("observations", []):
        v = obs.get("value")
        if v not in (None, "", "."):
            try:
                return float(v) / 100.0  # FRED entrega en %
            except ValueError:
                continue
    return None


def get_rf_annual() -> tuple[float, str]:
    """Devuelve (rf_anual, fuente)."""
    rf = fetch_fred_latest(RF_SERIES)
    if rf is None:
        return RF_FALLBACK, "fallback_static"
    return rf, f"FRED:{RF_SERIES}"


def get_rf_daily() -> float:
    rf_annual, _ = get_rf_annual()
    return (1 + rf_annual) ** (1 / 252) - 1


def fetch_yield_curve() -> tuple[list[float], list[float], str]:
    """Devuelve (vencimientos, rendimientos_decimal, fuente)."""
    mats: list[float] = []
    ylds: list[float] = []
    used_fallback = False
    for m, s in CURVE_SERIES:
        y = fetch_fred_latest(s)
        if y is None:
            used_fallback = True
            # Curva estatica de fallback (forma plausible enero 2026).
            fallback = {0.25: 0.045, 1.0: 0.044, 2.0: 0.042, 5.0: 0.041, 10.0: 0.042, 30.0: 0.043}
            y = fallback[m]
        mats.append(m)
        ylds.append(y)
    source = "fallback_static" if used_fallback else "FRED:tesoros"
    return mats, ylds, source


def now_iso() -> datetime:
    return datetime.now(timezone.utc)
