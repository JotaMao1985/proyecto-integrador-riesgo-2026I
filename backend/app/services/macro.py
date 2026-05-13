"""Datos macro desde FRED: Rf (tasa libre de riesgo) y curva de tesoros.

Material curricular:
- M13 (ML/integracion APIs): cliente HTTP `requests` + tenacity para retry,
  exception wrapper que evita filtrar `api_key` en logs, cache TTL in-memory.
- M5 (Pydantic settings): `settings.fred_api_key` viene de `.env`.
- M7 (Renta fija): la curva de tesoros alimenta Nelson-Siegel y duracion/convexidad.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

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


class FredRequestError(Exception):
    """Error de FRED sin URL ni params (evita leak de api_key en logs)."""


# Cache TTL de FRED (T1.7): 10 min para hits; 60s para misses (None).
# El TTL corto para None evita servir "no hay datos" durante 10 min cuando
# FRED se recupera. Usa `time.monotonic` para evitar saltos de clock.
_fred_cache: dict[str, tuple[float, float | None]] = {}
_FRED_CACHE_TTL_SECONDS = 600
_FRED_CACHE_NONE_TTL_SECONDS = 60


def reset_fred_cache() -> None:
    """Limpia el cache TTL. Pensado para tests."""
    _fred_cache.clear()


def _parse_observations(data: dict[str, Any]) -> float | None:
    """Extrae el primer valor numerico valido y lo convierte de % a decimal."""
    for obs in data.get("observations", []):
        v = obs.get("value")
        if v not in (None, "", "."):
            try:
                return float(v) / 100.0
            except ValueError:
                continue
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _fred_fetch_raw(series_id: str) -> dict[str, Any]:
    """Llamada HTTP a FRED con backoff exponencial (1s, 2s, 4s).

    Re-encapsula excepciones de `requests` para que tenacity y los logs
    nunca incluyan la URL con `api_key=...` en query string.
    """
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
        return resp.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        raise FredRequestError(f"FRED HTTP {status} series={series_id}") from None
    except requests.RequestException as exc:
        raise FredRequestError(
            f"FRED request failed series={series_id}: {type(exc).__name__}"
        ) from None


def fetch_fred_latest(series_id: str) -> float | None:
    """Obtiene el ultimo valor numerico no vacio de una serie de FRED.

    M13: integracion API externa con retry + cache TTL in-memory (10 min).
    FRED entrega porcentajes (ej. "4.52"); aqui se convierten a decimal (0.0452).
    El cache incluye resultados `None` (FRED retorno serie sin valores
    validos) para no martillar la API ante series problematicas.
    """
    if not settings.fred_api_key:
        logger.info("FRED_API_KEY vacia; uso fallback")
        return None

    # Cache hit dentro de TTL: evita el HTTP request.
    # TTL = 10 min para valores reales, 1 min para None (FRED puede recuperarse).
    entry = _fred_cache.get(series_id)
    if entry is not None:
        cached_at, cached_value = entry
        ttl = (
            _FRED_CACHE_TTL_SECONDS
            if cached_value is not None
            else _FRED_CACHE_NONE_TTL_SECONDS
        )
        if time.monotonic() - cached_at <= ttl:
            return cached_value

    try:
        data = _fred_fetch_raw(series_id)
    except Exception as exc:
        logger.warning("FRED fallo tras retries series=%s err=%s", series_id, exc)
        return None

    value = _parse_observations(data)
    _fred_cache[series_id] = (time.monotonic(), value)
    return value


def get_rf_annual() -> tuple[float, str]:
    """Devuelve `(rf_anual, fuente)`. Soporta CAPM (M4) y descuento bonos (M7)."""
    rf = fetch_fred_latest(RF_SERIES)
    if rf is None:
        return RF_FALLBACK, "fallback_static"
    return rf, f"FRED:{RF_SERIES}"


def get_rf_daily() -> float:
    rf_annual, _ = get_rf_annual()
    return (1 + rf_annual) ** (1 / 252) - 1


def fetch_yield_curve() -> tuple[list[float], list[float], str]:
    """Devuelve `(vencimientos, rendimientos_decimal, fuente)`.

    Datos para Nelson-Siegel (M7 renta fija): 6 puntos de la curva US Treasury
    (3M, 1Y, 2Y, 5Y, 10Y, 30Y). Fallback estatico si FRED no responde.
    """
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
