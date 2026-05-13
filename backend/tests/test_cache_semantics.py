"""Tests de semantica del cache de precios (T1.3).

Cubre los 3 estados HIT/MISS/STALE, el handler ValueError->404 para tickers
desconocidos, y el endpoint /health/cache con contadores y total_rows.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.config import settings
from app.services import data as data_mod


@pytest.fixture(autouse=True)
def _reset_cache_state():
    data_mod.reset_cache_stats()
    data_mod._circuit_state.clear()
    yield
    data_mod.reset_cache_stats()
    data_mod._circuit_state.clear()


@pytest.fixture
def no_yfinance(monkeypatch):
    """Evita que get_prices intente refrescar desde yfinance en estos tests."""
    monkeypatch.setattr(data_mod, "_yfinance_download_raw", lambda *a, **kw: None)


@pytest.fixture
def force_hit(monkeypatch):
    """TTL muy alto -> los datos del seed (ayer) cuentan como HIT."""
    monkeypatch.setattr(settings, "cache_ttl_minutes", 60 * 24 * 365)  # 1 ano


@pytest.fixture
def force_stale(monkeypatch):
    """TTL = 1 min -> los datos del seed (ayer) cuentan como STALE."""
    monkeypatch.setattr(settings, "cache_ttl_minutes", 1)


class TestTickerValidation:
    def test_unknown_ticker_raises_value_error(self, test_db, seed_synthetic):
        with pytest.raises(ValueError, match="not in assets table"):
            data_mod.get_prices(test_db, "XYZZ_NONEXISTENT")

    def test_unknown_ticker_returns_404(self, client, seed_synthetic):
        resp = client.get("/precios/XYZZ_NONEXISTENT")
        assert resp.status_code == 404
        assert "not in assets table" in resp.json()["detail"]


class TestCacheStates:
    def test_hit_increments_counter(
        self, client, seed_synthetic, no_yfinance, force_hit
    ):
        """Ticker con precios dentro del TTL -> HIT."""
        resp = client.get("/precios/AAPL")
        assert resp.status_code == 200
        assert data_mod.CACHE_STATS["AAPL"]["hit"] == 1
        assert data_mod.CACHE_STATS["AAPL"]["miss"] == 0
        assert data_mod.CACHE_STATS["AAPL"]["stale"] == 0

    def test_miss_when_no_prices(self, test_db, seed_synthetic, no_yfinance):
        """Tabla de precios vacia para el ticker -> MISS."""
        from app.models.db_models import Asset

        test_db.add(Asset(ticker="ZZZ", name="Test", sector="Test"))
        test_db.commit()

        data_mod.get_prices(test_db, "ZZZ", auto_fetch=True)
        assert data_mod.CACHE_STATS["ZZZ"]["miss"] == 1

    def test_stale_when_ttl_expired(
        self, test_db, seed_synthetic, no_yfinance, force_stale
    ):
        """TTL=1 min con datos del seed (ayer) -> STALE."""
        data_mod.get_prices(test_db, "AAPL", auto_fetch=True)
        assert data_mod.CACHE_STATS["AAPL"]["stale"] == 1


class TestRefreshInvariant:
    """Verifica el contrato del cache transparente: refresh solo si MISS/STALE."""

    def test_hit_no_invoca_refresh(
        self, test_db, seed_synthetic, force_hit, monkeypatch
    ):
        call_count = {"n": 0}

        def spy(*args, **kwargs):
            call_count["n"] += 1
            return 0

        monkeypatch.setattr(data_mod, "_refresh_from_yfinance", spy)
        data_mod.get_prices(test_db, "AAPL", auto_fetch=True)
        assert call_count["n"] == 0

    def test_miss_invoca_refresh(
        self, test_db, seed_synthetic, monkeypatch
    ):
        from app.models.db_models import Asset

        test_db.add(Asset(ticker="ZZZ", name="Test", sector="Test"))
        test_db.commit()

        call_count = {"n": 0}

        def spy(*args, **kwargs):
            call_count["n"] += 1
            return 0

        monkeypatch.setattr(data_mod, "_refresh_from_yfinance", spy)
        data_mod.get_prices(test_db, "ZZZ", auto_fetch=True)
        assert call_count["n"] == 1

    def test_stale_invoca_refresh(
        self, test_db, seed_synthetic, force_stale, monkeypatch
    ):
        call_count = {"n": 0}

        def spy(*args, **kwargs):
            call_count["n"] += 1
            return 0

        monkeypatch.setattr(data_mod, "_refresh_from_yfinance", spy)
        data_mod.get_prices(test_db, "AAPL", auto_fetch=True)
        assert call_count["n"] == 1


class TestHealthCacheEndpoint:
    def test_returns_empty_initially(self, client, seed_synthetic):
        resp = client.get("/health/cache")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tickers"] == {}
        assert body["total_rows"] > 0  # seed_synthetic inserto ~300 filas por ticker

    def test_reflects_stats_after_calls(
        self, client, seed_synthetic, no_yfinance, force_hit
    ):
        client.get("/precios/AAPL")
        client.get("/precios/AAPL")
        client.get("/precios/JPM")

        resp = client.get("/health/cache")
        body = resp.json()
        assert body["tickers"]["AAPL"]["hit"] == 2
        assert body["tickers"]["JPM"]["hit"] == 1
        assert "total_rows" in body
