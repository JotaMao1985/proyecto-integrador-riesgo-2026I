"""Tests de las mejoras de performance T1.7: FRED cache + indice + vectorizacion."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import UniqueConstraint, select

from app.config import settings
from app.models.db_models import Price
from app.services import data as data_mod
from app.services import macro as macro_mod


@pytest.fixture
def fred_key(monkeypatch):
    monkeypatch.setattr(settings, "fred_api_key", "fake-key-for-tests")


class TestFredCache:
    def test_segunda_llamada_evita_http(self, monkeypatch, fred_key):
        """Dentro del TTL, fetch_fred_latest no invoca el raw fetch."""
        call_count = {"n": 0}

        def fake_raw(series_id):
            call_count["n"] += 1
            return {"observations": [{"value": "4.52"}]}

        monkeypatch.setattr(macro_mod, "_fred_fetch_raw", fake_raw)

        v1 = macro_mod.fetch_fred_latest("DGS3MO")
        v2 = macro_mod.fetch_fred_latest("DGS3MO")
        v3 = macro_mod.fetch_fred_latest("DGS3MO")

        assert v1 == v2 == v3 == pytest.approx(0.0452)
        assert call_count["n"] == 1

    def test_cache_expira_tras_ttl(self, monkeypatch, fred_key):
        """Despues del TTL la siguiente llamada vuelve a pegar al raw."""
        call_count = {"n": 0}

        def fake_raw(series_id):
            call_count["n"] += 1
            return {"observations": [{"value": "4.52"}]}

        monkeypatch.setattr(macro_mod, "_fred_fetch_raw", fake_raw)
        macro_mod.fetch_fred_latest("DGS3MO")
        assert call_count["n"] == 1

        # Simular paso del tiempo: mover el timestamp del cache al pasado.
        cached_at, value = macro_mod._fred_cache["DGS3MO"]
        macro_mod._fred_cache["DGS3MO"] = (
            cached_at - macro_mod._FRED_CACHE_TTL_SECONDS - 1,
            value,
        )

        macro_mod.fetch_fred_latest("DGS3MO")
        assert call_count["n"] == 2

    def test_caches_none_corto_para_series_invalida(self, monkeypatch, fred_key):
        """None se cachea con TTL corto (60s) para no servir 'no hay datos'
        durante 10 min si FRED se recupera."""
        call_count = {"n": 0}

        def fake_raw(series_id):
            call_count["n"] += 1
            return {"observations": []}

        monkeypatch.setattr(macro_mod, "_fred_fetch_raw", fake_raw)

        assert macro_mod.fetch_fred_latest("BOGUS") is None
        # 2da llamada inmediata: usa cache
        assert macro_mod.fetch_fred_latest("BOGUS") is None
        assert call_count["n"] == 1

        # Simular paso de 61s (mas que el TTL para None): vuelve a pegar.
        cached_at, value = macro_mod._fred_cache["BOGUS"]
        macro_mod._fred_cache["BOGUS"] = (
            cached_at - macro_mod._FRED_CACHE_NONE_TTL_SECONDS - 1,
            value,
        )
        macro_mod.fetch_fred_latest("BOGUS")
        assert call_count["n"] == 2


class TestIndiceCompuesto:
    def test_unique_constraint_provee_indice_compuesto(self):
        """SQLite/Postgres genera indice unico compuesto a partir del
        UniqueConstraint. Las queries WHERE ticker=X AND date BETWEEN A AND B
        lo aprovechan sin necesidad de un Index adicional."""
        constraints = [
            arg for arg in Price.__table_args__ if isinstance(arg, UniqueConstraint)
        ]
        nombres = [c.name for c in constraints]
        assert "uq_ticker_date" in nombres
        # Verifica que el constraint es sobre (ticker, date) en ese orden:
        uc = next(c for c in constraints if c.name == "uq_ticker_date")
        col_names = [c.name for c in uc.columns]
        assert col_names == ["ticker", "date"]


class TestReadPricesDFVectorizado:
    def test_matches_orm_path(self, test_db, seed_synthetic):
        """El frame vectorizado coincide con la lectura ORM cruda."""
        start = date.today() - timedelta(days=30)
        end = date.today()

        df_new = data_mod._read_prices_df(test_db, "AAPL", start, end)

        # Lectura ORM cruda (replica del codigo previo a T1.7).
        rows = list(
            test_db.scalars(
                select(Price)
                .where(
                    Price.ticker == "AAPL",
                    Price.date >= start,
                    Price.date <= end,
                )
                .order_by(Price.date)
            )
        )
        df_orm = (
            pd.DataFrame(
                [{"date": r.date, "close": r.close, "volume": r.volume} for r in rows]
            ).set_index("date")
            if rows
            else pd.DataFrame(columns=["close", "volume"]).set_index(
                pd.Index([], name="date")
            )
        )

        assert df_new.shape == df_orm.shape
        assert list(df_new.columns) == list(df_orm.columns)
        if not df_new.empty:
            assert df_new["close"].sum() == pytest.approx(df_orm["close"].sum())
            assert df_new["volume"].sum() == pytest.approx(df_orm["volume"].sum())
            # Pedagogia: el indice debe ser identico al path ORM
            # (objetos `datetime.date`), no `str` ni `datetime64`.
            assert df_new.index.equals(df_orm.index)
            assert all(isinstance(d, date) for d in df_new.index)

    def test_empty_returns_empty_indexed_frame(self, test_db, seed_synthetic):
        """Sin datos en rango, devuelve frame vacio con index 'date'."""
        df = data_mod._read_prices_df(
            test_db, "AAPL", date(1900, 1, 1), date(1900, 1, 2)
        )
        assert df.empty
        assert df.index.name == "date"
