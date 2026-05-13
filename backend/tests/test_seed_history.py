"""Tests del CLI seed_history (T1.5)."""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import date

from app.scripts import seed_history
from app.services import data as data_mod


@pytest.fixture
def fake_yf_success(monkeypatch):
    """yfinance retorna un DataFrame con 3 filas distintas."""

    def fake_raw(ticker, start, end):
        return pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0], "Volume": [1e6, 1e6, 1e6]},
            index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
        )

    monkeypatch.setattr(data_mod, "_yfinance_download_raw", fake_raw)


@pytest.fixture
def fake_yf_fail(monkeypatch):
    from tenacity import wait_none

    monkeypatch.setattr(
        data_mod._yfinance_download_with_retry.retry, "wait", wait_none()
    )

    def fake_raw(ticker, start, end):
        raise RuntimeError(f"yfinance failed for {ticker}")

    monkeypatch.setattr(data_mod, "_yfinance_download_raw", fake_raw)


class TestRun:
    def test_inserta_filas_la_primera_vez(self, test_db, fake_yf_success):
        results = seed_history.run(tickers=["AAPL"], years=1, db=test_db)
        assert results["ok"] == 1
        assert results["failed"] == 0
        assert results["total_rows_added"] == 3
        assert results["by_ticker"]["AAPL"]["added"] == 3

    def test_idempotente(self, test_db, fake_yf_success):
        """Correr dos veces no duplica filas."""
        r1 = seed_history.run(tickers=["AAPL"], years=1, db=test_db)
        r2 = seed_history.run(tickers=["AAPL"], years=1, db=test_db)
        assert r1["total_rows_added"] == 3
        # 2da corrida: ya estaba todo, 0 nuevas filas pero "ok" porque no fallo
        assert r2["total_rows_added"] == 0
        assert r2["ok"] == 1
        # El total acumulado sigue siendo 3.
        assert r2["by_ticker"]["AAPL"]["total"] == 3

    def test_segunda_corrida_con_fechas_nuevas_agrega_filas(
        self, test_db, monkeypatch
    ):
        """Si yfinance devuelve fechas posteriores en la 2da corrida, se insertan."""
        call = {"n": 0}

        def fake_raw(ticker, start, end):
            call["n"] += 1
            if call["n"] == 1:
                return pd.DataFrame(
                    {"Close": [100.0, 101.0], "Volume": [1e6, 1e6]},
                    index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
                )
            # 2da corrida: trae 1 fecha vieja (skip) + 2 fechas nuevas
            return pd.DataFrame(
                {"Close": [101.0, 102.0, 103.0], "Volume": [1e6, 1e6, 1e6]},
                index=pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"]),
            )

        monkeypatch.setattr(data_mod, "_yfinance_download_raw", fake_raw)

        r1 = seed_history.run(tickers=["AAPL"], years=1, db=test_db)
        r2 = seed_history.run(tickers=["AAPL"], years=1, db=test_db)
        assert r1["total_rows_added"] == 2
        assert r2["total_rows_added"] == 2  # solo las 2 fechas nuevas
        assert r2["by_ticker"]["AAPL"]["total"] == 4

    def test_ticker_fallido_se_cuenta_pero_no_aborta_el_resto(
        self, test_db, monkeypatch
    ):
        from tenacity import wait_none

        monkeypatch.setattr(
            data_mod._yfinance_download_with_retry.retry, "wait", wait_none()
        )

        def fake_raw(ticker, start, end):
            if ticker == "FAKE":
                raise RuntimeError("FAKE no existe")
            return pd.DataFrame(
                {"Close": [100.0], "Volume": [1e6]},
                index=pd.to_datetime(["2025-01-02"]),
            )

        monkeypatch.setattr(data_mod, "_yfinance_download_raw", fake_raw)
        results = seed_history.run(tickers=["FAKE", "AAPL"], years=1, db=test_db)
        assert results["ok"] == 1
        assert results["failed"] == 1


class TestMain:
    def test_exit_1_si_todos_fallan(self, test_db, fake_yf_fail, monkeypatch):
        monkeypatch.setattr(seed_history, "SessionLocal", lambda: test_db)

        exit_code = seed_history.main(["--tickers", "FAKE1,FAKE2", "--years", "1"])
        assert exit_code == 1

    def test_exit_0_si_al_menos_uno_funciona(
        self, test_db, fake_yf_success, monkeypatch
    ):
        monkeypatch.setattr(seed_history, "SessionLocal", lambda: test_db)

        exit_code = seed_history.main(["--tickers", "AAPL", "--years", "1"])
        assert exit_code == 0

    def test_argparse_parsea_csv(self):
        args = seed_history._parse_args(["--tickers", "AAPL,JPM", "--years", "3"])
        assert args.tickers == "AAPL,JPM"
        assert args.years == 3
