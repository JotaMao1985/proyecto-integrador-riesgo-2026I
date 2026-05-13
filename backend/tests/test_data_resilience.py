"""Tests de resiliencia ante fallos de APIs externas.

Cubre la tarea T1.2 del PLAN_MEJORA_CAPA_1: tenacity con backoff exponencial
en yfinance.download y circuit breaker en memoria por ticker.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from tenacity import wait_none

from app.services import data as data_mod


@pytest.fixture
def no_wait(monkeypatch):
    """Desactiva el backoff exponencial para que los tests sean rapidos."""
    monkeypatch.setattr(
        data_mod._yfinance_download_with_retry.retry, "wait", wait_none()
    )


class TestYfinanceRetry:
    def test_reintenta_tres_veces_ante_fallo(self, monkeypatch, no_wait):
        """Tenacity reintenta 3 veces antes de propagar la excepcion."""
        call_count = {"n": 0}

        def fake_raw(ticker, start, end):
            call_count["n"] += 1
            raise RuntimeError("429 too many requests")

        monkeypatch.setattr(data_mod, "_yfinance_download_raw", fake_raw)

        with pytest.raises(RuntimeError):
            data_mod._yfinance_download_with_retry(
                "AAPL", date(2025, 1, 1), date(2025, 1, 31)
            )

        assert call_count["n"] == 3

    def test_exitoso_no_reintenta(self, monkeypatch, no_wait):
        """Si la primera llamada tiene exito, no se invoca de nuevo."""
        call_count = {"n": 0}

        def fake_raw(ticker, start, end):
            call_count["n"] += 1
            return pd.DataFrame({"Close": [100.0]}, index=[date(2025, 1, 1)])

        monkeypatch.setattr(data_mod, "_yfinance_download_raw", fake_raw)

        df = data_mod._yfinance_download_with_retry(
            "AAPL", date(2025, 1, 1), date(2025, 1, 31)
        )
        assert call_count["n"] == 1
        assert df is not None


class TestCircuitBreaker:
    def test_inicial_cerrado(self):
        assert data_mod._circuit_open("AAPL") is False

    def test_abre_tras_umbral(self):
        for _ in range(3):
            data_mod._circuit_record_failure("AAPL")
        assert data_mod._circuit_open("AAPL") is True

    def test_no_abre_con_menos_de_umbral(self):
        data_mod._circuit_record_failure("AAPL")
        data_mod._circuit_record_failure("AAPL")
        assert data_mod._circuit_open("AAPL") is False

    def test_resetea_tras_cooldown(self):
        """Tras 5 min sin fallos nuevos, el circuit se cierra y se limpia."""
        old_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
        data_mod._circuit_state["AAPL"] = (5, old_ts)
        assert data_mod._circuit_open("AAPL") is False
        assert "AAPL" not in data_mod._circuit_state

    def test_reset_explicito_limpia_estado(self):
        data_mod._circuit_state["AAPL"] = (5, datetime.now(timezone.utc))
        data_mod._circuit_reset("AAPL")
        assert "AAPL" not in data_mod._circuit_state


class TestRefreshIntegration:
    def test_circuit_abierto_evita_llamada_a_yfinance(self, monkeypatch, test_db):
        """Cuando el circuit esta OPEN, _refresh_from_yfinance retorna 0 sin invocar el raw."""
        call_count = {"n": 0}

        def fake_raw(ticker, start, end):
            call_count["n"] += 1
            return pd.DataFrame({"Close": [100.0]}, index=[date(2025, 1, 1)])

        monkeypatch.setattr(data_mod, "_yfinance_download_raw", fake_raw)
        data_mod._circuit_state["AAPL"] = (5, datetime.now(timezone.utc))

        result = data_mod._refresh_from_yfinance(
            test_db, "AAPL", date(2025, 1, 1), date(2025, 1, 31)
        )

        assert result == 0
        assert call_count["n"] == 0

    def test_fallo_total_registra_failure(self, monkeypatch, no_wait, test_db):
        """Tras reintentos agotados, _refresh registra una falla en el circuit."""

        def fake_raw(ticker, start, end):
            raise RuntimeError("network down")

        monkeypatch.setattr(data_mod, "_yfinance_download_raw", fake_raw)

        result = data_mod._refresh_from_yfinance(
            test_db, "AAPL", date(2025, 1, 1), date(2025, 1, 31)
        )

        assert result == 0
        assert "AAPL" in data_mod._circuit_state
        count, _ = data_mod._circuit_state["AAPL"]
        assert count == 1

    def test_yfinance_no_instalado_no_cuenta_como_failure(self, monkeypatch, test_db):
        """Si _raw retorna None (yfinance ausente), no se registra fallo."""

        def fake_raw(ticker, start, end):
            return None

        monkeypatch.setattr(data_mod, "_yfinance_download_raw", fake_raw)

        result = data_mod._refresh_from_yfinance(
            test_db, "AAPL", date(2025, 1, 1), date(2025, 1, 31)
        )

        assert result == 0
        assert "AAPL" not in data_mod._circuit_state
