"""Tests del bootstrap automatico de precios en startup (T1.6)."""
from __future__ import annotations

import asyncio

import pytest

from app.main import _bootstrap_in_background
from app.scripts import seed_history
from app.status import BOOTSTRAP_STATE


@pytest.mark.asyncio
async def test_bootstrap_state_completa_si_al_menos_uno_funciona(monkeypatch):
    """Tras un seed_history exitoso, BOOTSTRAP_STATE termina en 'complete'."""

    def fake_run(*args, **kwargs):
        return {"ok": 5, "failed": 0, "total_rows_added": 100, "by_ticker": {}}

    monkeypatch.setattr(seed_history, "run", fake_run)

    BOOTSTRAP_STATE["state"] = "pending"
    await _bootstrap_in_background()
    assert BOOTSTRAP_STATE["state"] == "complete"
    assert BOOTSTRAP_STATE["details"]["ok"] == 5


@pytest.mark.asyncio
async def test_bootstrap_state_failed_si_todos_fallan(monkeypatch):
    def fake_run(*args, **kwargs):
        return {"ok": 0, "failed": 6, "total_rows_added": 0, "by_ticker": {}}

    monkeypatch.setattr(seed_history, "run", fake_run)

    BOOTSTRAP_STATE["state"] = "pending"
    await _bootstrap_in_background()
    assert BOOTSTRAP_STATE["state"] == "failed"


@pytest.mark.asyncio
async def test_bootstrap_state_failed_si_excepcion_no_recuperable(monkeypatch):
    def fake_run(*args, **kwargs):
        raise RuntimeError("DB schema missing")

    monkeypatch.setattr(seed_history, "run", fake_run)

    BOOTSTRAP_STATE["state"] = "pending"
    await _bootstrap_in_background()
    assert BOOTSTRAP_STATE["state"] == "failed"
    assert "error" in BOOTSTRAP_STATE["details"]


def test_health_endpoint_expone_bootstrap_state(client, seed_synthetic):
    BOOTSTRAP_STATE["state"] = "complete"

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["bootstrap_state"] == "complete"


def test_bootstrap_on_startup_default_true(monkeypatch):
    """Por defecto el bootstrap esta habilitado en produccion.

    Independiente de cualquier `.env` o env var local: instanciamos
    `Settings` sin leer fichero y limpiando variables de entorno relevantes.
    """
    from app.config import Settings

    monkeypatch.delenv("BOOTSTRAP_ON_STARTUP", raising=False)
    monkeypatch.delenv("BOOTSTRAP_YEARS", raising=False)

    s = Settings(_env_file=None)
    assert s.bootstrap_on_startup is True
    assert s.bootstrap_years == 2
