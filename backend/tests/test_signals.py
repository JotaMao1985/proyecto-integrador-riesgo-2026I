"""Tests para /alertas (T3): threshold_params + persistencia en signals_log."""
from __future__ import annotations

import pandas as pd

from app.models.db_models import SignalLog
from app.services.signals import SignalThresholds, detect_signals


def test_thresholds_default_match_spec() -> None:
    th = SignalThresholds()
    assert th.rsi_overbought == 70.0
    assert th.rsi_oversold == 30.0
    assert th.bb_k == 2.0


def test_detect_signals_changes_with_custom_thresholds() -> None:
    """Cambiar el umbral RSI cambia la lista de senales devueltas."""
    # Serie con RSI alto al final (precios subiendo monotonamente).
    close = pd.Series([100 + i * 0.5 for i in range(60)])
    sigs_default = detect_signals("AAPL", close)
    sigs_strict = detect_signals(
        "AAPL", close, thresholds=SignalThresholds(rsi_overbought=99.9)
    )
    default_rules = {s.rule for s in sigs_default}
    strict_rules = {s.rule for s in sigs_strict}
    # Con umbral mas estricto, RSI overbought no deberia dispararse.
    if "rsi_overbought" in default_rules:
        assert "rsi_overbought" not in strict_rules


def test_alertas_endpoint_returns_200(client, seed_synthetic) -> None:
    r = client.get("/alertas")
    assert r.status_code == 200
    assert "signals" in r.json()


def test_alertas_persists_to_signals_log(client, seed_synthetic, test_db) -> None:
    """Cada llamada a /alertas debe insertar las senales detectadas en signals_log."""
    initial = test_db.query(SignalLog).count()
    r = client.get("/alertas")
    assert r.status_code == 200
    detected = len(r.json()["signals"])
    final = test_db.query(SignalLog).count()
    assert final - initial == detected


def test_alertas_rejects_invalid_rsi_overbought(client, seed_synthetic) -> None:
    """RSI overbought debe ser >=50; valor 30 es invalido."""
    r = client.get("/alertas?rsi_overbought=30")
    assert r.status_code == 422


def test_alertas_rejects_invalid_rsi_oversold(client, seed_synthetic) -> None:
    """RSI oversold debe ser <=50; valor 80 es invalido."""
    r = client.get("/alertas?rsi_oversold=80")
    assert r.status_code == 422


def test_alertas_rejects_invalid_bb_k(client, seed_synthetic) -> None:
    r = client.get("/alertas?bb_k=0")
    assert r.status_code == 422


def test_alertas_thresholds_change_signal_count(client, seed_synthetic) -> None:
    """Umbrales muy permisivos (RSI ovb=51, ovs=49) deberian generar mas senales."""
    r_strict = client.get("/alertas?rsi_overbought=99&rsi_oversold=1")
    r_loose = client.get("/alertas?rsi_overbought=51&rsi_oversold=49")
    assert r_strict.status_code == 200
    assert r_loose.status_code == 200
    n_strict = len(r_strict.json()["signals"])
    n_loose = len(r_loose.json()["signals"])
    assert n_loose >= n_strict
