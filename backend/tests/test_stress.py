"""Tests de stress testing (T1 — alineado con spec CIII)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.stress import (
    DEFAULT_SCENARIOS,
    apply_scenario,
    run_scenarios,
)


def _make_returns(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "AAPL": rng.normal(0.001, 0.02, n),
            "JPM": rng.normal(0.0005, 0.018, n),
        },
        index=dates,
    )


def test_default_scenarios_match_spec_magnitudes() -> None:
    s = DEFAULT_SCENARIOS
    assert s["rate_shock"].rate_shock_bp == 200.0
    assert s["market_crash_20"].market_drop_pct == -0.20
    assert s["market_crash_30"].market_drop_pct == -0.30
    assert s["vol_spike"].vol_multiplier == 2.0
    c = s["combined"]
    assert c.rate_shock_bp == 200.0
    assert c.market_drop_pct == -0.20
    assert c.vol_multiplier == 2.0


def test_vol_spike_widens_var() -> None:
    df = _make_returns()
    res = apply_scenario(df, {"AAPL": 0.5, "JPM": 0.5}, "vol_spike")
    assert res["var_stressed"] > res["var_base"]


def test_market_crash_30_worse_than_20() -> None:
    df = _make_returns()
    w = {"AAPL": 0.5, "JPM": 0.5}
    r20 = apply_scenario(df, w, "market_crash_20")
    r30 = apply_scenario(df, w, "market_crash_30")
    assert r30["portfolio_loss"] < r20["portfolio_loss"]
    assert r30["var_stressed"] >= r20["var_stressed"]


def test_rate_shock_produces_negative_loss() -> None:
    df = _make_returns()
    res = apply_scenario(df, {"AAPL": 0.5, "JPM": 0.5}, "rate_shock")
    assert res["portfolio_loss"] < 0
    # +200 pb anual = -2 % aproximadamente.
    assert res["portfolio_loss"] == pytest.approx(-0.02, abs=1e-6)


def test_sensitivity_close_to_portfolio_loss() -> None:
    df = _make_returns()
    res = apply_scenario(df, {"AAPL": 0.6, "JPM": 0.4}, "combined")
    total = sum(res["sensitivity_by_asset"].values())
    # La descomposicion ignora interaccion media*vol; tolerancia generosa.
    assert abs(total - res["portfolio_loss"]) < 0.5


def test_run_scenarios_filters_unknown_names() -> None:
    df = _make_returns()
    out = run_scenarios(
        df,
        {"AAPL": 0.5, "JPM": 0.5},
        ["rate_shock", "nonexistent", "vol_spike"],
    )
    assert [o["name"] for o in out] == ["rate_shock", "vol_spike"]


def test_stress_endpoint_returns_five_default_scenarios(client, seed_synthetic) -> None:
    resp = client.post("/stress", json={"weights": {"AAPL": 0.5, "JPM": 0.5}})
    assert resp.status_code == 200
    data = resp.json()
    assert "base_var" in data
    names = [s["name"] for s in data["scenarios"]]
    assert names == [
        "rate_shock",
        "market_crash_20",
        "market_crash_30",
        "vol_spike",
        "combined",
    ]
    for sc in data["scenarios"]:
        assert "sensitivity_by_asset" in sc
        assert set(sc["sensitivity_by_asset"].keys()) == {"AAPL", "JPM"}


def test_stress_endpoint_rejects_unbalanced_weights(client, seed_synthetic) -> None:
    resp = client.post("/stress", json={"weights": {"AAPL": 0.5, "JPM": 0.3}})
    assert resp.status_code == 422
