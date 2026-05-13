"""Stress testing del portafolio bajo escenarios de shock.

Implementacion conforme a la spec CIII:
- rate_shock: Δr = +200 pb (impacto sobre valuacion equity por descuento)
- market_crash_20: caida -20 % sobre el horizonte (un anio operativo)
- market_crash_30: caida -30 % sobre el horizonte
- vol_spike: σ → σ·2 (preserva media, multiplica desviaciones)
- combined: -20 % + σ·2 + Δr +200 pb (peor caso integrado)

Modelo: aplicamos los 3 shocks como modificacion de la serie de retornos
diarios del portafolio. Tasa y mercado se prorratean a impacto diario
asumiendo 252 dias habiles. Volatilidad escala (r - μ) preservando μ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.var import historical_var

TRADING_DAYS = 252


@dataclass(frozen=True)
class Scenario:
    """Parametros de un escenario de stress."""

    name: str
    rate_shock_bp: float = 0.0
    market_drop_pct: float = 0.0
    vol_multiplier: float = 1.0


DEFAULT_SCENARIOS: dict[str, Scenario] = {
    "rate_shock": Scenario("rate_shock", rate_shock_bp=200.0),
    "market_crash_20": Scenario("market_crash_20", market_drop_pct=-0.20),
    "market_crash_30": Scenario("market_crash_30", market_drop_pct=-0.30),
    "vol_spike": Scenario("vol_spike", vol_multiplier=2.0),
    "combined": Scenario(
        "combined", rate_shock_bp=200.0, market_drop_pct=-0.20, vol_multiplier=2.0
    ),
}


def _stress_series(returns: np.ndarray, scenario: Scenario) -> np.ndarray:
    """Aplica los tres shocks a una serie de retornos diarios."""
    r = np.array(returns, copy=True, dtype=float)
    if scenario.vol_multiplier != 1.0:
        mu = float(np.mean(r))
        r = (r - mu) * scenario.vol_multiplier + mu
    daily_rate_impact = -(scenario.rate_shock_bp / 10000.0) / TRADING_DAYS
    daily_market_impact = scenario.market_drop_pct / TRADING_DAYS
    return r + daily_rate_impact + daily_market_impact


def apply_scenario(
    returns: pd.DataFrame, weights: dict[str, float], scenario_name: str
) -> dict:
    """Aplica un escenario al portafolio y devuelve metricas para el frontend."""
    scenario = DEFAULT_SCENARIOS[scenario_name]
    cols = [c for c in returns.columns if c in weights]
    w = np.array([weights[c] for c in cols])
    r = returns[cols].dropna()
    portfolio_r = r.to_numpy() @ w

    var_base, _ = historical_var(portfolio_r, alpha=0.95)
    stressed = _stress_series(portfolio_r, scenario)
    var_stressed, _ = historical_var(stressed, alpha=0.95)

    # Perdida anualizada: diferencia de medias × dias habiles.
    daily_loss = float(np.mean(stressed) - np.mean(portfolio_r))
    portfolio_loss = daily_loss * TRADING_DAYS

    # Sensibilidad por activo (descomposicion lineal, ignora interacciones).
    sensitivity_by_asset: dict[str, float] = {}
    rate_term = -(scenario.rate_shock_bp / 10000.0)
    market_term = scenario.market_drop_pct
    for i, col in enumerate(cols):
        asset_sigma = float(np.std(r[col].to_numpy(), ddof=1))
        vol_term = (scenario.vol_multiplier - 1.0) * asset_sigma * np.sqrt(TRADING_DAYS)
        sensitivity_by_asset[col] = float(w[i] * (rate_term + market_term + vol_term))

    return {
        "name": scenario_name,
        "var_base": float(var_base),
        "var_stressed": float(var_stressed),
        "portfolio_loss": portfolio_loss,
        "sensitivity_by_asset": sensitivity_by_asset,
    }


def run_scenarios(
    returns: pd.DataFrame, weights: dict[str, float], scenarios: list[str]
) -> list[dict]:
    out: list[dict] = []
    for name in scenarios:
        if name not in DEFAULT_SCENARIOS:
            continue
        out.append(apply_scenario(returns, weights, name))
    return out
