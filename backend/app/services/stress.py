"""Stress testing del portafolio optimo bajo escenarios de shock."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.var import historical_var


SCENARIO_SHOCKS: dict[str, dict[str, float]] = {
    "rate_shock": {"return_shock": -0.005, "vol_multiplier": 1.2},
    "market_crash": {"return_shock": -0.03, "vol_multiplier": 1.5},
    "vol_spike": {"return_shock": 0.0, "vol_multiplier": 2.0},
    "combined": {"return_shock": -0.02, "vol_multiplier": 1.8},
}


def apply_scenario(returns: pd.DataFrame, weights: dict[str, float], scenario: str) -> tuple[float, float]:
    """Devuelve (var_base, var_estresado, perdida_portafolio) bajo el escenario."""
    spec = SCENARIO_SHOCKS[scenario]
    cols = [c for c in returns.columns if c in weights]
    w = np.array([weights[c] for c in cols])
    r = returns[cols].dropna()
    portfolio_r = r.to_numpy() @ w
    var_base, _ = historical_var(portfolio_r, alpha=0.95)
    stressed = portfolio_r * spec["vol_multiplier"] + spec["return_shock"]
    var_str, _ = historical_var(stressed, alpha=0.95)
    portfolio_loss = float(np.mean(stressed))
    return float(var_base), float(var_str), portfolio_loss


def run_scenarios(
    returns: pd.DataFrame, weights: dict[str, float], scenarios: list[str]
) -> list[dict]:
    out: list[dict] = []
    for name in scenarios:
        if name not in SCENARIO_SHOCKS:
            continue
        v0, vs, loss = apply_scenario(returns, weights, name)
        out.append(
            {
                "name": name,
                "var_base": v0,
                "var_stressed": vs,
                "portfolio_loss": loss,
            }
        )
    return out
