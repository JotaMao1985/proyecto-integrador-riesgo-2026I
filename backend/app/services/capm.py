"""CAPM y Beta. Riesgo libre se consulta a FRED (via macro.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(df) < 5:
        return float("nan")
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def alpha_jensen(
    asset_returns: pd.Series, market_returns: pd.Series, rf_daily: float
) -> float:
    """Alpha de Jensen: a = E[R_a] - [Rf + beta*(E[R_m] - Rf)]."""
    b = beta(asset_returns, market_returns)
    e_a = float(asset_returns.mean())
    e_m = float(market_returns.mean())
    return e_a - (rf_daily + b * (e_m - rf_daily))


def capm_expected_return(beta_v: float, rf_daily: float, market_return: float) -> float:
    return rf_daily + beta_v * (market_return - rf_daily)


def annualize_daily(r: float) -> float:
    return (1 + r) ** 252 - 1
