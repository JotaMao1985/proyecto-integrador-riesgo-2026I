"""Rendimientos simples y logaritmicos + propiedades empiricas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def simple_returns(closes: pd.Series) -> pd.Series:
    return closes.pct_change().dropna()


def log_returns(closes: pd.Series) -> pd.Series:
    return np.log(closes / closes.shift(1)).dropna()


def descriptive_stats(returns: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {
            "mean": 0.0,
            "std": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "jarque_bera_stat": 0.0,
            "jarque_bera_pvalue": 1.0,
            "shapiro_stat": 0.0,
            "shapiro_pvalue": 1.0,
        }
    arr = returns.to_numpy()
    jb_stat, jb_p = stats.jarque_bera(arr)
    # Shapiro requiere n <= 5000; submuestreamos si hace falta.
    sh_arr = arr if len(arr) <= 5000 else np.random.default_rng(0).choice(arr, 5000, replace=False)
    sh_stat, sh_p = stats.shapiro(sh_arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "skew": float(stats.skew(arr)),
        "kurtosis": float(stats.kurtosis(arr)),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "shapiro_stat": float(sh_stat),
        "shapiro_pvalue": float(sh_p),
    }
