"""VaR (parametrico, historico, Montecarlo) + CVaR + backtesting de Kupiec."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class VaRResult:
    method: str
    var: float
    cvar: float
    kupiec_lr: float
    kupiec_pvalue: float
    kupiec_pass: bool


def parametric_var(returns: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR y CVaR bajo supuesto Normal. Devuelve magnitudes positivas (perdida)."""
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    z = float(stats.norm.ppf(1 - alpha))
    var = -(mu + z * sigma)
    cvar = -(mu - sigma * float(stats.norm.pdf(z)) / (1 - alpha))
    return float(var), float(cvar)


def historical_var(returns: np.ndarray, alpha: float) -> tuple[float, float]:
    q = float(np.quantile(returns, 1 - alpha))
    var = -q
    tail = returns[returns <= q]
    cvar = -float(np.mean(tail)) if len(tail) > 0 else var
    return var, cvar


def montecarlo_var(
    returns: np.ndarray, alpha: float, n: int = 10000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    sim = rng.normal(mu, sigma, size=n)
    return historical_var(sim, alpha)


def kupiec_pof(violations: int, n: int, alpha: float) -> tuple[float, float, bool]:
    """Test POF de Kupiec. Devuelve (LR, p-valor, pasa)."""
    p = 1 - alpha
    if n == 0 or violations < 0:
        return 0.0, 1.0, True
    x = violations
    # Evitamos log(0).
    pi = x / n if n > 0 else 0
    log_h0 = (n - x) * math.log(1 - p) + x * math.log(p) if 0 < p < 1 else 0.0
    if 0 < pi < 1:
        log_h1 = (n - x) * math.log(1 - pi) + x * math.log(pi)
    else:
        log_h1 = 0.0
    lr = -2 * (log_h0 - log_h1)
    # LR ~ chi2(1)
    pvalue = float(1 - stats.chi2.cdf(lr, df=1))
    return float(lr), pvalue, pvalue > 0.05


def count_violations(returns: np.ndarray, var: float) -> int:
    """Cuenta dias con perdida > VaR. VaR es positivo (perdida)."""
    return int(np.sum(returns < -var))


def run_all_methods(returns: np.ndarray, alpha: float, n_mc: int = 10000) -> list[VaRResult]:
    methods: list[VaRResult] = []
    for name, func in (
        ("parametric", lambda r: parametric_var(r, alpha)),
        ("historical", lambda r: historical_var(r, alpha)),
        ("montecarlo", lambda r: montecarlo_var(r, alpha, n=n_mc)),
    ):
        var, cvar = func(returns)
        viol = count_violations(returns, var)
        lr, pv, ok = kupiec_pof(viol, len(returns), alpha)
        methods.append(VaRResult(name, var, cvar, lr, pv, ok))
    return methods
