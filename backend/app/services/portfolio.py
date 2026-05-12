"""Markowitz: QP con cvxpy. Con y sin restriccion de no negatividad."""
from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


def min_variance(
    cov: np.ndarray, target_return: float | None, mu: np.ndarray, non_negative: bool
) -> tuple[float, float, np.ndarray]:
    """Resuelve min w' Sigma w s.t. sum(w)=1, opcional w>=0, opcional w'mu = target."""
    n = cov.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]
    if non_negative:
        constraints.append(w >= 0)
    if target_return is not None:
        constraints.append(mu @ w == target_return)

    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints)
    prob.solve()
    if w.value is None:
        raise ValueError("QP no convergio")
    weights = np.array(w.value).flatten()
    ret = float(mu @ weights)
    vol = float(np.sqrt(weights @ cov @ weights))
    return ret, vol, weights


def efficient_frontier(
    mu: np.ndarray, cov: np.ndarray, n_points: int, non_negative: bool, rf_daily: float = 0.0
) -> dict:
    """Frontera eficiente entre min(mu) y max(mu)."""
    lo, hi = float(mu.min()), float(mu.max())
    targets = np.linspace(lo, hi, n_points)
    points = []
    for t in targets:
        try:
            r, v, w = min_variance(cov, t, mu, non_negative)
            sharpe = (r - rf_daily) / v if v > 0 else 0.0
            points.append({"ret": r, "vol": v, "sharpe": float(sharpe), "weights": w})
        except Exception as exc:
            logger.debug("frontier point failed target=%.4f err=%s", t, exc)

    # Minima varianza global y maximo Sharpe.
    mv_r, mv_v, mv_w = min_variance(cov, None, mu, non_negative)
    mv = {"ret": mv_r, "vol": mv_v, "sharpe": float((mv_r - rf_daily) / mv_v) if mv_v > 0 else 0.0, "weights": mv_w}
    ms = max(points, key=lambda p: p["sharpe"]) if points else mv

    return {"points": points, "min_var": mv, "max_sharpe": ms}
