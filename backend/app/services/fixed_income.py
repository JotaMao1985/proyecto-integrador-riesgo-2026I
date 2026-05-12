"""Renta fija: Nelson-Siegel + duracion/convexidad."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class NSParams:
    beta0: float
    beta1: float
    beta2: float
    tau: float


def nelson_siegel(t: np.ndarray, p: NSParams) -> np.ndarray:
    """Modelo Nelson-Siegel: y(t) = b0 + b1*(1-exp(-t/tau))/(t/tau) + b2*[(1-exp(-t/tau))/(t/tau) - exp(-t/tau)]."""
    ttau = t / p.tau
    factor = (1 - np.exp(-ttau)) / ttau
    return p.beta0 + p.beta1 * factor + p.beta2 * (factor - np.exp(-ttau))


def fit_ns(maturities: np.ndarray, yields: np.ndarray) -> tuple[NSParams, float, np.ndarray]:
    """Ajusta NS por MCO no lineal. Devuelve params, RMSE y fitted."""
    def loss(x: np.ndarray) -> float:
        b0, b1, b2, tau = x
        if tau <= 0:
            return 1e6
        fit = nelson_siegel(maturities, NSParams(b0, b1, b2, tau))
        return float(np.mean((fit - yields) ** 2))

    x0 = np.array([yields.mean(), -0.02, 0.02, 2.0])
    res = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-9})
    b0, b1, b2, tau = res.x
    params = NSParams(float(b0), float(b1), float(b2), float(tau))
    fitted = nelson_siegel(maturities, params)
    rmse = float(np.sqrt(np.mean((fitted - yields) ** 2)))
    return params, rmse, fitted


def bond_price(
    face: float, coupon_rate: float, ytm: float, years: float, cpy: int = 2
) -> float:
    n = int(round(years * cpy))
    c = face * coupon_rate / cpy
    y = ytm / cpy
    if n == 0:
        return face
    price = sum(c / (1 + y) ** t for t in range(1, n + 1))
    price += face / (1 + y) ** n
    return float(price)


def macaulay_duration(
    face: float, coupon_rate: float, ytm: float, years: float, cpy: int = 2
) -> float:
    n = int(round(years * cpy))
    c = face * coupon_rate / cpy
    y = ytm / cpy
    p = bond_price(face, coupon_rate, ytm, years, cpy)
    pv_w = sum((t / cpy) * (c / (1 + y) ** t) for t in range(1, n + 1))
    pv_w += (n / cpy) * (face / (1 + y) ** n)
    return float(pv_w / p)


def modified_duration(
    face: float, coupon_rate: float, ytm: float, years: float, cpy: int = 2
) -> float:
    d = macaulay_duration(face, coupon_rate, ytm, years, cpy)
    return float(d / (1 + ytm / cpy))


def convexity(
    face: float, coupon_rate: float, ytm: float, years: float, cpy: int = 2
) -> float:
    n = int(round(years * cpy))
    c = face * coupon_rate / cpy
    y = ytm / cpy
    p = bond_price(face, coupon_rate, ytm, years, cpy)
    s = sum(
        (t * (t + 1)) * (c / (1 + y) ** t)
        for t in range(1, n + 1)
    )
    s += n * (n + 1) * (face / (1 + y) ** n)
    return float(s / (p * (1 + y) ** 2 * cpy**2))


def price_sensitivity(
    face: float, coupon_rate: float, ytm: float, years: float, cpy: int = 2
) -> dict[str, dict[str, float]]:
    """Compara aproximaciones: lineal-D, D+C, reprice exacto. Shocks +-50/100/200 bp."""
    shocks_bp = [-200, -100, -50, 50, 100, 200]
    p0 = bond_price(face, coupon_rate, ytm, years, cpy)
    md = modified_duration(face, coupon_rate, ytm, years, cpy)
    cx = convexity(face, coupon_rate, ytm, years, cpy)
    out: dict[str, dict[str, float]] = {}
    for s in shocks_bp:
        dy = s / 10000.0
        p_lin = p0 * (1 - md * dy)
        p_dc = p0 * (1 - md * dy + 0.5 * cx * dy**2)
        p_exact = bond_price(face, coupon_rate, ytm + dy, years, cpy)
        out[f"{s:+d}bp"] = {
            "linear_D": p_lin,
            "D_plus_C": p_dc,
            "exact": p_exact,
        }
    return out
