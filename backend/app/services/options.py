"""Black-Scholes europeo + 5 Greeks + paridad put-call + IV (Newton-Raphson)."""
from __future__ import annotations

import math

from scipy import stats


def _d1_d2(s: float, k: float, t: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def bs_price(s: float, k: float, t: float, r: float, sigma: float, option: str = "call") -> float:
    d1, d2 = _d1_d2(s, k, t, r, sigma)
    if option == "call":
        return s * stats.norm.cdf(d1) - k * math.exp(-r * t) * stats.norm.cdf(d2)
    return k * math.exp(-r * t) * stats.norm.cdf(-d2) - s * stats.norm.cdf(-d1)


def bs_greeks(s: float, k: float, t: float, r: float, sigma: float, option: str = "call") -> dict[str, float]:
    d1, d2 = _d1_d2(s, k, t, r, sigma)
    pdf_d1 = float(stats.norm.pdf(d1))
    if option == "call":
        delta = float(stats.norm.cdf(d1))
        theta = -(s * pdf_d1 * sigma) / (2 * math.sqrt(t)) - r * k * math.exp(-r * t) * float(stats.norm.cdf(d2))
        rho = k * t * math.exp(-r * t) * float(stats.norm.cdf(d2))
    else:
        delta = float(stats.norm.cdf(d1)) - 1
        theta = -(s * pdf_d1 * sigma) / (2 * math.sqrt(t)) + r * k * math.exp(-r * t) * float(stats.norm.cdf(-d2))
        rho = -k * t * math.exp(-r * t) * float(stats.norm.cdf(-d2))
    gamma = pdf_d1 / (s * sigma * math.sqrt(t))
    vega = s * pdf_d1 * math.sqrt(t)
    return {
        "delta": delta,
        "gamma": float(gamma),
        "vega": float(vega) / 100.0,  # por 1% de cambio en vol
        "theta": float(theta) / 365.0,  # por dia
        "rho": float(rho) / 100.0,  # por 1% de cambio en r
    }


def parity_check(call: float, put: float, s: float, k: float, t: float, r: float) -> float:
    """Devuelve C - P - (S - K*exp(-rT)). En teoria 0 para opciones europeas."""
    return call - put - (s - k * math.exp(-r * t))


def implied_vol(price: float, s: float, k: float, t: float, r: float, option: str = "call") -> float:
    """Newton-Raphson para volatilidad implicita."""
    sigma = 0.25
    for _ in range(100):
        try:
            p = bs_price(s, k, t, r, sigma, option)
            d1, _ = _d1_d2(s, k, t, r, sigma)
            vega = s * float(stats.norm.pdf(d1)) * math.sqrt(t)
            if vega < 1e-10:
                break
            diff = p - price
            if abs(diff) < 1e-7:
                return sigma
            sigma -= diff / vega
            if sigma <= 0:
                sigma = 1e-4
        except (ValueError, ZeroDivisionError):
            break
    return sigma
