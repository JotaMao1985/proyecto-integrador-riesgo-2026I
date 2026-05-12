"""Volatilidad: EWMA + familia GARCH (criterio 3 estrella)."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """EWMA estilo RiskMetrics: sigma2_t = lam*sigma2_{t-1} + (1-lam)*r_{t-1}^2."""
    r2 = (returns**2).to_numpy()
    var = np.empty_like(r2)
    var[0] = r2[0]
    for t in range(1, len(r2)):
        var[t] = lam * var[t - 1] + (1 - lam) * r2[t - 1]
    sigma = np.sqrt(var)
    return pd.Series(sigma, index=returns.index, name="ewma_sigma")


def fit_garch_family(
    returns: pd.Series, model_names: list[str] | None = None
) -> list[dict]:
    """Ajusta GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1). Devuelve lista AIC/BIC/sigma_t."""
    model_names = model_names or ["GARCH", "EGARCH", "GJR"]

    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch no instalado; GARCH no se ajusta")
        return []

    # arch espera retornos en %.
    r = returns.dropna().to_numpy() * 100.0
    out: list[dict] = []

    for name in model_names:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name == "GARCH":
                    am = arch_model(r, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
                elif name == "EGARCH":
                    am = arch_model(r, mean="Zero", vol="EGARCH", p=1, q=1, dist="normal")
                elif name == "GJR":
                    am = arch_model(r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal")
                else:
                    continue
                res = am.fit(disp="off", show_warning=False)
            sigma_last = float(res.conditional_volatility[-1]) / 100.0
            out.append(
                {
                    "name": name,
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    "sigma_last": sigma_last,
                }
            )
        except Exception as exc:  # pragma: no cover - depende de muestra
            logger.warning("GARCH fit fallo name=%s err=%s", name, exc)

    return out


def best_garch(results: list[dict]) -> str:
    if not results:
        return "n/a"
    return min(results, key=lambda r: r["aic"])["name"]
