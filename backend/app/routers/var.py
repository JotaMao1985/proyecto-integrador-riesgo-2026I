"""Endpoint /var - VaR + CVaR + Kupiec."""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import VaRMethodResult, VaROut, VaRRequest
from app.services.data import get_prices
from app.services.returns import log_returns
from app.services.var import run_all_methods

router = APIRouter(tags=["riesgo"])


@router.post("/var", response_model=VaROut)
async def var_endpoint(req: VaRRequest, db: Session = Depends(get_db)) -> VaROut:
    cols = list(req.weights.keys())
    series: list[np.ndarray] = []
    dates = None
    for t in cols:
        df = get_prices(db, t)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Sin datos para {t}")
        r = log_returns(df["close"])
        series.append(r.to_numpy())
        dates = r.index if dates is None else dates.intersection(r.index)

    # Alinear longitudes minimas.
    n = min(len(s) for s in series)
    matrix = np.column_stack([s[-n:] for s in series])
    weights = np.array([req.weights[c] for c in cols])
    portfolio = matrix @ weights

    results = run_all_methods(portfolio, req.confidence, n_mc=req.n_simulations)
    return VaROut(
        confidence=req.confidence,
        horizon_days=req.horizon_days,
        methods=[
            VaRMethodResult(
                method=r.method,
                var=r.var,
                cvar=r.cvar,
                kupiec_lr=r.kupiec_lr,
                kupiec_pvalue=r.kupiec_pvalue,
                kupiec_pass=r.kupiec_pass,
            )
            for r in results
        ],
        portfolio_returns=portfolio.tolist(),
    )
