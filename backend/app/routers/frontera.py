"""Endpoint /frontera-eficiente: Markowitz QP."""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import FrontierOut, FrontierRequest, PortfolioPoint
from app.services.data import get_prices
from app.services.macro import get_rf_daily
from app.services.portfolio import efficient_frontier
from app.services.returns import log_returns

router = APIRouter(tags=["portafolios"])


@router.post("/frontera-eficiente", response_model=FrontierOut)
async def frontera(req: FrontierRequest, db: Session = Depends(get_db)) -> FrontierOut:
    # Construye matriz alineada de retornos.
    cols: list[str] = []
    returns_list = []
    for t in req.tickers:
        df = get_prices(db, t)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Sin datos para {t}")
        returns_list.append(log_returns(df["close"]))
        cols.append(t)

    # Alineacion por interseccion de fechas.
    import pandas as pd

    aligned = pd.concat(returns_list, axis=1, join="inner")
    aligned.columns = cols
    if len(aligned) < 30:
        raise HTTPException(status_code=400, detail="Muestra alineada insuficiente")

    mu = aligned.mean().to_numpy()
    cov = aligned.cov().to_numpy()
    rf_daily = get_rf_daily()

    result = efficient_frontier(
        mu=mu, cov=cov, n_points=req.n_points, non_negative=req.non_negative, rf_daily=rf_daily
    )

    def _to_point(d: dict) -> PortfolioPoint:
        w = {cols[i]: float(d["weights"][i]) for i in range(len(cols))}
        return PortfolioPoint(ret=d["ret"], vol=d["vol"], sharpe=d["sharpe"], weights=w)

    return FrontierOut(
        non_negative=req.non_negative,
        points=[_to_point(p) for p in result["points"]],
        min_var=_to_point(result["min_var"]),
        max_sharpe=_to_point(result["max_sharpe"]),
    )
