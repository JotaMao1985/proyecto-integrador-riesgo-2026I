"""Endpoint /stress: escenarios sobre el portafolio."""
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.schemas import ScenarioResult, StressOut, StressRequest
from app.services.data import get_prices
from app.services.returns import log_returns
from app.services.stress import run_scenarios
from app.services.var import historical_var

router = APIRouter(tags=["riesgo"])


@router.post("/stress", response_model=StressOut)
async def stress(req: StressRequest, db: Session = Depends(get_db)) -> StressOut:
    series = []
    cols = list(req.weights.keys())
    for t in cols:
        df = get_prices(db, t)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Sin datos para {t}")
        series.append(log_returns(df["close"]).rename(t))
    aligned = pd.concat(series, axis=1, join="inner")

    portfolio_r = aligned.to_numpy() @ [req.weights[c] for c in cols]
    base_var, _ = historical_var(portfolio_r, alpha=0.95)

    scenarios = run_scenarios(aligned, req.weights, req.scenarios)
    return StressOut(
        base_var=base_var,
        scenarios=[ScenarioResult(**s) for s in scenarios],
    )
