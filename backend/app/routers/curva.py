"""Endpoint /curva-rendimiento: curva FRED + Nelson-Siegel."""
import numpy as np
from fastapi import APIRouter

from app.models.schemas import YieldCurveOut
from app.services.fixed_income import fit_ns
from app.services.macro import fetch_yield_curve

router = APIRouter(tags=["renta_fija"])


@router.get("/curva-rendimiento", response_model=YieldCurveOut)
async def curva() -> YieldCurveOut:
    mats, ylds, _src = fetch_yield_curve()
    m = np.array(mats)
    y = np.array(ylds)
    params, rmse, fitted = fit_ns(m, y)
    return YieldCurveOut(
        maturities=mats,
        yields=ylds,
        ns_beta0=params.beta0,
        ns_beta1=params.beta1,
        ns_beta2=params.beta2,
        ns_tau=params.tau,
        rmse=rmse,
        fitted=fitted.tolist(),
    )
