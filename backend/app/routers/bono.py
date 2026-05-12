"""Endpoint /bono/duracion."""
from fastapi import APIRouter

from app.models.schemas import BondOut, BondRequest
from app.services.fixed_income import (
    bond_price,
    convexity,
    macaulay_duration,
    modified_duration,
    price_sensitivity,
)

router = APIRouter(tags=["renta_fija"])


@router.post("/bono/duracion", response_model=BondOut)
async def bono(req: BondRequest) -> BondOut:
    args = (req.face_value, req.coupon_rate, req.ytm, req.years, req.coupons_per_year)
    p = bond_price(*args)
    md = macaulay_duration(*args)
    modd = modified_duration(*args)
    cx = convexity(*args)
    sens = price_sensitivity(*args)
    return BondOut(
        price=p,
        macaulay_duration=md,
        modified_duration=modd,
        convexity=cx,
        sensitivity=sens,
    )
