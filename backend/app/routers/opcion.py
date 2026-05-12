"""Endpoint /opcion/precio: Black-Scholes + Greeks."""
from fastapi import APIRouter

from app.models.schemas import Greeks, OptionOut, OptionRequest
from app.services.options import bs_greeks, bs_price, parity_check

router = APIRouter(tags=["derivados"])


@router.post("/opcion/precio", response_model=OptionOut)
async def opcion(req: OptionRequest) -> OptionOut:
    p = bs_price(req.spot, req.strike, req.time_to_expiry, req.rf, req.sigma, req.option_type)
    g = bs_greeks(req.spot, req.strike, req.time_to_expiry, req.rf, req.sigma, req.option_type)
    # Para paridad necesitamos call y put.
    p_call = bs_price(req.spot, req.strike, req.time_to_expiry, req.rf, req.sigma, "call")
    p_put = bs_price(req.spot, req.strike, req.time_to_expiry, req.rf, req.sigma, "put")
    parity = parity_check(p_call, p_put, req.spot, req.strike, req.time_to_expiry, req.rf)
    return OptionOut(
        option_type=req.option_type,
        price=p,
        greeks=Greeks(**g),
        parity_check=parity,
    )
