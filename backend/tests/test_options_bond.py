import math

from app.services.fixed_income import (
    bond_price,
    convexity,
    fit_ns,
    macaulay_duration,
    modified_duration,
)
from app.services.options import bs_greeks, bs_price, parity_check


def test_bs_call_put_parity():
    s, k, t, r, sigma = 100, 100, 1.0, 0.05, 0.20
    c = bs_price(s, k, t, r, sigma, "call")
    p = bs_price(s, k, t, r, sigma, "put")
    assert abs(parity_check(c, p, s, k, t, r)) < 1e-6


def test_bs_greeks_delta_in_range():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2, "call")
    assert 0 < g["delta"] < 1
    g_p = bs_greeks(100, 100, 1.0, 0.05, 0.2, "put")
    assert -1 < g_p["delta"] < 0


def test_bs_price_increases_with_volatility():
    p1 = bs_price(100, 100, 1.0, 0.05, 0.10)
    p2 = bs_price(100, 100, 1.0, 0.05, 0.30)
    assert p2 > p1


def test_bond_zero_coupon_close_to_face_when_short():
    p = bond_price(face=1000, coupon_rate=0.0, ytm=0.05, years=0.5, cpy=2)
    # ZCB 6m al 5% anual => ~975
    assert 970 < p < 1000


def test_macaulay_duration_le_years():
    d = macaulay_duration(face=1000, coupon_rate=0.05, ytm=0.05, years=10, cpy=2)
    assert d <= 10


def test_modified_duration_positive():
    md = modified_duration(face=1000, coupon_rate=0.05, ytm=0.05, years=10, cpy=2)
    assert md > 0


def test_convexity_positive():
    cx = convexity(face=1000, coupon_rate=0.05, ytm=0.05, years=10, cpy=2)
    assert cx > 0


def test_ns_fits_flat_curve_approx_constant():
    import numpy as np

    mats = np.array([0.25, 1.0, 2.0, 5.0, 10.0, 30.0])
    ylds = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
    params, rmse, fitted = fit_ns(mats, ylds)
    assert rmse < 0.005
    assert all(abs(f - 0.04) < 0.01 for f in fitted)


def test_endpoint_opcion(client):
    payload = {
        "spot": 100,
        "strike": 100,
        "time_to_expiry": 1.0,
        "rf": 0.05,
        "sigma": 0.20,
        "option_type": "call",
    }
    r = client.post("/opcion/precio", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["price"] > 0
    assert -1e-3 < body["parity_check"] < 1e-3
    assert "delta" in body["greeks"]


def test_endpoint_bono_duracion(client):
    r = client.post(
        "/bono/duracion",
        json={
            "face_value": 1000,
            "coupon_rate": 0.05,
            "ytm": 0.05,
            "years": 10,
            "coupons_per_year": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["price"] > 0
    assert body["macaulay_duration"] > 0
    assert body["convexity"] > 0
    assert "+100bp" in body["sensitivity"]


def test_endpoint_opcion_rejects_negative_sigma(client):
    payload = {"spot": 100, "strike": 100, "time_to_expiry": 1, "rf": 0.05, "sigma": -0.1}
    r = client.post("/opcion/precio", json=payload)
    assert r.status_code == 422
