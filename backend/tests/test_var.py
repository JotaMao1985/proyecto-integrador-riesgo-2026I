import numpy as np

from app.services.var import (
    historical_var,
    kupiec_pof,
    montecarlo_var,
    parametric_var,
    run_all_methods,
)


def _sample(n: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.012, n)


def test_parametric_var_positive():
    var, cvar = parametric_var(_sample(), 0.95)
    assert var > 0 and cvar > var


def test_historical_var_positive():
    var, cvar = historical_var(_sample(), 0.95)
    assert var > 0 and cvar > 0


def test_montecarlo_var_close_to_parametric():
    s = _sample()
    pv, _ = parametric_var(s, 0.95)
    mv, _ = montecarlo_var(s, 0.95, n=20000)
    assert abs(pv - mv) / pv < 0.15


def test_kupiec_pof_passes_when_violations_match_alpha():
    # Con 50 violaciones en 1000 dias y alpha=0.95 (p=0.05), pasa el test.
    lr, p, ok = kupiec_pof(violations=50, n=1000, alpha=0.95)
    assert ok is True


def test_kupiec_pof_fails_with_too_many_violations():
    # 200 violaciones en 1000 dias => muy lejos del 5% esperado.
    lr, p, ok = kupiec_pof(violations=200, n=1000, alpha=0.95)
    assert ok is False


def test_var_endpoint_three_methods(client, seed_synthetic):
    payload = {
        "weights": {"AAPL": 0.5, "JPM": 0.5},
        "confidence": 0.95,
        "horizon_days": 1,
        "n_simulations": 5000,
    }
    r = client.post("/var", json=payload)
    assert r.status_code == 200
    methods = {m["method"] for m in r.json()["methods"]}
    assert methods == {"parametric", "historical", "montecarlo"}


def test_var_weights_must_sum_one(client, seed_synthetic):
    r = client.post(
        "/var",
        json={"weights": {"AAPL": 0.3, "JPM": 0.3}, "confidence": 0.95},
    )
    assert r.status_code == 422


def test_run_all_methods_returns_three():
    s = _sample(500)
    res = run_all_methods(s, 0.95, n_mc=5000)
    assert len(res) == 3
