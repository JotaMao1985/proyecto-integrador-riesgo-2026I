def test_portfolio_crud(client):
    payload = {"name": "Test 5050", "holdings": {"AAPL": 0.5, "JPM": 0.5}}
    r = client.post("/portafolios", json=payload)
    assert r.status_code == 201
    pid = r.json()["id"]

    g = client.get(f"/portafolios/{pid}")
    assert g.status_code == 200
    assert g.json()["holdings"]["AAPL"] == 0.5

    lst = client.get("/portafolios")
    assert lst.status_code == 200
    assert any(p["id"] == pid for p in lst.json())

    d = client.delete(f"/portafolios/{pid}")
    assert d.status_code == 204
    not_found = client.get(f"/portafolios/{pid}")
    assert not_found.status_code == 404


def test_portfolio_rejects_bad_weights(client):
    r = client.post("/portafolios", json={"name": "x", "holdings": {"A": 0.3, "B": 0.3}})
    assert r.status_code == 422


def test_frontera_eficiente(client, seed_synthetic):
    r = client.post(
        "/frontera-eficiente",
        json={
            "tickers": ["AAPL", "JPM", "XOM"],
            "non_negative": True,
            "n_points": 10,
            "n_random": 0,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) > 0
    mv = body["min_var"]
    # Pesos cercanos a sumar 1.
    assert abs(sum(mv["weights"].values()) - 1.0) < 0.01
    # No-negatividad: ningun peso negativo significativo.
    for w in mv["weights"].values():
        assert w >= -1e-6


def test_frontera_eficiente_con_nube_simulada(client, seed_synthetic):
    """Verifica nube Monte Carlo (T6, spec exige 10k portafolios simulados)."""
    r = client.post(
        "/frontera-eficiente",
        json={
            "tickers": ["AAPL", "JPM", "XOM"],
            "non_negative": True,
            "n_points": 5,
            "n_random": 500,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["simulated"]) == 500
    # Los simulados tienen vol >= vol de min_var (la frontera es la cota inferior).
    min_vol_frontier = body["min_var"]["vol"]
    min_vol_simulated = min(s["vol"] for s in body["simulated"])
    # Toleramos pequena divergencia numerica.
    assert min_vol_simulated >= min_vol_frontier - 1e-4


def test_simulate_random_portfolios_unit():
    """Test directo del servicio: Dirichlet pesa el simplex correctamente."""
    import numpy as np

    from app.services.portfolio import simulate_random_portfolios

    mu = np.array([0.001, 0.0008, 0.0012])
    cov = np.eye(3) * 0.0004
    points = simulate_random_portfolios(mu, cov, n=200, non_negative=True, seed=42)
    assert len(points) == 200
    # Todos los retornos deben estar entre min(mu) y max(mu) si pesos no negativos.
    rets = [p["ret"] for p in points]
    assert min(rets) >= float(mu.min()) - 1e-9
    assert max(rets) <= float(mu.max()) + 1e-9
