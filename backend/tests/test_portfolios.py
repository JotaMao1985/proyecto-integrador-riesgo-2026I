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
        json={"tickers": ["AAPL", "JPM", "XOM"], "non_negative": True, "n_points": 10},
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
