def test_activos_listado(client, seed_synthetic):
    r = client.get("/activos")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    tickers = {x["ticker"] for x in body}
    assert {"AAPL", "JPM", "XOM", "JNJ", "KO"} <= tickers


def test_precios_de_ticker(client, seed_synthetic):
    r = client.get("/precios/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert len(body["points"]) > 100


def test_precios_404_si_ticker_inexistente(client, seed_synthetic):
    r = client.get("/precios/NOEXISTE")
    # 404 esperado: nuestra cache vacia + fetch fallido => sin datos
    assert r.status_code in (404, 503)
