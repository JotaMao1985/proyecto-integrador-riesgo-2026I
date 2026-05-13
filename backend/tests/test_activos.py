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


def test_precios_filtra_por_rango_de_fechas(client, seed_synthetic):
    """T8: query params start y end filtran la serie."""
    # Sin filtro: serie completa.
    r_full = client.get("/precios/AAPL")
    n_full = len(r_full.json()["points"])

    # Con start reciente: serie mucho mas corta.
    from datetime import date, timedelta

    start = (date.today() - timedelta(days=60)).isoformat()
    r_partial = client.get(f"/precios/AAPL?start={start}")
    assert r_partial.status_code == 200
    n_partial = len(r_partial.json()["points"])
    assert n_partial < n_full
    # Todos los puntos devueltos deben estar dentro del rango.
    for p in r_partial.json()["points"]:
        assert p["date"] >= start


def test_precios_rechaza_start_mayor_que_end(client, seed_synthetic):
    r = client.get("/precios/AAPL?start=2025-12-01&end=2025-01-01")
    assert r.status_code == 422
