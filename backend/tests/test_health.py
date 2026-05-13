def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "env" in body
    assert "app_name" in body


def test_root_returns_links(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["docs"] == "/docs"


def test_cors_preflight_returns_allow_origin(client):
    """CORSMiddleware debe responder a OPTIONS con cabecera Allow-Origin."""
    r = client.options(
        "/health",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette devuelve 200 con CORS habilitado.
    assert r.status_code in (200, 204)
    assert "access-control-allow-origin" in {k.lower() for k in r.headers.keys()}


def test_cors_simple_request_has_allow_origin(client):
    """Una peticion GET con Origin debe devolver Access-Control-Allow-Origin."""
    r = client.get("/health", headers={"Origin": "https://example.com"})
    assert r.status_code == 200
    assert "access-control-allow-origin" in {k.lower() for k in r.headers.keys()}
