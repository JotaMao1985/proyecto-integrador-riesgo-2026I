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
