import numpy as np
import pandas as pd

from app.services.indicators import bollinger, ema, macd, rsi, sma


def _series(n: int = 200) -> pd.Series:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.012, n)
    closes = 100 * np.exp(np.cumsum(rets))
    return pd.Series(closes)


def test_sma_constant_returns_constant():
    s = pd.Series([10.0] * 50)
    out = sma(s, window=10)
    assert out.dropna().iloc[-1] == 10.0


def test_ema_length_matches():
    s = _series(150)
    out = ema(s, span=20)
    assert len(out) == len(s)


def test_rsi_in_0_100_range():
    s = _series(200)
    r = rsi(s, 14).dropna()
    assert r.between(0, 100).all()


def test_macd_returns_two_series():
    s = _series(200)
    macd_l, sig = macd(s)
    assert len(macd_l) == len(s)
    assert len(sig) == len(s)


def test_bollinger_upper_above_lower():
    s = _series(200)
    up, _mid, lo = bollinger(s)
    diff = (up - lo).dropna()
    assert (diff > 0).all()


def test_indicadores_endpoint(client, seed_synthetic):
    r = client.get("/indicadores/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert "rsi_14" in body
    assert len(body["dates"]) == len(body["close"])


def test_rendimientos_endpoint(client, seed_synthetic):
    r = client.get("/rendimientos/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert "stats" in body
    assert isinstance(body["stats"]["mean"], float)
