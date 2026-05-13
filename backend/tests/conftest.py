"""Fixtures de pytest: BD en memoria + override de dependencias FastAPI.

Filosofia: los tests no deben tocar internet. yfinance/FRED se evitan
preinsertando precios sinteticos en la BD de prueba antes de cada test.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models.db_models import Asset, Price


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Aisla estado modulo-level (cache, circuit, bootstrap) entre tests."""
    from app.config import settings
    from app.services import data as data_mod
    from app.status import BOOTSTRAP_STATE

    # Desactivar bootstrap automatico: los tests no deben tocar internet.
    original_bootstrap = settings.bootstrap_on_startup
    settings.bootstrap_on_startup = False
    data_mod._circuit_state.clear()
    data_mod.reset_cache_stats()
    BOOTSTRAP_STATE.clear()
    BOOTSTRAP_STATE["state"] = "pending"
    yield
    data_mod._circuit_state.clear()
    data_mod.reset_cache_stats()
    settings.bootstrap_on_startup = original_bootstrap


@pytest.fixture(scope="function")
def test_db():
    """BD SQLite en memoria, una por test (aislamiento estricto)."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db):
    """TestClient con get_db sobrescrito por la BD de prueba."""

    def _override():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def seed_synthetic(test_db):
    """Inserta 5 activos + ~300 precios sinteticos por ticker."""
    tickers = [
        ("AAPL", "Apple Inc.", "Technology"),
        ("JPM", "JPMorgan Chase & Co.", "Financials"),
        ("XOM", "Exxon Mobil Corp.", "Energy"),
        ("JNJ", "Johnson & Johnson", "Healthcare"),
        ("KO", "The Coca-Cola Company", "Consumer Staples"),
        ("SPY", "S&P 500 ETF", "Index"),
    ]
    for t, n, s in tickers:
        test_db.add(Asset(ticker=t, name=n, sector=s))
    test_db.commit()

    rng = np.random.default_rng(42)
    end = date.today()
    days = 600
    for i, (t, _, _) in enumerate(tickers):
        mu = 0.0004 + i * 0.0001
        sigma = 0.012 + i * 0.001
        rets = rng.normal(mu, sigma, days)
        closes = 100 * np.exp(np.cumsum(rets))
        for k in range(days):
            d = end - timedelta(days=days - k)
            # Saltamos fines de semana para parecerse a datos de mercado.
            if d.weekday() >= 5:
                continue
            test_db.add(Price(ticker=t, date=d, close=float(closes[k]), volume=1e6))
    test_db.commit()
    return tickers
