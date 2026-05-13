"""Verifica que el Singleton del modelo no recarga por request."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.ml import predictor as ml_pred
from app.ml.predictor import Predictor


def _fake_predictor() -> Predictor:
    pipe = MagicMock()
    pipe.predict.return_value = np.array([1])
    pipe.predict_proba.return_value = np.array([[0.4, 0.6]])
    metadata = {
        "model_version": "test-1.0",
        "features": [
            "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
            "ewma_vol_20", "rsi_14", "macd_signal", "ema_ratio_20_50",
        ],
        "n_samples": 100,
        "cv_accuracy": 0.55,
    }
    return Predictor(pipe, metadata)


def test_singleton_uses_cached_instance():
    ml_pred.reset_predictor()
    ml_pred._predictor = _fake_predictor()
    a = ml_pred.get_predictor()
    b = ml_pred.get_predictor()
    assert a is b, "Singleton debe devolver la misma instancia entre llamadas"


def test_predict_endpoint_logs_to_db(client, seed_synthetic):
    """Verifica /predict: usa Singleton + persiste en PredictionLog."""
    ml_pred.reset_predictor()
    ml_pred._predictor = _fake_predictor()

    r = client.post("/predict", json={"ticker": "AAPL", "lookback_days": 250})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["model_version"] == "test-1.0"
    assert "log_id" in body and isinstance(body["log_id"], int)
    assert "timestamp" in body


def test_predict_persists_input_features_column(client, seed_synthetic, test_db):
    """Verifica que la columna nueva input_features se llena correctamente."""
    from app.models.db_models import PredictionLog

    ml_pred.reset_predictor()
    ml_pred._predictor = _fake_predictor()
    r = client.post("/predict", json={"ticker": "AAPL", "lookback_days": 250})
    assert r.status_code == 200
    log_id = r.json()["log_id"]
    log = test_db.get(PredictionLog, log_id)
    assert log is not None
    assert log.input_features  # dict de features no vacio
    assert log.actual is None  # nullable por defecto
    assert log.timestamp is not None


def test_actual_back_fill_endpoint(client, seed_synthetic, test_db):
    """POST /predict/{id}/actual actualiza la fila."""
    from app.models.db_models import PredictionLog

    ml_pred.reset_predictor()
    ml_pred._predictor = _fake_predictor()
    r = client.post("/predict", json={"ticker": "AAPL", "lookback_days": 250})
    log_id = r.json()["log_id"]

    r2 = client.post(f"/predict/{log_id}/actual", json={"actual": 1.0})
    assert r2.status_code == 200
    assert r2.json()["actual"] == 1.0

    log = test_db.get(PredictionLog, log_id)
    assert log.actual == 1.0


def test_actual_back_fill_404_si_id_inexistente(client, seed_synthetic):
    ml_pred.reset_predictor()
    ml_pred._predictor = _fake_predictor()
    r = client.post("/predict/99999/actual", json={"actual": 1.0})
    assert r.status_code == 404


def test_predict_rejects_unknown_ticker(client, seed_synthetic):
    ml_pred.reset_predictor()
    ml_pred._predictor = _fake_predictor()
    r = client.post("/predict", json={"ticker": "ZZZZ"})
    assert r.status_code in (400, 404)
