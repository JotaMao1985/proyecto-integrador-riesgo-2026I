"""Script de entrenamiento offline. Corre con: python -m app.ml.train

Genera app/ml/model.joblib y un metadata JSON al lado.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import settings
from app.ml.features import FEATURE_NAMES, build_xy

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"


def _synthetic_series(n: int = 600, seed: int = 7) -> "pd.Series":  # pragma: no cover
    import pandas as pd

    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.012, n)
    closes = 100 * np.exp(np.cumsum(returns))
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    return pd.Series(closes, index=dates, name="close")


def _try_real_series(ticker: str) -> "pd.Series | None":
    """Intenta yfinance; si falla, devuelve None."""
    try:
        import yfinance as yf

        df = yf.download(ticker, period="3y", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if hasattr(df.columns, "get_level_values"):
            close = df["Close"]
            if hasattr(close, "iloc") and close.ndim == 2:
                close = close.iloc[:, 0]
            return close
        return df["Close"]
    except Exception as exc:
        logger.warning("yfinance fallo: %s", exc)
        return None


def train(ticker: str | None = None) -> Path:
    ticker = (ticker or settings.ml_default_ticker).upper()
    logger.info("Entrenando modelo para ticker=%s", ticker)

    series = _try_real_series(ticker)
    if series is None:
        logger.warning("Usando serie sintetica (sin internet).")
        series = _synthetic_series()

    X, y = build_xy(series)
    if len(X) < 100:
        raise RuntimeError("Muestra insuficiente para entrenar")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=0, n_estimators=150)),
        ]
    )

    # Time-series CV para metricas honestas.
    tscv = TimeSeriesSplit(n_splits=5)
    accs, f1s, aucs = [], [], []
    for tr, te in tscv.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        accs.append(accuracy_score(y.iloc[te], pred))
        f1s.append(f1_score(y.iloc[te], pred))
        try:
            aucs.append(roc_auc_score(y.iloc[te], proba))
        except ValueError:
            aucs.append(float("nan"))

    # Fit final en todo el historico.
    pipe.fit(X, y)

    metrics = {
        "ticker_trained": ticker,
        "n_samples": int(len(X)),
        "features": FEATURE_NAMES,
        "cv_accuracy": float(np.mean(accs)),
        "cv_f1": float(np.mean(f1s)),
        "cv_roc_auc": float(np.nanmean(aucs)),
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out = Path(__file__).resolve().parent / "model.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "metadata": metrics}, out)

    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(metrics, indent=2))

    logger.info("Modelo guardado en %s", out)
    logger.info("Metricas CV: acc=%.3f f1=%.3f auc=%.3f", metrics["cv_accuracy"], metrics["cv_f1"], metrics["cv_roc_auc"])
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    train(ticker)
