"""Singleton del modelo ML via lifespan + module-level cache.

El modelo se carga UNA sola vez al startup de FastAPI. Tres llamadas a
get_predictor() devuelven el mismo objeto; el log "modelo cargado" aparece
una sola vez en la salida.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from app.config import settings

logger = logging.getLogger(__name__)


class Predictor:
    """Wrapper Singleton del pipeline sklearn."""

    def __init__(self, pipeline: Any, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = metadata
        self.model_version = metadata.get("model_version", "unknown")
        self.features = metadata.get("features", [])
        self.loaded_at = datetime.now(timezone.utc)

    def predict(self, X: "pd.DataFrame") -> tuple[int, float]:
        """Devuelve (clase, probabilidad_clase_positiva)."""
        pred = int(self.pipeline.predict(X)[-1])
        proba = float(self.pipeline.predict_proba(X)[-1, 1])
        return pred, proba


_predictor: Predictor | None = None


def load_predictor(model_path: str | Path | None = None) -> Predictor:
    """Carga el modelo desde disco; se llama UNA vez en el lifespan."""
    global _predictor
    if _predictor is not None:
        return _predictor

    path = Path(model_path or settings.ml_model_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent.parent / path

    if not path.exists():
        logger.warning("Modelo no encontrado en %s; entrena con 'python -m app.ml.train'", path)
        raise FileNotFoundError(f"Modelo ML no encontrado: {path}")

    blob = joblib.load(path)
    _predictor = Predictor(blob["pipeline"], blob["metadata"])
    logger.info(
        "modelo cargado version=%s n_samples=%s acc=%.3f",
        _predictor.model_version,
        _predictor.metadata.get("n_samples", "?"),
        _predictor.metadata.get("cv_accuracy", float("nan")),
    )
    return _predictor


def get_predictor() -> Predictor:
    """Dependencia FastAPI. Si el modelo no esta cargado, lanza error 503-ish."""
    if _predictor is None:
        raise RuntimeError(
            "Predictor no inicializado. Entrena el modelo (python -m app.ml.train) "
            "y reinicia el backend."
        )
    return _predictor


def reset_predictor() -> None:
    """Helper para tests: limpia el cache modulo-level."""
    global _predictor
    _predictor = None
