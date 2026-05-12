"""Re-exporta dependencias de FastAPI (Semana 8 del curso)."""
from app.config import Settings, get_settings
from app.database import get_db
from app.ml.predictor import get_predictor

__all__ = ["Settings", "get_settings", "get_db", "get_predictor"]
