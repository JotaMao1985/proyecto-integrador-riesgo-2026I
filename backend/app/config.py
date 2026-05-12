"""Configuracion via BaseSettings + .env (Semana 6 del curso)."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracion tipada del backend. Lee desde .env en la raiz de backend/."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    env: str = Field(default="dev", description="dev | prod")
    app_name: str = Field(default="Riesgo API")
    log_level: str = Field(default="INFO")

    database_url: str = Field(default="sqlite:///./risk.db")

    fred_api_key: str = Field(default="", description="Clave de FRED")
    cache_ttl_minutes: int = Field(default=1440, ge=1)

    ml_model_path: str = Field(default="app/ml/model.joblib")
    ml_default_ticker: str = Field(default="AAPL")


@lru_cache
def get_settings() -> Settings:
    """Singleton de Settings (Depends-friendly)."""
    return Settings()


settings = get_settings()
