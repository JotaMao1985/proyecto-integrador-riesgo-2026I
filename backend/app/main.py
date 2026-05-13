"""FastAPI app. Lifespan event: create_all + seed + carga del modelo Singleton."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import SessionLocal, create_all
from app.ml import predictor as ml_predictor
from app.services.data import TickerNotFoundError
from app.routers import (
    activos,
    alertas,
    bono,
    capm,
    curva,
    frontera,
    health,
    indicadores,
    macro,
    opcion,
    portafolios,
    precios,
    predict,
    rendimientos,
    stress,
    var,
    volatilidad,
)
from app.services.data import seed_assets_if_empty

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
    logger.info("app startup env=%s", settings.env)
    create_all()
    db = SessionLocal()
    try:
        seed_assets_if_empty(db)
    finally:
        db.close()

    try:
        ml_predictor.load_predictor()
    except FileNotFoundError:
        logger.warning(
            "Modelo no encontrado. Entrena con: python -m app.ml.train. "
            "El endpoint /predict retornara 500 hasta entrenar."
        )

    yield
    logger.info("app shutdown")


app = FastAPI(
    title=settings.app_name,
    description=(
        "API de la solucion de referencia del Proyecto Integrador de Teoria del Riesgo "
        "(Python para APIs e IA, USTA 2026-I). Cinco capas: datos, analisis clasico, "
        "renta fija + derivados, ML y deploy."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: necesario para que el frontend desplegado (Streamlit Cloud / Render)
# pueda consumir el backend cuando viven en dominios distintos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(TickerNotFoundError)
async def _ticker_not_found_to_404(
    _request: Request, exc: TickerNotFoundError
) -> JSONResponse:
    """Mapea TickerNotFoundError (subclase de ValueError) a HTTP 404."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


app.include_router(health.router)
app.include_router(activos.router)
app.include_router(precios.router)
app.include_router(portafolios.router)
app.include_router(rendimientos.router)
app.include_router(indicadores.router)
app.include_router(volatilidad.router)
app.include_router(var.router)
app.include_router(capm.router)
app.include_router(frontera.router)
app.include_router(alertas.router)
app.include_router(macro.router)
app.include_router(curva.router)
app.include_router(bono.router)
app.include_router(opcion.router)
app.include_router(stress.router)
app.include_router(predict.router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"app": settings.app_name, "docs": "/docs", "redoc": "/redoc"}
