"""FastAPI app. Lifespan event: create_all + seed + carga del modelo Singleton."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import SessionLocal, create_all
from app.ml import predictor as ml_predictor
from app.services.data import TickerNotFoundError
from app.status import BOOTSTRAP_STATE
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


# Referencias fuertes a tasks de background: el event loop solo guarda
# weakrefs y el GC puede recolectar el task antes de que termine si nadie
# mas lo retiene (CPython issue #88831).
_background_tasks: set[asyncio.Task] = set()


async def _bootstrap_in_background() -> None:
    """Descarga historico via `seed_history.run()` en thread aux.

    Actualiza `BOOTSTRAP_STATE` en cada transicion. Captura excepciones de
    `Exception` y las expone como state="failed"; deja propagar
    `CancelledError` para que el shutdown del lifespan funcione.
    """
    from app.scripts import seed_history

    BOOTSTRAP_STATE["state"] = "running"
    logger.info("bootstrap started years=%d", settings.bootstrap_years)
    try:
        results = await asyncio.to_thread(
            seed_history.run, None, settings.bootstrap_years
        )
        BOOTSTRAP_STATE["details"] = results
        if results["ok"] > 0:
            BOOTSTRAP_STATE["state"] = "complete"
            logger.info(
                "bootstrap complete ok=%d failed=%d total_added=%d",
                results["ok"],
                results["failed"],
                results["total_rows_added"],
            )
        else:
            BOOTSTRAP_STATE["state"] = "failed"
            logger.warning("bootstrap failed: ningun ticker exitoso")
    except asyncio.CancelledError:
        # Lifespan se esta apagando; no marcamos failed, propagamos.
        raise
    except Exception as exc:
        BOOTSTRAP_STATE["state"] = "failed"
        BOOTSTRAP_STATE["details"] = {"error": str(exc)}
        logger.exception("bootstrap error: %s", exc)


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

    if settings.bootstrap_on_startup:
        task = asyncio.create_task(_bootstrap_in_background())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    else:
        logger.info("bootstrap_on_startup=False; skip background fetch")

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
