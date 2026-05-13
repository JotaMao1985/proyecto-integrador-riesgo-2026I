"""Engine y session de SQLAlchemy.

Material curricular: M9 (SQLAlchemy + BD relacional) ejerce `DeclarativeBase`,
`sessionmaker` y `Session`; M6 (FastAPI) ejerce el patron `Depends(get_db)`
para inyectar la sesion por request.
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base declarativa para los modelos ORM (M9: SQLAlchemy 2.0 typed)."""


engine_kwargs: dict = {"future": True}
if settings.database_url.startswith("sqlite"):
    # SQLite + FastAPI: cada request usa un hilo distinto.
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """Dependencia FastAPI: yield session, cierra al finalizar el request.

    Patron de M6 (FastAPI Depends) + M9 (Session lifecycle): la sesion se
    abre al inicio del request y se cierra siempre, incluso si hay excepcion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all() -> None:
    """Crea todas las tablas registradas en `Base.metadata` (M9).

    Invocado desde el `lifespan` de FastAPI (M6). NO migra schemas existentes:
    para evolucion controlada se usaria Alembic (bonificacion opcional).
    """
    # Import diferido para que los modelos se registren con Base antes de create_all.
    from app.models import db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
