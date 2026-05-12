"""Engine y session de SQLAlchemy (Semana 9 del curso)."""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base declarativa para todos los modelos ORM."""


engine_kwargs: dict = {"future": True}
if settings.database_url.startswith("sqlite"):
    # SQLite + FastAPI: cada request usa un hilo distinto.
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """Dependencia FastAPI: yield session, cierra al finalizar el request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all() -> None:
    """Crea todas las tablas. Llamado por el lifespan event."""
    # Import diferido para que los modelos se registren con Base antes de create_all.
    from app.models import db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
