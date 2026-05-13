"""Modelos ORM: Asset, Price, Portfolio, PredictionLog, SignalLog."""
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Asset(Base):
    __tablename__ = "assets"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    sector: Mapped[str] = mapped_column(String(64))
    currency: Mapped[str] = mapped_column(String(8), default="USD")

    prices: Mapped[list["Price"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Asset(ticker={self.ticker!r}, sector={self.sector!r})"


class Price(Base):
    __tablename__ = "prices"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("assets.ticker"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    asset: Mapped["Asset"] = relationship(back_populates="prices")


class Portfolio(Base):
    """Portafolio del usuario: composicion en JSON {ticker: peso}."""

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    holdings: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class PredictionLog(Base):
    """Log de cada llamada a /predict (criterio 11 de la rubrica).

    Nombres alineados con spec CIII: `input_features`, `timestamp`, `actual`.
    El campo `actual` es nullable para back-fill posterior (monitoreo de drift).
    """

    __tablename__ = "prediction_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    input_features: Mapped[dict] = mapped_column(JSON)
    prediction: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    actual: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    model_version: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class SignalLog(Base):
    """Persistencia de senales tecnicas detectadas por /alertas (criterio 1)."""

    __tablename__ = "signals_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    rule: Mapped[str] = mapped_column(String(50))
    value: Mapped[float] = mapped_column(Float, default=0.0)
