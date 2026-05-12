"""Schemas Pydantic v2 (Semanas 4, 5, 7): request/response tipados.

Cubre los criterios 10 y 11 de la rubrica: validacion estricta, modelos
anidados, field_validators personalizados.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,15}$")


# ---------- Activos y precios ----------


class AssetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    ticker: str
    name: str
    sector: str
    currency: str = "USD"


class PricePoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    date: date
    close: float
    volume: float


class PricesOut(BaseModel):
    ticker: str
    points: list[PricePoint]


# ---------- Rendimientos ----------


class ReturnsOut(BaseModel):
    ticker: str
    simple: list[float]
    log: list[float]
    dates: list[date]
    stats: dict[str, float]


# ---------- Indicadores ----------


class IndicatorsOut(BaseModel):
    ticker: str
    dates: list[date]
    close: list[float]
    sma_20: list[float | None]
    ema_20: list[float | None]
    rsi_14: list[float | None]
    macd: list[float | None]
    macd_signal: list[float | None]
    bb_upper: list[float | None]
    bb_lower: list[float | None]
    stoch_k: list[float | None]


# ---------- Volatilidad ----------


class VolatilityRequest(BaseModel):
    ticker: str
    ewma_lambda: float = Field(default=0.94, gt=0.5, lt=1.0)
    garch_models: list[str] = Field(default_factory=lambda: ["GARCH", "EGARCH", "GJR"])

    @field_validator("ticker")
    @classmethod
    def _ticker_format(cls, v: str) -> str:
        if not TICKER_RE.match(v.upper()):
            raise ValueError("Ticker invalido: debe ser MAYUSCULAS con .-")
        return v.upper()


class GarchModelResult(BaseModel):
    name: str
    aic: float
    bic: float
    sigma_last: float


class VolatilityOut(BaseModel):
    ticker: str
    ewma_lambda: float
    ewma_sigma: list[float]
    garch_results: list[GarchModelResult]
    best_model: str
    dates: list[date]


# ---------- VaR / CVaR ----------


class VaRRequest(BaseModel):
    weights: dict[str, float]
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    horizon_days: int = Field(default=1, ge=1, le=30)
    n_simulations: int = Field(default=10000, ge=1000, le=100000)

    @field_validator("weights")
    @classmethod
    def _weights_sum_one(cls, v: dict[str, float]) -> dict[str, float]:
        s = sum(v.values())
        if not (0.999 <= s <= 1.001):
            raise ValueError(f"Los pesos deben sumar 1.0 (suma actual={s:.4f})")
        return v


class VaRMethodResult(BaseModel):
    method: str
    var: float
    cvar: float
    kupiec_lr: float
    kupiec_pvalue: float
    kupiec_pass: bool


class VaROut(BaseModel):
    confidence: float
    horizon_days: int
    methods: list[VaRMethodResult]
    portfolio_returns: list[float]


# ---------- CAPM ----------


class CapmResult(BaseModel):
    ticker: str
    beta: float
    alpha: float
    expected_return: float
    rf: float
    market_return: float


class CapmOut(BaseModel):
    benchmark: str
    rf: float
    results: list[CapmResult]


# ---------- Markowitz ----------


class FrontierRequest(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=20)
    target_returns: list[float] | None = None
    non_negative: bool = True
    n_points: int = Field(default=30, ge=5, le=200)

    @field_validator("tickers")
    @classmethod
    def _uppercase_tickers(cls, v: list[str]) -> list[str]:
        return [t.upper() for t in v]


class PortfolioPoint(BaseModel):
    ret: float
    vol: float
    sharpe: float
    weights: dict[str, float]


class FrontierOut(BaseModel):
    non_negative: bool
    points: list[PortfolioPoint]
    min_var: PortfolioPoint
    max_sharpe: PortfolioPoint


# ---------- Portafolios (CRUD) ----------


class PortfolioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    holdings: dict[str, float]

    @field_validator("holdings")
    @classmethod
    def _holdings_sum_one(cls, v: dict[str, float]) -> dict[str, float]:
        s = sum(v.values())
        if not (0.999 <= s <= 1.001):
            raise ValueError(f"Holdings deben sumar 1.0 (suma actual={s:.4f})")
        return v


class PortfolioOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    holdings: dict[str, float]
    created_at: datetime


# ---------- Renta fija ----------


class YieldCurveOut(BaseModel):
    maturities: list[float]
    yields: list[float]
    ns_beta0: float
    ns_beta1: float
    ns_beta2: float
    ns_tau: float
    rmse: float
    fitted: list[float]


class BondRequest(BaseModel):
    face_value: float = Field(default=1000.0, gt=0)
    coupon_rate: float = Field(ge=0, le=1)
    ytm: float = Field(gt=0, lt=1)
    years: float = Field(gt=0, le=50)
    coupons_per_year: int = Field(default=2, ge=1, le=12)


class BondOut(BaseModel):
    price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    sensitivity: dict[str, dict[str, float]]


# ---------- Opciones ----------


class OptionRequest(BaseModel):
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    time_to_expiry: float = Field(gt=0, le=10, description="En anios")
    rf: float = Field(ge=0, le=1)
    sigma: float = Field(gt=0, le=5)
    option_type: str = Field(default="call")

    @field_validator("option_type")
    @classmethod
    def _opt_type(cls, v: str) -> str:
        if v.lower() not in {"call", "put"}:
            raise ValueError("option_type debe ser 'call' o 'put'")
        return v.lower()


class Greeks(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class OptionOut(BaseModel):
    option_type: str
    price: float
    greeks: Greeks
    parity_check: float


# ---------- Stress ----------


class StressRequest(BaseModel):
    weights: dict[str, float]
    scenarios: list[str] = Field(
        default_factory=lambda: ["rate_shock", "market_crash", "vol_spike", "combined"]
    )

    @field_validator("weights")
    @classmethod
    def _weights_sum_one(cls, v: dict[str, float]) -> dict[str, float]:
        s = sum(v.values())
        if not (0.999 <= s <= 1.001):
            raise ValueError(f"Pesos deben sumar 1.0 (suma actual={s:.4f})")
        return v


class ScenarioResult(BaseModel):
    name: str
    var_base: float
    var_stressed: float
    portfolio_loss: float


class StressOut(BaseModel):
    base_var: float
    scenarios: list[ScenarioResult]


# ---------- Macro ----------


class MacroOut(BaseModel):
    rf: float
    rf_source: str
    cpi_yoy: float | None = None
    fetched_at: datetime


# ---------- Senales ----------


class SignalItem(BaseModel):
    ticker: str
    rule: str
    side: str
    strength: float
    note: str


class SignalsOut(BaseModel):
    as_of: date
    signals: list[SignalItem]


# ---------- ML / Predict ----------


class PredictRequest(BaseModel):
    ticker: str
    lookback_days: int = Field(default=120, ge=30, le=1000)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class PredictOut(BaseModel):
    ticker: str
    prediction: int = Field(description="1 = sube, 0 = baja/igual")
    probability: float
    model_version: str
    features_used: list[str]
    ts: datetime


# ---------- Health ----------


class HealthOut(BaseModel):
    status: str
    env: str
    app_name: str

    @model_validator(mode="after")
    def _status_known(self) -> "HealthOut":
        if self.status not in {"ok", "degraded"}:
            raise ValueError("status inesperado")
        return self
