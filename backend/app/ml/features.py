"""Feature engineering para el modelo de direccion next-day."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.indicators import ema, macd, rsi


FEATURE_NAMES: list[str] = [
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag4",
    "ret_lag5",
    "ewma_vol_20",
    "rsi_14",
    "macd_signal",
    "ema_ratio_20_50",
]


def build_features(close: pd.Series) -> pd.DataFrame:
    """Devuelve DataFrame de features alineado con close.index."""
    ret = np.log(close / close.shift(1))
    df = pd.DataFrame(index=close.index)
    for k in range(1, 6):
        df[f"ret_lag{k}"] = ret.shift(k - 1)  # lag1 = retorno mas reciente disponible
    df["ewma_vol_20"] = ret.ewm(span=20, adjust=False).std()
    df["rsi_14"] = rsi(close, 14)
    _, signal = macd(close)
    df["macd_signal"] = signal
    df["ema_ratio_20_50"] = ema(close, 20) / ema(close, 50)
    return df


def build_target(close: pd.Series) -> pd.Series:
    """y_t = 1 si return_{t+1} > 0 else 0."""
    fwd = close.shift(-1) / close - 1
    return (fwd > 0).astype(int)


def build_xy(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X = build_features(close)
    y = build_target(close)
    df = X.join(y.rename("y")).dropna()
    return df[FEATURE_NAMES], df["y"]
