"""Indicadores tecnicos clasicos: SMA, EMA, RSI, MACD, Bollinger, Estocastico."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def ema(close: pd.Series, span: int = 20) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=1)
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def stochastic_k(close: pd.Series, window: int = 14) -> pd.Series:
    hi = close.rolling(window=window, min_periods=window).max()
    lo = close.rolling(window=window, min_periods=window).min()
    denom = (hi - lo).replace(0, np.nan)
    return 100 * (close - lo) / denom


def all_indicators(close: pd.Series) -> dict[str, pd.Series]:
    """Devuelve un dict con todos los indicadores listos para serializar."""
    macd_l, macd_s = macd(close)
    bb_up, _bb_mid, bb_lo = bollinger(close)
    return {
        "sma_20": sma(close, 20),
        "ema_20": ema(close, 20),
        "rsi_14": rsi(close, 14),
        "macd": macd_l,
        "macd_signal": macd_s,
        "bb_upper": bb_up,
        "bb_lower": bb_lo,
        "stoch_k": stochastic_k(close, 14),
    }
