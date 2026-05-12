"""Senales de compra/venta a partir de indicadores."""
from __future__ import annotations

from datetime import date

import pandas as pd

from app.models.schemas import SignalItem
from app.services.indicators import all_indicators


def detect_signals(ticker: str, close: pd.Series, as_of: date | None = None) -> list[SignalItem]:
    """Reglas clasicas: RSI sobrecompra/sobreventa, cruce MACD, ruptura Bollinger."""
    if close.empty or len(close) < 30:
        return []

    ind = all_indicators(close)
    last_idx = close.index[-1]
    prev_idx = close.index[-2]

    signals: list[SignalItem] = []

    rsi_last = ind["rsi_14"].iloc[-1]
    if pd.notna(rsi_last):
        if rsi_last < 30:
            signals.append(
                SignalItem(
                    ticker=ticker, rule="rsi_oversold", side="buy",
                    strength=float((30 - rsi_last) / 30),
                    note=f"RSI={rsi_last:.1f} bajo 30",
                )
            )
        elif rsi_last > 70:
            signals.append(
                SignalItem(
                    ticker=ticker, rule="rsi_overbought", side="sell",
                    strength=float((rsi_last - 70) / 30),
                    note=f"RSI={rsi_last:.1f} sobre 70",
                )
            )

    macd_now = ind["macd"].iloc[-1]
    macd_prev = ind["macd"].iloc[-2]
    sig_now = ind["macd_signal"].iloc[-1]
    sig_prev = ind["macd_signal"].iloc[-2]
    if pd.notna(macd_now) and pd.notna(sig_now) and pd.notna(macd_prev) and pd.notna(sig_prev):
        if macd_prev < sig_prev and macd_now > sig_now:
            signals.append(
                SignalItem(
                    ticker=ticker, rule="macd_bullish_cross", side="buy",
                    strength=0.7,
                    note="MACD cruza signal hacia arriba",
                )
            )
        elif macd_prev > sig_prev and macd_now < sig_now:
            signals.append(
                SignalItem(
                    ticker=ticker, rule="macd_bearish_cross", side="sell",
                    strength=0.7,
                    note="MACD cruza signal hacia abajo",
                )
            )

    bb_up_last = ind["bb_upper"].iloc[-1]
    bb_lo_last = ind["bb_lower"].iloc[-1]
    last_close = close.iloc[-1]
    if pd.notna(bb_up_last) and last_close > bb_up_last:
        signals.append(
            SignalItem(
                ticker=ticker, rule="bb_breakout_upper", side="sell",
                strength=float((last_close - bb_up_last) / bb_up_last),
                note="Cierre por encima de la banda superior",
            )
        )
    elif pd.notna(bb_lo_last) and last_close < bb_lo_last:
        signals.append(
            SignalItem(
                ticker=ticker, rule="bb_breakout_lower", side="buy",
                strength=float((bb_lo_last - last_close) / bb_lo_last),
                note="Cierre por debajo de la banda inferior",
            )
        )

    return signals
