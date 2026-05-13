"""Senales de compra/venta a partir de indicadores tecnicos (criterio 1)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.models.schemas import SignalItem
from app.services.indicators import all_indicators, bollinger


@dataclass(frozen=True)
class SignalThresholds:
    """Parametros configurables de las reglas de senales."""

    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bb_k: float = 2.0


def detect_signals(
    ticker: str,
    close: pd.Series,
    thresholds: SignalThresholds | None = None,
) -> list[SignalItem]:
    """Reglas: RSI sobrecompra/sobreventa, cruce MACD, ruptura Bollinger."""
    th = thresholds or SignalThresholds()
    if close.empty or len(close) < 30:
        return []

    ind = all_indicators(close)
    # Recomputar Bollinger si bb_k no es el default.
    if th.bb_k != 2.0:
        bb_up, _bb_mid, bb_lo = bollinger(close, k=th.bb_k)
        ind["bb_upper"] = bb_up
        ind["bb_lower"] = bb_lo

    signals: list[SignalItem] = []

    rsi_last = ind["rsi_14"].iloc[-1]
    if pd.notna(rsi_last):
        if rsi_last < th.rsi_oversold:
            signals.append(
                SignalItem(
                    ticker=ticker,
                    rule="rsi_oversold",
                    side="buy",
                    strength=float((th.rsi_oversold - rsi_last) / th.rsi_oversold),
                    note=f"RSI={rsi_last:.1f} bajo {th.rsi_oversold:g}",
                )
            )
        elif rsi_last > th.rsi_overbought:
            signals.append(
                SignalItem(
                    ticker=ticker,
                    rule="rsi_overbought",
                    side="sell",
                    strength=float(
                        (rsi_last - th.rsi_overbought) / (100 - th.rsi_overbought)
                    ),
                    note=f"RSI={rsi_last:.1f} sobre {th.rsi_overbought:g}",
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
