from __future__ import annotations
"""Core indicators: EMA, Bollinger Bands, ATR."""
import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def bollinger_bands(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    upper = ma + k * std
    lower = ma - k * std
    return pd.DataFrame({"ma": ma, "upper": upper, "lower": lower})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()
