from __future__ import annotations
"""Core indicators: EMA, Bollinger Bands, ATR."""
import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int = 50) -> pd.Series:
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


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Suavizado Wilder: primer valor = suma de los primeros `period` valores,
    luego smoothed[i] = smoothed[i-1] - (smoothed[i-1]/period) + series[i]
    """
    out = pd.Series(index=series.index, dtype='float64')
    if len(series) < period:
        return out  # todo NaN
    first = series.iloc[:period].sum()
    out.iloc[period - 1] = first
    for i in range(period, len(series)):
        out.iloc[i] = out.iloc[i - 1] - (out.iloc[i - 1] / period) + series.iloc[i]
    return out


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas ['+DI', '-DI', 'ADX'].
    No modifica `ohlcv`.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    up_move = high.diff()
    down_move = low.shift(1) - low  # positivo cuando baja el low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_smooth = _wilder_smooth(tr.fillna(0), period)
    plus_dm_smooth = _wilder_smooth(pd.Series(plus_dm, index=df.index).fillna(0), period)
    minus_dm_smooth = _wilder_smooth(pd.Series(minus_dm, index=df.index).fillna(0), period)

    # evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

    plus_di = plus_di.replace([np.inf, -np.inf], np.nan)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ( (plus_di - minus_di).abs() / denom ) * 100
    dx = dx.fillna(0)

    adx_series = _wilder_smooth(dx, period)

    result = pd.DataFrame({
        '+DI': plus_di,
        '-DI': minus_di,
        'ADX': adx_series
    }, index=df.index)

    return result