from __future__ import annotations
"""Basic ICT-inspired strategy primitives (FVG detection + simple structure filter).

This is a simplified, backtestable version: detects fair value gaps (FVG) and
triggers entries when price revisits the gap in trend direction.
"""
from typing import Dict, Any
import pandas as pd
from .base_strategy import BaseStrategy


def detect_fvg(df: pd.DataFrame, min_size_ticks: float = 0.0) -> pd.DataFrame:
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    # Bullish FVG at t if low[t] > high[t-2]; Bearish if high[t] < low[t-2]
    bull = (l > h.shift(2)).astype(int)
    bear = (h < l.shift(2)).astype(int)
    out = pd.DataFrame({'bull_fvg': bull, 'bear_fvg': bear}, index=df.index)
    if min_size_ticks and min_size_ticks > 0:
        bull_size = (l - h.shift(2)).fillna(0)
        bear_size = (l.shift(2) - h).fillna(0)
        out['bull_fvg'] = (out['bull_fvg'] & (bull_size >= min_size_ticks)).astype(int)
        out['bear_fvg'] = (out['bear_fvg'] & (bear_size >= min_size_ticks)).astype(int)
    return out.fillna(0)


def swing_structure(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    # very simple: uptrend if higher highs and higher lows vs lookback
    hh = df['high'] > df['high'].rolling(lookback).max().shift(1)
    hl = df['low'] > df['low'].rolling(lookback).min().shift(1)
    up = (hh & hl).astype(int)
    down = (~up).astype(int)
    trend = pd.Series(0, index=df.index)
    trend[up] = 1
    trend[down & ~up] = -1
    return trend


class IctStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.fvg_min_size_ticks = float(params.get('fvg_min_size_ticks', 0.0))
        self.structure_lb = int(params.get('structure_lb', 10))

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = ohlcv.copy()
        fvg = detect_fvg(df, self.fvg_min_size_ticks)
        trend = swing_structure(df, self.structure_lb)
        df = df.join(fvg)
        df['trend'] = trend
        df['signal'] = 0
        # If uptrend and bullish FVG appears, signal long next bar
        df.loc[(df['trend'] > 0) & (df['bull_fvg'] == 1), 'signal'] = 1
        # If downtrend and bearish FVG appears, signal short
        df.loc[(df['trend'] < 0) & (df['bear_fvg'] == 1), 'signal'] = -1
        return df[['signal']]
