from __future__ import annotations
"""Support/Resistance strategy via pivot-based detection.

Signals:
- Long on bounce from support or breakout above resistance
- Short on rejection from resistance or breakdown below support
"""
from typing import Dict, Any
import pandas as pd
from .base_strategy import BaseStrategy


def pivots(df: pd.DataFrame, lb: int = 5) -> pd.DataFrame:
    high = df['high']
    low = df['low']
    piv_high = (high.shift(lb).rolling(lb*2+1).max() == high).astype(int)
    piv_low = (low.shift(lb).rolling(lb*2+1).min() == low).astype(int)
    out = pd.DataFrame({'piv_high': piv_high, 'piv_low': piv_low}, index=df.index)
    return out.fillna(0)


class SupportResistanceStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.lb = int(params.get('pivot_lookback', 5))

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = ohlcv.copy()
        pv = pivots(df, self.lb)
        df = df.join(pv)
        df['signal'] = 0
        # simplistic rules:
        # breakout: close above recent pivot high -> long; below pivot low -> short
        recent_high = df['high'].rolling(self.lb).max()
        recent_low = df['low'].rolling(self.lb).min()
        df.loc[df['close'] > recent_high.shift(1), 'signal'] = 1
        df.loc[df['close'] < recent_low.shift(1), 'signal'] = -1
        return df[['signal']]
