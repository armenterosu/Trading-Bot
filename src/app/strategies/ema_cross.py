from __future__ import annotations
"""EMA cross strategy (fast vs slow, optional trend filter)."""
from typing import Dict, Any
import pandas as pd
from .base_strategy import BaseStrategy
from app.indicators.indicators import ema


class EmaCrossStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.fast = int(params.get('fast', 20))
        self.slow = int(params.get('slow', 50))
        self.trend = int(params.get('trend', 200))

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = ohlcv.copy()
        df['ema_fast'] = ema(df['close'], self.fast)
        df['ema_slow'] = ema(df['close'], self.slow)
        df['ema_trend'] = ema(df['close'], self.trend) if self.trend else df['close']
        df['cross'] = 0
        df.loc[df['ema_fast'] > df['ema_slow'], 'cross'] = 1
        df.loc[df['ema_fast'] < df['ema_slow'], 'cross'] = -1
        # signal when cross changes
        df['signal'] = df['cross'].diff().fillna(0)
        # filter with trend: only long if price > ema_trend; only short if price < ema_trend
        df.loc[df['close'] < df['ema_trend'], 'signal'] = df.loc[df['close'] < df['ema_trend'], 'signal'].apply(lambda s: s if s < 0 else 0)
        df.loc[df['close'] > df['ema_trend'], 'signal'] = df.loc[df['close'] > df['ema_trend'], 'signal'].apply(lambda s: s if s > 0 else 0)
        return df[['signal']]
