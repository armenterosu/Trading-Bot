from __future__ import annotations
"""Bollinger Bands strategy: breakout and reversal modes."""
from typing import Dict, Any
import pandas as pd
from .base_strategy import BaseStrategy
from src.app.indicators.indicators import bollinger_bands


class BollingerStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.period = int(params.get('period', 20))
        self.k = float(params.get('k', 2.0))
        self.mode = params.get('mode', 'breakout')  # 'breakout' or 'reversal'

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = ohlcv.copy()
        bb = bollinger_bands(df['close'], self.period, self.k)
        df = df.join(bb)
        df['signal'] = 0
        if self.mode == 'breakout':
            df.loc[df['close'] > df['upper'], 'signal'] = 1
            df.loc[df['close'] < df['lower'], 'signal'] = -1
        else:  # reversal
            df.loc[df['close'] < df['lower'], 'signal'] = 1
            df.loc[df['close'] > df['upper'], 'signal'] = -1
        return df[['signal']]
