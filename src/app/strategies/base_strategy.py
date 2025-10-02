from __future__ import annotations
"""Strategy base class and typing."""
from typing import Dict, Any
import pandas as pd


class BaseStrategy:
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def position_size(self, account: Dict[str, Any], risk_per_trade: float) -> float:
        balance = float(account.get("equity", account.get("balance", 0.0)))
        # Simplified: risk_per_trade fraction of balance, assume 1 unit per $100 for FX/CFD as placeholder
        usd_per_unit = 100.0
        size = max((balance * risk_per_trade) / usd_per_unit, 0.0)
        return round(size, 4)

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        return None
