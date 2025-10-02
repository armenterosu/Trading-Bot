from __future__ import annotations
"""Risk and position sizing utilities.

Includes ATR-based stop distance sizing helper.
"""
from typing import Dict
import pandas as pd


def fixed_fractional(account: Dict, risk_per_trade: float, usd_per_unit: float = 100.0) -> float:
    balance = float(account.get("equity", account.get("balance", 0.0)))
    size = max((balance * risk_per_trade) / usd_per_unit, 0.0)
    return round(size, 4)


def atr_position_size(account: Dict, risk_per_trade: float, atr_value: float, atr_mult: float, tick_value: float = 1.0) -> float:
    """Size so that risk_per_trade * balance equals stop distance in $.

    stop_distance = atr_mult * atr_value * tick_value
    qty = (balance * risk_per_trade) / max(stop_distance, 1e-9)
    """
    balance = float(account.get("equity", account.get("balance", 0.0)))
    stop_distance = atr_mult * max(atr_value, 1e-9) * tick_value
    qty = (balance * risk_per_trade) / max(stop_distance, 1e-9)
    return max(round(qty, 6), 0.0)
