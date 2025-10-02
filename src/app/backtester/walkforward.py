from __future__ import annotations
"""Basic walk-forward / cross-validation utilities for parameter evaluation."""
from typing import Dict, Any, List, Tuple
from copy import deepcopy
import pandas as pd

from src.app.backtester.backtest import Backtester


def time_splits(df: pd.DataFrame, n_splits: int = 3) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    n = len(df)
    if n_splits < 2:
        return [(df.index.min(), df.index.max())]
    step = n // n_splits
    splits = []
    for i in range(1, n_splits + 1):
        start = df.index[0]
        end = df.index[min(i * step - 1, n - 1)]
        splits.append((start, end))
    return splits


def walk_forward(config: Dict[str, Any], symbol: str, exchange: str, timeframe: str, n_splits: int = 3) -> List[Dict[str, Any]]:
    """Runs backtests on rolling expanding windows and returns metrics per split."""
    # We reuse Backtester data loader by temporarily patching dates in config
    bt = Backtester(config)
    # load once
    df = bt._load_ohlcv(symbol, timeframe)
    splits = time_splits(df, n_splits=n_splits)
    results: List[Dict[str, Any]] = []
    for (start, end) in splits:
        cfg = deepcopy(config)
        cfg.setdefault("backtest", {})
        cfg["backtest"]["start_date"] = str(start)
        cfg["backtest"]["end_date"] = str(end)
        res = Backtester(cfg).run(symbol=symbol, exchange=exchange, timeframe=timeframe)
        res["split_start"] = str(start)
        res["split_end"] = str(end)
        results.append(res)
    return results
