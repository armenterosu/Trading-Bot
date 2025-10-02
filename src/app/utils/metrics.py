from __future__ import annotations
"""Performance metrics utilities."""
from typing import Dict
import numpy as np
import pandas as pd


def compute_metrics(equity_curve: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0, "DD_Duration": 0.0,
                "Calmar": 0.0, "WinRate": 0.0, "ProfitFactor": 0.0, "Expectancy": 0.0,
                "Trades": 0}

    # CAGR
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25 if hasattr(equity_curve.index, 'freq') or isinstance(equity_curve.index, pd.DatetimeIndex) else len(equity_curve) / 252
    years = max(years, 1e-9)
    cagr = (1 + total_return) ** (1 / years) - 1

    # Sharpe (daily/periodic approximated)
    sharpe = (returns.mean() - rf) / (returns.std() + 1e-12) * np.sqrt(252)

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    dd_std = downside.std() if not downside.empty else 0.0
    sortino = (returns.mean() - rf) / (dd_std + 1e-12) * np.sqrt(252)

    # Drawdown metrics
    roll_max = equity_curve.cummax()
    dd = (equity_curve - roll_max) / roll_max
    maxdd = -dd.min()
    # Duration
    dd_dur = (dd < 0).astype(int)
    # approximate longest streak below high-water mark
    max_dur = int((dd_dur.groupby((dd_dur != dd_dur.shift()).cumsum()).cumsum()).max())

    # Placeholder trade metrics (backtester should compute trades)
    # We compute simple winrate/profit factor from positive/negative returns as an approximation
    pos = returns[returns > 0]
    neg = returns[returns < 0]
    win_rate = len(pos) / max(len(returns), 1)
    profit_factor = (pos.sum() / -neg.sum()) if not neg.empty else float('inf') if not pos.empty else 0.0
    expectancy = returns.mean()

    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "MaxDD": float(maxdd),
        "DD_Duration": float(max_dur),
        "Calmar": float(cagr / (maxdd + 1e-12)),
        "WinRate": float(win_rate),
        "ProfitFactor": float(profit_factor),
        "Expectancy": float(expectancy),
        "Trades": int(len(returns)),
    }


def save_metrics_csv(metrics: Dict[str, float], path: str) -> None:
    import os
    import csv
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
