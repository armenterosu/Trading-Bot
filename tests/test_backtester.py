import pandas as pd
import numpy as np
import os
from app.backtester.backtest import Backtester


def make_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # create small deterministic dataset
    idx = pd.date_range("2022-01-01", periods=200, freq="H")
    price = np.linspace(100, 110, len(idx)) + np.sin(np.arange(len(idx))/5)
    df = pd.DataFrame({
        "timestamp": idx,
        "open": price,
        "high": price + 0.5,
        "low": price - 0.5,
        "close": price,
        "volume": 1.0,
    })
    csv = data_dir / "BTC_USDT_1h.csv"
    df.to_csv(csv, index=False)

    cfg = {
        "global": {"starting_capital": 10000},
        "strategies": {"enabled": ["ema_cross"], "ema_cross": {"fast": 5, "slow": 10, "trend": 0}},
        "indicators": {"atr_period": 14, "atr_mult": 2.0},
        "risk": {"max_risk_per_trade": 0.01},
        "backtest": {"data_path": str(data_dir), "start_date": "2022-01-01", "end_date": "2022-01-05", "commission": 0.0001, "slippage": 0.0001, "metrics_path": str(tmp_path / "metrics")},
        "logging": {"level": "INFO", "file": str(tmp_path / "logs" / "bot.log"), "retention_days": 1},
    }
    return cfg


def test_backtester_runs(tmp_path):
    cfg = make_config(tmp_path)
    bt = Backtester(cfg)
    summary = bt.run(symbol="BTC/USDT", exchange="binance", timeframe="1h")
    # Check presence of key metrics
    for key in ["CAGR", "Sharpe", "MaxDD", "DD_Duration", "Calmar", "NumTrades", "FinalEquity"]:
        assert key in summary
    # Equity files written
    eq_file = os.path.join(cfg["backtest"]["metrics_path"], "equity_BTC_USDT_1h.csv")
    assert os.path.exists(eq_file)
