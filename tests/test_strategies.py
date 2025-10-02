import pandas as pd
import numpy as np
from app.strategies.ema_cross import EmaCrossStrategy
from app.strategies.bollinger import BollingerStrategy
from app.strategies.support_resistance import SupportResistanceStrategy
from app.strategies.ict import IctStrategy


def make_df(n=200):
    idx = pd.date_range("2022-01-01", periods=n, freq="H")
    price = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame({
        "open": price + np.random.randn(n)*0.1,
        "high": price + np.abs(np.random.randn(n)*0.2),
        "low": price - np.abs(np.random.randn(n)*0.2),
        "close": price,
        "volume": np.random.rand(n),
    }, index=idx)
    return df


def test_ema_cross_signals():
    df = make_df()
    strat = EmaCrossStrategy({"fast": 5, "slow": 10, "trend": 0})
    sig = strat.generate_signals(df)
    assert "signal" in sig.columns
    assert len(sig) == len(df)


def test_bollinger_signals():
    df = make_df()
    strat = BollingerStrategy({"period": 20, "k": 2.0, "mode": "breakout"})
    sig = strat.generate_signals(df)
    assert "signal" in sig.columns


def test_support_resistance_signals():
    df = make_df()
    strat = SupportResistanceStrategy({"pivot_lookback": 5})
    sig = strat.generate_signals(df)
    assert "signal" in sig.columns


def test_ict_signals():
    df = make_df()
    strat = IctStrategy({"fvg_min_size_ticks": 0.0, "structure_lb": 10})
    sig = strat.generate_signals(df)
    assert "signal" in sig.columns
