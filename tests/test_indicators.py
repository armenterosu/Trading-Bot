import pandas as pd
import numpy as np
from app.indicators.indicators import ema, bollinger_bands, atr


def test_ema_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    e = ema(s, 3)
    assert len(e) == 5
    assert np.isfinite(e.iloc[-1])


def test_bollinger_shapes():
    s = pd.Series(np.arange(1, 51, dtype=float))
    bb = bollinger_bands(s, period=20, k=2.0)
    assert set(bb.columns) == {"ma", "upper", "lower"}
    assert len(bb) == len(s)


def test_atr_monotonicity():
    idx = pd.date_range("2022-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "open": np.linspace(100, 110, 30),
        "high": np.linspace(101, 111, 30),
        "low": np.linspace(99, 109, 30),
        "close": np.linspace(100, 110, 30),
        "volume": 1.0,
    }, index=idx)
    a = atr(df, 14)
    assert len(a) == 30
    assert np.isfinite(a.iloc[-1])
