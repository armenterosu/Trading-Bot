from __future__ import annotations
"""Backtester: simulates strategy execution over OHLCV with slippage, commission, ATR stops.

Outputs an equity curve and summary metrics (CAGR, Sharpe, MaxDD, DD Duration, Calmar, WinRate,
Profit Factor, Expectancy, number of trades) plus PnL summary.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import os
import json
import pandas as pd
import numpy as np
from loguru import logger

from app.utils.metrics import compute_metrics, save_metrics_csv
from app.indicators.indicators import atr
import importlib


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    side: int  # +1 long, -1 short
    qty: float
    stop: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None

    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        direction = 1 if self.side > 0 else -1
        return (self.exit_price - self.entry_price) * direction * self.qty


class Backtester:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.bt = config.get("backtest", {})
        self.risk = config.get("risk", {})

    def _load_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        data_path = self.bt.get("data_path", "data/")
        base = symbol.replace("/", "_") + f"_{timeframe}"
        # Try parquet then csv
        pq = os.path.join(data_path, base + ".parquet")
        csv = os.path.join(data_path, base + ".csv")
        if os.path.exists(pq):
            df = pd.read_parquet(pq)
        elif os.path.exists(csv):
            df = pd.read_csv(csv)
        else:
            # As fallback, create a small synthetic dataset for demo
            logger.warning(f"No data found at {pq} or {csv}. Generating synthetic OHLCV for demo.")
            idx = pd.date_range(self.bt.get("start_date", "2022-01-01"), self.bt.get("end_date", "2022-01-31"), freq="1H")
            price = np.cumsum(np.random.randn(len(idx))) + 100.0
            df = pd.DataFrame({
                "timestamp": idx,
                "open": price + np.random.randn(len(idx))*0.2,
                "high": price + np.abs(np.random.randn(len(idx))*0.5),
                "low": price - np.abs(np.random.randn(len(idx))*0.5),
                "close": price,
                "volume": np.random.rand(len(idx))*10,
            })
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        return df[["open", "high", "low", "close", "volume"]]

    def _load_strategies(self) -> List[Any]:
        strategies_cfg = self.cfg.get("strategies", {})
        enabled = strategies_cfg.get("enabled", [])
        loaded: List[Any] = []
        for name in enabled:
            module_name = f"app.strategies.{name}"
            class_name = ''.join(part.capitalize() for part in name.split('_')) + "Strategy"
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                params = strategies_cfg.get(name, {})
                loaded.append(cls(params))
            except Exception as e:
                logger.error(f"Failed to load strategy {name}: {e}")
        return loaded

    def _commission_slippage(self, price: float) -> float:
        commission = float(self.bt.get("commission", 0.0))
        slippage = float(self.bt.get("slippage", 0.0))
        # price impact
        return price * (1 + slippage) * (1 + commission)

    def _commission_slippage_exit(self, price: float) -> float:
        commission = float(self.bt.get("commission", 0.0))
        slippage = float(self.bt.get("slippage", 0.0))
        return price * (1 - slippage) * (1 - commission)

    def run(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        df = self._load_ohlcv(symbol, timeframe)
        start_date = pd.to_datetime(self.bt.get("start_date")) if self.bt.get("start_date") else df.index.min()
        end_date = pd.to_datetime(self.bt.get("end_date")) if self.bt.get("end_date") else df.index.max()
        df = df.loc[(df.index >= start_date) & (df.index <= end_date)].copy()
        if len(df) < 50:
            raise ValueError("Insufficient data for backtest")

        strategies = self._load_strategies()

        # Aggregate signals (sum of signals; could be weighted)
        signals_all = []
        for strat in strategies:
            sig = strat.generate_signals(df)
            signals_all.append(sig.rename(columns={"signal": f"signal_{strat.__class__.__name__}"}))
        signals = pd.concat(signals_all, axis=1).fillna(0)
        signals['signal'] = signals.sum(axis=1).clip(-1, 1)

        # ATR-based stop
        atr_period = int(self.cfg.get("indicators", {}).get("atr_period", 14))
        atr_series = atr(df, atr_period)
        atr_mult = float(self.cfg.get("indicators", {}).get("atr_mult", 2.0))

        equity = float(self.cfg.get("global", {}).get("starting_capital", 10_000))
        equity_curve = []
        peak = equity
        open_trade: Trade | None = None
        trades: List[Trade] = []

        for ts, row in df.iterrows():
            price = float(row['close'])
            # check existing trade stop
            if open_trade is not None:
                if open_trade.side > 0:
                    # long stop
                    if row['low'] <= open_trade.stop:
                        exit_price = self._commission_slippage_exit(open_trade.stop)
                        open_trade.exit_time = ts
                        open_trade.exit_price = exit_price
                        equity += open_trade.pnl()
                        trades.append(open_trade)
                        open_trade = None
                else:
                    # short stop
                    if row['high'] >= open_trade.stop:
                        exit_price = self._commission_slippage_exit(open_trade.stop)
                        open_trade.exit_time = ts
                        open_trade.exit_price = exit_price
                        equity += open_trade.pnl()
                        trades.append(open_trade)
                        open_trade = None

            sig = int(signals.loc[ts, 'signal']) if ts in signals.index else 0
            current_atr = float(atr_series.loc[ts]) if ts in atr_series.index and not np.isnan(atr_series.loc[ts]) else None

            # simple execution: open/flip position on non-zero signal when flat
            if open_trade is None and sig != 0 and current_atr is not None:
                side = sig
                qty = max(equity * self.risk.get("max_risk_per_trade", 0.01) / max(current_atr * atr_mult, 1e-9), 0.0)
                entry = self._commission_slippage(price)
                if side > 0:
                    stop = entry - current_atr * atr_mult
                else:
                    stop = entry + current_atr * atr_mult
                open_trade = Trade(entry_time=ts, entry_price=entry, side=side, qty=qty, stop=stop)

            # optional trailing stop: update stop with ATR
            if open_trade is not None and current_atr is not None:
                if open_trade.side > 0:
                    new_stop = price - current_atr * atr_mult
                    open_trade.stop = max(open_trade.stop, new_stop)
                else:
                    new_stop = price + current_atr * atr_mult
                    open_trade.stop = min(open_trade.stop, new_stop)

            equity_curve.append((ts, equity))
            peak = max(peak, equity)

        eq = pd.Series({ts: val for ts, val in equity_curve}).sort_index()

        # Close any open trade at last price
        if open_trade is not None:
            last_price = float(df.iloc[-1]['close'])
            open_trade.exit_time = df.index[-1]
            open_trade.exit_price = self._commission_slippage_exit(last_price)
            equity += open_trade.pnl()
            trades.append(open_trade)
            eq.iloc[-1] = equity

        metrics = compute_metrics(eq)

        # Trade stats
        trade_pnls = np.array([t.pnl() for t in trades]) if trades else np.array([])
        wins = (trade_pnls > 0).sum()
        losses = (trade_pnls < 0).sum()
        avg_win = float(trade_pnls[trade_pnls > 0].mean()) if wins > 0 else 0.0
        avg_loss = float(trade_pnls[trade_pnls < 0].mean()) if losses > 0 else 0.0
        total_pnl = float(trade_pnls.sum()) if trades else 0.0

        summary = {
            **metrics,
            "InitialEquity": float(self.cfg.get("global", {}).get("starting_capital", 10_000)),
            "FinalEquity": float(eq.iloc[-1]),
            "TotalPnL": total_pnl,
            "AvgWin": avg_win,
            "AvgLoss": avg_loss,
            "WinningTrades": int(wins),
            "LosingTrades": int(losses),
            "NumTrades": int(len(trades)),
        }

        # Save outputs
        metrics_path = self.bt.get("metrics_path", "metrics/")
        os.makedirs(metrics_path, exist_ok=True)
        save_metrics_csv(summary, os.path.join(metrics_path, f"backtest_{symbol.replace('/', '_')}_{timeframe}.csv"))
        # Save equity curve
        eq.to_csv(os.path.join(metrics_path, f"equity_{symbol.replace('/', '_')}_{timeframe}.csv"), header=["equity"])

        # Optional HTML report (simple)
        try:
            import matplotlib.pyplot as plt
            os.makedirs("html_reports", exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(eq.index, eq.values)
            ax.set_title(f"Equity Curve {symbol} {timeframe}")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join("html_reports", f"equity_{symbol.replace('/', '_')}_{timeframe}.png"))
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to generate plot: {e}")

        # Pretty print JSON summary to logs
        logger.info(json.dumps(summary))
        return summary
