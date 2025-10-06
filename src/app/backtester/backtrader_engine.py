from __future__ import annotations
"""Backtrader-based backtesting engine.

Runs each configured strategy independently over an online-fetched OHLCV
DataFrame by wrapping strategy signals into a Backtrader Strategy.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from loguru import logger
import backtrader as bt
import importlib
import pkgutil
import os

from app.utils.metrics import compute_metrics, save_metrics_csv, generate_metrics_images

import os as _os
import ccxt
from pathlib import Path
try:
    # As requested by user: simple Client API
    from ctrader_open_api import Client as CTraderClient  # type: ignore
except Exception:  # pragma: no cover
    CTraderClient = None  # type: ignore

try:
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
except Exception:  # pragma: no cover
    KaggleApi = None  # type: ignore


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _get_nested(d: Dict[str, Any], *keys: Any) -> Any:
    """Safely navigate nested dicts from Backtrader analyzers.

    Handles cases where TradeAnalyzer may have slightly different nesting per version.
    """
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _num_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _ccxt_binance_df(cfg: Dict[str, Any], symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    bx_cfg = (cfg.get("exchanges", {}).get("binance", {}) or {})
    params = {
        'apiKey': bx_cfg.get('api_key') or _os.environ.get('BINANCE_API_KEY'),
        'secret': bx_cfg.get('api_secret') or _os.environ.get('BINANCE_API_SECRET'),
        'enableRateLimit': True,
        'options': {'defaultType': bx_cfg.get('default_market', 'spot')},
    }
    client = ccxt.binance(params)
    if bx_cfg.get('testnet', True):
        client.set_sandbox_mode(True)
    data = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def _ctrader_df(cfg: Dict[str, Any], symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    if CTraderClient is None:
        raise RuntimeError("ctrader-open-api not available")
    cx = (cfg.get("exchanges", {}).get("pepperstone", {}) or {})
    c = (cx.get("ctrader", {}) or {})
    client_id = c.get("client_id") or _os.environ.get("CTRADER_CLIENT_ID")
    client_secret = c.get("client_secret") or _os.environ.get("CTRADER_CLIENT_SECRET")
    access_token = c.get("access_token") or _os.environ.get("CTRADER_ACCESS_TOKEN")
    if not client_id or not client_secret or not access_token:
        raise RuntimeError("Missing cTrader credentials (client_id/client_secret/access_token)")

    client = CTraderClient(client_id, client_secret, access_token)  # type: ignore

    # Map timeframe like "1h" -> "H1", "15m" -> "M15"
    tf_map = {
        '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
        '1h': 'H1', '4h': 'H4', '1d': 'D1',
    }
    ct_tf = tf_map.get(timeframe)
    if not ct_tf:
        raise ValueError(f"Unsupported cTrader timeframe: {timeframe}")

    # Symbols usually without slash, e.g., EURUSD
    ct_symbol = symbol.replace('/', '')

    candles = client.get_candles(symbol=ct_symbol, timeframe=ct_tf, count=int(count))  # type: ignore
    # Normalize candles into DataFrame
    if isinstance(candles, list) and candles:
        if isinstance(candles[0], dict):
            # Expect keys like time/open/high/low/close/volume (be permissive)
            def _get(item, *keys, default=None):
                for k in keys:
                    if k in item:
                        return item[k]
                return default
            rows = []
            for c in candles:
                t = _get(c, 'time', 'timestamp')
                o = float(_get(c, 'open', 'o', default=0.0))
                h = float(_get(c, 'high', 'h', default=0.0))
                l = float(_get(c, 'low', 'l', default=0.0))
                cl = float(_get(c, 'close', 'c', default=0.0))
                v = float(_get(c, 'volume', 'v', default=0.0))
                rows.append([t, o, h, l, cl, v])
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
        else:
            # Assume list format [ts, o, h, l, c, v]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
    else:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    # Convert timestamp to datetime
    if not df.empty:
        # Try ms, then seconds
        ts = pd.to_datetime(df["timestamp"], unit="ms", errors='ignore')
        if not isinstance(ts.iloc[0], pd.Timestamp):
            ts = pd.to_datetime(df["timestamp"], unit="s", errors='coerce')
        df["timestamp"] = pd.to_datetime(ts)
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_online_df(cfg: Dict[str, Any], exchange: str, symbol: str, timeframe: str,
                     start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # Per user request: use direct libraries with limit/count
    limit = int(_os.environ.get('BT_LIMIT') or cfg.get('backtest', {}).get('limit', 500))
    if exchange == 'binance':
        return _ccxt_binance_df(cfg, symbol, timeframe, limit)
    if exchange == 'pepperstone':
        return _ctrader_df(cfg, symbol, timeframe, limit)
    if exchange == 'kaggle':
        return _kaggle_df(cfg, symbol, timeframe, start, end)
    raise ValueError(f"Unknown exchange: {exchange}")


def _discover_strategy_names() -> List[str]:
    strategies_dir = __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), "..", "strategies"))
    names: List[str] = []
    for _, modname, ispkg in pkgutil.iter_modules([strategies_dir]):
        if ispkg:
            continue
        if modname in {"base_strategy", "__init__"}:
            continue
        names.append(modname)
    return sorted(names)


def _load_strategies(cfg: Dict[str, Any], strategy_names: Optional[List[str]]) -> List[Any]:
    strategies_cfg = cfg.get("strategies", {})
    names = strategy_names if strategy_names is not None else _discover_strategy_names()
    loaded: List[Any] = []
    for name in names:
        module_name = f"app.strategies.{name}"
        class_name = ''.join(part.capitalize() for part in name.split('_')) + "Strategy"
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            params = strategies_cfg.get(name, {})
            loaded.append(cls(params))
            logger.info(f"Loaded strategy {name}")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to load strategy {name}: {e}")
    return loaded


class _SignalWrapperStrategy(bt.Strategy):
    params = dict(
        signals_map=None,  # dict[datetime -> int]
        risk_per_trade=0.01,
    )

    def __init__(self):
        self.order = None

    def next(self):
        dt = self.data.datetime.datetime(0)
        sig = self.p.signals_map.get(dt, 0)
        if self.position.size == 0:
            if sig != 0:
                cash = self.broker.getcash()
                price = float(self.data.close[0])
                # position sizing: invest fraction of cash
                qty = max((cash * self.p.risk_per_trade) / max(price, 1e-9), 0.0)
                if qty <= 0:
                    return
                if sig > 0:
                    self.order = self.buy(size=qty)
                else:
                    self.order = self.sell(size=qty)
        else:
            # Close on opposite signal or neutral signal
            if sig == 0 or (self.position.size > 0 and sig < 0) or (self.position.size < 0 and sig > 0):
                self.close()
    
    def stop(self):
        # Close any open positions at the end of backtest
        if self.position.size != 0:
            self.close()


class _EquityCurveAnalyzer(bt.Analyzer):
    def __init__(self):
        self.values: List[Tuple[pd.Timestamp, float]] = []

    def next(self):
        dt = pd.to_datetime(self.strategy.data.datetime.datetime(0))
        val = float(self.strategy.broker.getvalue())
        self.values.append((dt, val))

    def get_analysis(self):
        return self.values


class BacktraderBacktester:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.bt_cfg = config.get("backtest", {})
        self.risk_cfg = config.get("risk", {})
        self._risk_per_trade = _as_float(self.risk_cfg.get("max_risk_per_trade", 0.01), 0.01)

    def run(self, symbol: str, exchange: str, timeframe: str, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        # Fetch online data
        start = pd.to_datetime(self.bt_cfg.get("start_date")) if self.bt_cfg.get("start_date") else pd.Timestamp.utcnow() - pd.Timedelta(days=30)
        end = pd.to_datetime(self.bt_cfg.get("end_date")) if self.bt_cfg.get("end_date") else pd.Timestamp.utcnow()
        df = _fetch_online_df(self.cfg, exchange, symbol, timeframe, start, end)
        if df is None or df.empty:
            raise ValueError("Failed to fetch online OHLCV for Backtrader")
        df = df.sort_index()

        # Backtrader feed
        data = bt.feeds.PandasData(dataname=df)

        results: Dict[str, Any] = {}
        for strat in _load_strategies(self.cfg, strategies):
            strat_name = strat.__class__.__name__
            logger.info(f"Backtrader run for strategy: {strat_name}")
            sig_df = strat.generate_signals(df)
            if 'signal' not in sig_df.columns:
                logger.error(f"Strategy {strat_name} did not return 'signal' column; skipping")
                continue
            # Map datetime -> signal (ensure index is datetime)
            sig_df = sig_df.copy()
            sig_df.index = pd.to_datetime(sig_df.index)
            signals_map = {ts.to_pydatetime(): int(s) for ts, s in sig_df['signal'].fillna(0).items()}

            cerebro = bt.Cerebro()
            cerebro.broker.setcash(_as_float(self.cfg.get("global", {}).get("starting_capital", 10_000), 10_000.0))
            commission = _as_float(self.bt_cfg.get("commission", 0.0), 0.0)
            cerebro.broker.setcommission(commission=commission)
            cerebro.adddata(data)

            S = _SignalWrapperStrategy
            cerebro.addstrategy(S, signals_map=signals_map, risk_per_trade=self._risk_per_trade)

            # Analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True)
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(_EquityCurveAnalyzer, _name='equity')

            run = cerebro.run()
            pnl_final = cerebro.broker.getvalue()
            analyzers = run[0].analyzers

            # Extract metrics safely
            sharpe = None
            try:
                sharpe = analyzers.sharpe.get_analysis().get('sharperatio')
            except Exception:
                pass
            dd = analyzers.dd.get_analysis() if hasattr(analyzers, 'dd') else {}
            maxdd = dd.get('max', {}).get('drawdown') if isinstance(dd.get('max'), dict) else dd.get('maxdrawdown')
            returns = analyzers.returns.get_analysis() if hasattr(analyzers, 'returns') else {}
            cagr = returns.get('rnorm100')  # normalized annual return (%)
            trades = analyzers.trades.get_analysis() if hasattr(analyzers, 'trades') else {}
            
            # Debug: log raw trades analyzer output
            logger.debug(f"TradeAnalyzer raw output: {trades}")
            
            # Trade stats (robust to analyzer structure differences)
            # Backtrader TradeAnalyzer structure:
            # trades = {
            #   'total': {'total': N, 'open': X, 'closed': Y},
            #   'won': {'total': W, 'pnl': {...}},
            #   'lost': {'total': L, 'pnl': {...}},
            #   'long': {'total': LT, 'won': LW, 'lost': LL, 'pnl': {...}},
            #   'short': {'total': ST, 'won': SW, 'lost': SL, 'pnl': {...}}
            # }
            
            total_trades = _get_nested(trades, 'total', 'total')
            if total_trades is None:
                # Fallback: if 'total' is just a number
                total_val = trades.get('total')
                total_trades = total_val if isinstance(total_val, (int, float)) else 0
            
            total_closed = _get_nested(trades, 'total', 'closed') or 0
            wins_total = _get_nested(trades, 'won', 'total') or 0
            losses_total = _get_nested(trades, 'lost', 'total') or 0

            long_total = _get_nested(trades, 'long', 'total') or 0
            short_total = _get_nested(trades, 'short', 'total') or 0
            
            # Wins/losses by direction
            long_won = _get_nested(trades, 'long', 'won') or 0
            short_won = _get_nested(trades, 'short', 'won') or 0
            long_lost = _get_nested(trades, 'long', 'lost') or 0
            short_lost = _get_nested(trades, 'short', 'lost') or 0

            # Directional win rates
            wr_long = None
            if _num_or_none(long_total) and float(long_total) > 0 and _num_or_none(long_won) is not None:
                wr_long = float(long_won) / float(long_total)
            wr_short = None
            if _num_or_none(short_total) and float(short_total) > 0 and _num_or_none(short_won) is not None:
                wr_short = float(short_won) / float(short_total)

            # Build equity curve and compute professional metrics
            eq_values = analyzers.equity.get_analysis() if hasattr(analyzers, 'equity') else []
            eq_series = pd.Series({ts: val for ts, val in eq_values}).sort_index() if eq_values else pd.Series(dtype=float)
            prof_metrics = compute_metrics(eq_series) if not eq_series.empty else {}

            summary = {
                "Strategy": strat_name,
                "FinalEquity": float(pnl_final),
                # Prefer our metrics util if available; fallback to analyzers otherwise
                "CAGR": prof_metrics.get("CAGR", cagr),
                "Sharpe": prof_metrics.get("Sharpe", sharpe),
                "MaxDD": prof_metrics.get("MaxDD", maxdd),
                "DD_Duration": prof_metrics.get("DD_Duration"),
                "Calmar": prof_metrics.get("Calmar"),
                "WinRate": prof_metrics.get("WinRate"),
                "ProfitFactor": prof_metrics.get("ProfitFactor"),
                "Expectancy": prof_metrics.get("Expectancy"),
                "NumTrades": prof_metrics.get("Trades", total_trades),
                # Extended trade breakdown
                "Trades_Closed": total_closed,
                "Trades_Long": long_total,
                "Trades_Short": short_total,
                "Wins_Total": wins_total,
                "Losses_Total": losses_total,
                "Wins_Long": long_won,
                "Losses_Long": long_lost,
                "Wins_Short": short_won,
                "Losses_Short": short_lost,
                "WinRate_Long": wr_long,
                "WinRate_Short": wr_short,
            }
            
            # Formatted trade statistics log
            logger.info(f"=== Backtest Results: {strat_name} ===")
            logger.info(f"Final Equity: {pnl_final:.2f} | CAGR: {summary.get('CAGR', 0)*100:.2f}% | Sharpe: {summary.get('Sharpe', 0):.2f} | MaxDD: {summary.get('MaxDD', 0)*100:.2f}%")
            logger.info(f"Total Trades: {total_trades} | Closed: {total_closed} | Wins: {wins_total} | Losses: {losses_total}")
            
            # Long trades
            long_wr_pct = f"{wr_long*100:.1f}%" if wr_long is not None else "N/A"
            logger.info(f"Long Trades: {long_total} | Wins: {long_won} | Losses: {long_lost} | WinRate: {long_wr_pct}")
            
            # Short trades
            short_wr_pct = f"{wr_short*100:.1f}%" if wr_short is not None else "N/A"
            logger.info(f"Short Trades: {short_total} | Wins: {short_won} | Losses: {short_lost} | WinRate: {short_wr_pct}")
            
            logger.info(f"Full metrics: {summary}")
            results[strat_name] = summary

            # Persist outputs
            metrics_path = self.bt_cfg.get("metrics_path", "metrics/")
            os.makedirs(metrics_path, exist_ok=True)
            base_name = f"{symbol.replace('/', '_')}_{timeframe}_{strat_name}_bt"
            if not eq_series.empty:
                eq_series.to_csv(os.path.join(metrics_path, f"equity_{base_name}.csv"), header=["equity"])
            save_metrics_csv({k: (v if v is not None else "") for k, v in summary.items()}, os.path.join(metrics_path, f"backtest_{base_name}.csv"))

        # Generate visual metric images after all strategies complete
        logger.info("Generating metric visualization images...")
        generate_metrics_images(metrics_path)
        
        return results
