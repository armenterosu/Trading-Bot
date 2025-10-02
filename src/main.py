"""Main entrypoint for trading bot.

Usage:
  python src/main.py --mode backtest --config config.yaml --exchange binance --symbol BTC/USDT --timeframe 1h
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Any

from app.utils.config_loader import load_config
from app.utils.logging_config import setup_logging
from app.utils.kill_switch import DrawdownKillSwitch
from app.core.engine import TradingEngine
from threading import Thread
from app.utils.metrics_server import create_app
from app.backtester.backtrader_engine import BacktraderBacktester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-market Trading Bot")
    parser.add_argument("--mode", choices=["backtest", "paper", "live"], default="backtest")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--exchange", choices=["binance", "pepperstone", "kaggle"], default="binance")
    parser.add_argument("--symbol", required=False, default="BTC/USDT")
    parser.add_argument("--timeframe", required=False, default="1h")
    parser.add_argument("--dry-run", dest="dry_run", default="true")
    parser.add_argument("--metrics", dest="metrics", default="false")
    parser.add_argument("--strategies", dest="strategies", default="", help="Comma-separated strategy names to backtest (e.g. ema_cross,bollinger). Empty = all.")
    return parser.parse_args()


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def create_engine(cfg: Dict[str, Any], args: argparse.Namespace) -> TradingEngine:
    return TradingEngine(
        config=cfg,
        exchange_name=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        dry_run=str2bool(args.dry_run),
        metrics=str2bool(args.metrics),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    os.makedirs("logs", exist_ok=True)
    os.makedirs(cfg.get("backtest", {}).get("metrics_path", "metrics"), exist_ok=True)

    setup_logging(cfg["logging"])  # structured, rotating

    kill_switch = DrawdownKillSwitch(limit=cfg["risk"]["max_drawdown_limit"])

    # Optional metrics server
    if str2bool(args.metrics):
        metrics_path = cfg.get("backtest", {}).get("metrics_path", "metrics/")
        app = create_app(metrics_path)
        t = Thread(target=lambda: app.run(host="0.0.0.0", port=8000), daemon=True)
        t.start()

    if args.mode == "backtest":
        backtester = BacktraderBacktester(config=cfg)
        # Parse strategies list (optional)
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()] or None
        report = backtester.run(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            strategies=strategies,
        )
        print("Backtest Summary:")
        # If multiple strategies, print per-strategy compact summaries
        if isinstance(report, dict):
            for strat, summary in report.items():
                final_eq = summary.get("FinalEquity")
                cagr = summary.get("CAGR")
                sharpe = summary.get("Sharpe")
                print(f" - {strat}: FinalEquity={final_eq} CAGR={cagr} Sharpe={sharpe}")
        else:
            for k, v in report.items():
                print(f" - {k}: {v}")
        return

    engine = create_engine(cfg, args)
    engine.register_kill_switch(kill_switch)
    engine.run(mode=args.mode)


if __name__ == "__main__":
    main()
