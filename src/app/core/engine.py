from __future__ import annotations
"""Runtime engine: wires adapters, strategies, and executes loops for paper/live."""
from typing import Dict, Any, List
import time
import importlib
from loguru import logger

from app.exchanges.base_adapter import ExchangeAdapter
from app.exchanges.binance_adapter import BinanceAdapter
from app.exchanges.pepperstone_adapter import PepperstoneAdapter
from app.strategies.base_strategy import BaseStrategy
from app.utils.kill_switch import DrawdownKillSwitch


class TradingEngine:
    def __init__(self, config: Dict[str, Any], exchange_name: str, symbol: str, timeframe: str, dry_run: bool, metrics: bool) -> None:
        self.config = config
        self.symbol = symbol
        self.timeframe = timeframe
        self.dry_run = dry_run
        self.metrics = metrics
        self.exchange: ExchangeAdapter = self._init_exchange(exchange_name)
        self.strategies: List[BaseStrategy] = self._load_strategies()
        self.kill_switch: DrawdownKillSwitch | None = None
        self.peak_equity = float(config["global"]["starting_capital"])
        self.current_equity = self.peak_equity

    def register_kill_switch(self, ks: DrawdownKillSwitch) -> None:
        self.kill_switch = ks

    def _init_exchange(self, name: str) -> ExchangeAdapter:
        if name == "binance":
            return BinanceAdapter(self.config["exchanges"]["binance"], dry_run=self.dry_run)
        if name == "pepperstone":
            return PepperstoneAdapter(self.config["exchanges"]["pepperstone"], dry_run=self.dry_run)
        raise ValueError(f"Unknown exchange: {name}")

    def _load_strategies(self) -> List[BaseStrategy]:
        strategies_cfg = self.config.get("strategies", {})
        enabled = strategies_cfg.get("enabled", [])
        loaded: List[BaseStrategy] = []
        for name in enabled:
            module_name = f"app.strategies.{name}"
            class_name = ''.join(part.capitalize() for part in name.split('_')) + "Strategy"
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                params = strategies_cfg.get(name, {})
                loaded.append(cls(params))
                logger.info(f"Loaded strategy {name}")
            except Exception as e:
                logger.error(f"Failed to load strategy {name}: {e}")
        return loaded

    def run(self, mode: str) -> None:
        logger.info(f"Starting engine mode={mode} symbol={self.symbol} tf={self.timeframe} dry_run={self.dry_run}")
        self.exchange.connect()
        try:
            while True:
                ohlcv = self.exchange.get_ohlcv(self.symbol, self.timeframe, limit=300)
                for strat in self.strategies:
                    signals = strat.generate_signals(ohlcv)
                    # Simple example: act on last signal
                    if not signals.empty and signals.iloc[-1]["signal"] != 0:
                        side = "buy" if signals.iloc[-1]["signal"] > 0 else "sell"
                        qty = strat.position_size(self.exchange.get_account(), self.config["risk"]["max_risk_per_trade"])
                        order = self.exchange.place_order(self.symbol, side=side, order_type="market", quantity=qty)
                        strat.on_order_filled(order)

                # Update equity (simplified)
                bal = self.exchange.get_account()
                self.current_equity = float(bal.get("equity", bal.get("balance", self.current_equity)))
                self.peak_equity = max(self.peak_equity, self.current_equity)
                if self.kill_switch and self.kill_switch.should_stop(self.peak_equity, self.current_equity):
                    logger.error("Kill-switch triggered due to drawdown. Stopping engine.")
                    break

                time.sleep(5)
        finally:
            self.exchange.disconnect()
