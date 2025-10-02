from __future__ import annotations
"""Binance adapter via ccxt."""
from typing import Dict, Any
import pandas as pd
import ccxt
from loguru import logger
import time
from datetime import datetime, timezone


class BinanceAdapter:
    def __init__(self, config: Dict[str, Any], dry_run: bool = True) -> None:
        self.cfg = config
        self.dry_run = dry_run
        self.client: ccxt.binance | None = None

    def connect(self) -> None:
        params = {
            'apiKey': self.cfg.get('api_key'),
            'secret': self.cfg.get('api_secret'),
            'enableRateLimit': True,
            'options': {'defaultType': self.cfg.get('default_market', 'spot')},
        }
        self.client = ccxt.binance(params)
        if self.cfg.get('testnet', True):
            self.client.set_sandbox_mode(True)
        logger.info("Connected to Binance (ccxt)")

    def disconnect(self) -> None:
        logger.info("Disconnected from Binance")

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        assert self.client is not None
        data = self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_ohlcv_range(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV for a date range by paginating ccxt fetch_ohlcv.

        Args:
            symbol: e.g., "BTC/USDT"
            timeframe: e.g., "1h", "15m"
            start: timezone-aware or naive UTC datetime
            end: timezone-aware or naive UTC datetime
        """
        assert self.client is not None
        # Normalize to UTC ms
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_rows: list[list[float]] = []
        last_ts = since_ms
        limit = 1000  # max per call for many timeframes
        while last_ts < end_ms:
            batch = self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=last_ts, limit=limit)
            if not batch:
                break
            all_rows.extend(batch)
            # ccxt returns candles up to now; avoid infinite loops
            new_last = batch[-1][0]
            if new_last <= last_ts:
                break
            last_ts = new_last + 1
            # Rate limit safety
            time.sleep(self.client.rateLimit / 1000 if hasattr(self.client, 'rateLimit') else 0.2)

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])  # empty

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
        df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] <= end_ms)]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df.set_index("timestamp", inplace=True)
        # Deduplicate and sort
        df = df[~df.index.duplicated(keep='last')].sort_index()
        return df

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float | None = None, **kwargs: Any) -> Dict[str, Any]:
        logger.info(f"place_order dry_run={self.dry_run} {side} {quantity} {symbol}")
        if self.dry_run:
            return {"id": "dryrun", "status": "filled", "symbol": symbol, "side": side, "qty": quantity}
        assert self.client is not None
        if order_type == "market":
            order = self.client.create_order(symbol, 'market', side, quantity)
        else:
            assert price is not None
            order = self.client.create_order(symbol, 'limit', side, quantity, price)
        return order  # type: ignore

    def modify_order(self, order_id: str, **kwargs: Any) -> Dict[str, Any]:
        logger.warning("modify_order not implemented for Binance via ccxt (placeholder)")
        return {"id": order_id, "status": "modified"}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        logger.warning("cancel_order not implemented in example (needs symbol)")
        return {"id": order_id, "status": "canceled"}

    def get_account(self) -> Dict[str, Any]:
        if self.dry_run:
            return {"balance": 10000.0, "equity": 10000.0}
        assert self.client is not None
        bal = self.client.fetch_balance()
        total = bal.get('total', {})
        equity = sum(total.values()) if isinstance(total, dict) else 0.0
        return {"balance": equity, "equity": equity}

    def get_positions(self) -> Dict[str, Any]:
        return {}
