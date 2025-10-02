from __future__ import annotations
"""Pepperstone adapter (cTrader Open API only).

This adapter connects to Pepperstone via cTrader Open API using the
`ctrader-open-api` SDK (Twisted-based). It performs application and
optional account authentication. Market data and order placement can be
implemented on top of this connection (left out by request).
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from loguru import logger

import threading
import time
from datetime import datetime, timezone

# Optional cTrader Open API SDK
try:  # pragma: no cover - optional at runtime
    from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf  # type: ignore
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *  # type: ignore
    from ctrader_open_api.messages.OpenApiMessages_pb2 import *  # type: ignore
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *  # type: ignore
    from twisted.internet import reactor  # type: ignore
    _CTRADER_AVAILABLE = True
except Exception:  # pragma: no cover
    _CTRADER_AVAILABLE = False


class PepperstoneAdapter:
    def __init__(self, config: Dict[str, Any], dry_run: bool = True) -> None:
        self.cfg = config
        self.dry_run = dry_run
        self.mode = "ctrader"
        self.connected = False
        # cTrader config
        ctrader_cfg = (config.get("ctrader", {}) or {})
        self.ctrader_token = ctrader_cfg.get("access_token", "")  # OAuth token for account
        self.ctrader_account_id = ctrader_cfg.get("account_id")
        self.ctrader_environment = ctrader_cfg.get("environment", "demo")  # demo | live
        self.ctrader_client_id = ctrader_cfg.get("client_id", "")
        self.ctrader_client_secret = ctrader_cfg.get("client_secret", "")

        # SDK state
        self._ct_client: "Client | None" = None
        self._reactor_thread: "threading.Thread | None" = None
        self._ct_connected_evt = threading.Event()
        self._symbols_cache: dict[str, int] = {}

    def connect(self) -> None:
        if not _CTRADER_AVAILABLE:
            raise RuntimeError("ctrader-open-api SDK not installed. Add it to requirements and install.")
        if not self.ctrader_client_id or not self.ctrader_client_secret:
            raise RuntimeError("cTrader requires client_id and client_secret in config.exchanges.pepperstone.ctrader")

        host = EndPoints.PROTOBUF_LIVE_HOST if self.ctrader_environment == "live" else EndPoints.PROTOBUF_DEMO_HOST
        port = EndPoints.PROTOBUF_PORT

        self._ct_client = Client(host, port, TcpProtocol)

        def _on_connected(client: Client) -> None:
            logger.info("Connected to cTrader Open API host, authenticating application...")
            req = ProtoOAApplicationAuthReq()
            req.clientId = self.ctrader_client_id
            req.clientSecret = self.ctrader_client_secret
            d = client.send(req)

            def _app_authed(_res: Any) -> None:
                logger.info("Application authenticated. Authenticating account...")
                if not self.ctrader_account_id or not self.ctrader_token:
                    logger.warning("Account auth skipped: account_id or access_token missing. Read-only connection.")
                    self._ct_connected_evt.set()
                    return
                acc_req = ProtoOAAccountAuthReq()
                acc_req.ctidTraderAccountId = int(self.ctrader_account_id)
                acc_req.accessToken = self.ctrader_token
                d2 = client.send(acc_req)

                def _acc_ok(_res2: Any) -> None:
                    logger.info("cTrader account authenticated")
                    self._ct_connected_evt.set()

                def _acc_err(f) -> None:
                    logger.error(f"Account auth failed: {f}")
                    self._ct_connected_evt.set()

                d2.addCallbacks(_acc_ok, _acc_err)

            def _app_err(f) -> None:
                logger.error(f"Application auth failed: {f}")
                self._ct_connected_evt.set()

            d.addCallbacks(_app_authed, _app_err)

        def _on_disconnected(_client: Client, reason: Any) -> None:
            logger.warning(f"Disconnected from cTrader: {reason}")

        self._ct_client.setConnectedCallback(_on_connected)
        self._ct_client.setDisconnectedCallback(_on_disconnected)
        self._ct_client.startService()

        if self._reactor_thread is None or not self._reactor_thread.is_alive():
            def _run_reactor() -> None:
                try:
                    # Run reactor in background; don't install signal handlers in threads
                    from twisted.internet import reactor as _reactor  # type: ignore
                    _reactor.run(installSignalHandlers=False)
                except Exception as e:  # pragma: no cover
                    logger.error(f"Twisted reactor error: {e}")
            self._reactor_thread = threading.Thread(target=_run_reactor, name="ctrader-reactor", daemon=True)
            self._reactor_thread.start()

        # Wait a short time for authentication
        self._ct_connected_evt.wait(timeout=10)
        if not self._ct_connected_evt.is_set():
            logger.warning("cTrader authentication did not complete within timeout; continuing")
        logger.info("Connected to Pepperstone via cTrader Open API")
        self.connected = True

    def disconnect(self) -> None:
        if self._ct_client is not None:
            try:
                self._ct_client.stopService()
            except Exception:
                pass
        self.connected = False
        logger.info("Disconnected from Pepperstone")

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        # Convenience method: fetch recent range based on limit
        end = datetime.now(timezone.utc)
        # Approximate start using limit * timeframe
        tf_minutes = self._timeframe_to_minutes(timeframe)
        start = end - pd.Timedelta(minutes=tf_minutes * max(1, limit))
        return self.get_ohlcv_range(symbol, timeframe, start, end)

    def get_ohlcv_range(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        if not _CTRADER_AVAILABLE:
            raise RuntimeError("ctrader-open-api SDK not installed. Add it to requirements and install.")
        if not self.connected or self._ct_client is None:
            raise RuntimeError("PepperstoneAdapter is not connected. Call connect() first.")

        symbol_id = self._lookup_symbol_id(symbol)
        period = self._map_timeframe_to_period(timeframe)
        if period is None:
            raise ValueError(f"Unsupported timeframe for cTrader: {timeframe}")

        # Normalize times to epoch ms UTC
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        from_ms = int(start.timestamp() * 1000)
        to_ms = int(end.timestamp() * 1000)

        rows: List[list[float]] = []

        # cTrader limits per request; we request in chunks if needed
        max_bars = 1000
        fetch_from = from_ms

        while fetch_from < to_ms:
            fetch_to = to_ms
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = int(self.ctrader_account_id) if self.ctrader_account_id else 0
            req.symbolId = int(symbol_id)
            req.period = period
            req.fromTimestamp = fetch_from
            req.toTimestamp = fetch_to

            evt = threading.Event()
            result_container: dict[str, Any] = {}

            def _ok(res: Any) -> None:
                result_container['ok'] = res
                evt.set()

            def _err(err: Any) -> None:
                result_container['err'] = err
                evt.set()

            d = self._ct_client.send(req)
            d.addCallbacks(_ok, _err)

            evt.wait(timeout=10)
            if 'err' in result_container:
                raise RuntimeError(f"Trendbars request failed: {result_container['err']}")
            if 'ok' not in result_container:
                logger.warning("Trendbars request timed out; breaking")
                break

            res = result_container['ok']
            # res contains trendbar list: trendbar element has period, low, high, open, close, volume, utcTimestampInMinutes (?)
            # We normalize using available fields; typical field is 'utcTimestampInMinutes' or 'timestamp'
            bars = getattr(res, 'trendbar', [])
            if not bars:
                break

            last_ts = None
            for b in bars:
                # Try common field names defensively
                ts_field = getattr(b, 'utcTimestampInMinutes', None)
                if ts_field is None:
                    ts_field = getattr(b, 'timestamp', None)
                if ts_field is None:
                    continue
                ts_ms = int(ts_field)
                o = float(getattr(b, 'open', 0.0))
                h = float(getattr(b, 'high', 0.0))
                l = float(getattr(b, 'low', 0.0))
                c = float(getattr(b, 'close', 0.0))
                v = float(getattr(b, 'volume', 0.0))
                rows.append([ts_ms, o, h, l, c, v])
                last_ts = ts_ms

            if last_ts is None or last_ts <= fetch_from:
                break
            fetch_from = last_ts + 1

            # Respect rate limiting a bit
            time.sleep(0.2)

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore
        df = df[(df["timestamp"] >= from_ms) & (df["timestamp"] <= to_ms)]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep='last')].sort_index()
        return df

    # ---------- helpers ----------
    def _timeframe_to_minutes(self, tf: str) -> int:
        m = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240,
            '1d': 1440,
        }
        return m.get(tf, 60)

    def _map_timeframe_to_period(self, tf: str):
        # Map to ProtoOATrendbarPeriod enum
        try:
            mapping = {
                '1m': ProtoOATrendbarPeriod.M1,
                '5m': ProtoOATrendbarPeriod.M5,
                '15m': ProtoOATrendbarPeriod.M15,
                '30m': ProtoOATrendbarPeriod.M30,
                '1h': ProtoOATrendbarPeriod.H1,
                '4h': ProtoOATrendbarPeriod.H4,
                '1d': ProtoOATrendbarPeriod.D1,
            }
            return mapping.get(tf)
        except Exception:
            return None

    def _lookup_symbol_id(self, symbol: str) -> int:
        # Cached first
        if symbol in self._symbols_cache:
            return self._symbols_cache[symbol]
        if self._ct_client is None:
            raise RuntimeError("cTrader client not initialized")

        # Request symbols list and find matching symbol name
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = int(self.ctrader_account_id) if self.ctrader_account_id else 0
        evt = threading.Event()
        result_container: dict[str, Any] = {}

        def _ok(res: Any) -> None:
            result_container['ok'] = res
            evt.set()

        def _err(err: Any) -> None:
            result_container['err'] = err
            evt.set()

        d = self._ct_client.send(req)
        d.addCallbacks(_ok, _err)
        evt.wait(timeout=10)
        if 'err' in result_container:
            raise RuntimeError(f"Symbols list failed: {result_container['err']}")
        if 'ok' not in result_container:
            raise RuntimeError("Symbols list timed out")

        res = result_container['ok']
        symbols = getattr(res, 'symbol', [])
        # Try exact match and common name match
        wanted = symbol.replace('-', '/').upper()
        for s in symbols:
            name = str(getattr(s, 'symbolName', '')).upper()
            if name == wanted or name.replace('-', '/').upper() == wanted:
                sid = int(getattr(s, 'symbolId'))
                self._symbols_cache[symbol] = sid
                return sid
        raise ValueError(f"Symbol not found on cTrader: {symbol}")

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float | None = None, **kwargs: Any) -> Dict[str, Any]:
        logger.info(f"Pepperstone place_order dry_run={self.dry_run} {side} {quantity} {symbol}")
        if self.dry_run:
            return {"id": "dryrun", "status": "filled", "symbol": symbol, "side": side, "qty": quantity}
        raise NotImplementedError("cTrader order placement via Open API requires symbolId and order request building. Provide details or let me implement it next.")

    def modify_order(self, order_id: str, **kwargs: Any) -> Dict[str, Any]:
        logger.warning("modify_order placeholder (MT5 modify by position/order) — not implemented here")
        return {"id": order_id, "status": "modified"}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        logger.warning("cancel_order placeholder — not implemented here")
        return {"id": order_id, "status": "canceled"}

    def get_account(self) -> Dict[str, Any]:
        # Simplified
        return {"balance": 10000.0, "equity": 10000.0}

    def get_positions(self) -> Dict[str, Any]:
        return {}
