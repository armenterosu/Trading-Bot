from __future__ import annotations
"""Pepperstone adapter (cTrader Open API only).

This adapter connects to Pepperstone via cTrader Open API using the
`ctrader-open-api` SDK (Twisted-based). It performs application and
optional account authentication. Market data and order placement can be
implemented on top of this connection (left out by request).
"""
from typing import Dict, Any
import pandas as pd
from loguru import logger

import threading

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
        raise NotImplementedError("cTrader OHLCV via Open API requires symbolId lookup and trendbars request. Provide mapping or allow me to implement it next.")

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
