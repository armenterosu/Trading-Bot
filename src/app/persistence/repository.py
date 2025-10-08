from __future__ import annotations
from typing import Any, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import select
from .db import get_engine, Base
from .models import Trade
from loguru import logger


class PgTradeStore:
    def __init__(self, database_url: str) -> None:
        self.engine = get_engine(database_url)
        # Ensure tables are created
        Base.metadata.create_all(self.engine)
        self._Session = Session(bind=self.engine, autoflush=False)

    def session(self) -> Session:
        # Provide a new session when needed (non-threadsafe simple impl)
        return Session(bind=self.engine, autoflush=False)

    def record_trade(
        self,
        *,
        ccxt_trade: Optional[Dict[str, Any]] = None,
        exchange_trade_id: Optional[str] = None,
        order_id: Optional[str] = None,
        timestamp: Optional[int] = None,
        datetime_str: Optional[str] = None,
        symbol: Optional[str] = None,
        type: Optional[str] = None,
        side: Optional[str] = None,
        taker_or_maker: Optional[str] = None,
        price: Optional[float] = None,
        amount: Optional[float] = None,
        qty: Optional[float] = None,
        cost: Optional[float] = None,
        fee_cost: Optional[float] = None,
        fee_currency: Optional[str] = None,
        raw: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        strategy: Optional[str] = None,
        is_open: bool = True,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> int:
        """Insert a trade; returns DB id.
        Accepts either a ccxt_trade dict or explicit fields.
        """
        data: Dict[str, Any] = {}
        if ccxt_trade is not None:
            t = ccxt_trade
            data = {
                "exchange_trade_id": str(t.get("id")) if t.get("id") is not None else None,
                "order_id": t.get("order"),
                "timestamp": t.get("timestamp"),
                "datetime": t.get("datetime"),
                "symbol": t.get("symbol"),
                "type": t.get("type"),
                "side": t.get("side"),
                "taker_or_maker": t.get("takerOrMaker"),
                "price": t.get("price"),
                "amount": t.get("amount"),
                "cost": t.get("cost"),
                "fee_cost": (t.get("fee") or {}).get("cost") if isinstance(t.get("fee"), dict) else None,
                "fee_currency": (t.get("fee") or {}).get("currency") if isinstance(t.get("fee"), dict) else None,
                "raw": t.get("info") or t,
            }
        else:
            data = {
                "exchange_trade_id": exchange_trade_id,
                "order_id": order_id,
                "timestamp": timestamp,
                "datetime": datetime_str,
                "symbol": symbol,
                "type": type,
                "side": side,
                "taker_or_maker": taker_or_maker,
                "price": price,
                "amount": amount if amount is not None else qty,
                "cost": cost,
                "fee_cost": fee_cost,
                "fee_currency": fee_currency,
                "raw": raw,
            }
        data.update({
            "mode": mode,
            "strategy": strategy,
            "is_open": is_open,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        })

        with self.session() as s:
            obj = Trade(**data)  # type: ignore[arg-type]
            s.add(obj)
            s.commit()
            s.refresh(obj)
            return int(obj.id)

    def update_sl_tp_by_order(self, *, order_id: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> None:
        with self.session() as s:
            q = s.execute(select(Trade).where(Trade.order_id == order_id))
            obj = q.scalar_one_or_none()
            if not obj:
                logger.warning(f"Trade not found for order_id={order_id}")
                return
            if stop_loss is not None:
                obj.stop_loss = stop_loss
            if take_profit is not None:
                obj.take_profit = take_profit
            s.commit()

    def close_trade_by_order(self, *, order_id: str, exit_price: Optional[float] = None, extra: Optional[str] = None) -> None:
        with self.session() as s:
            q = s.execute(select(Trade).where(Trade.order_id == order_id))
            obj = q.scalar_one_or_none()
            if not obj:
                logger.warning(f"Trade not found for order_id={order_id}")
                return
            obj.is_open = False
            if exit_price is not None:
                obj.price = exit_price
            if extra is not None:
                # Store extra info in raw under a custom key
                raw = dict(obj.raw or {})
                raw["close_extra"] = extra
                obj.raw = raw
            s.commit()

    def close(self) -> None:
        try:
            self.engine.dispose()
        except Exception:
            pass
