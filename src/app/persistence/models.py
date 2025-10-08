from __future__ import annotations
from typing import Optional
from sqlalchemy import Column, Integer, String, BigInteger, Numeric, Boolean, Text
from sqlalchemy.dialects.postgresql import JSONB
from .db import Base


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # ccxt trade fields (normalized)
    exchange_trade_id = Column(String(128), index=True)  # ccxt trade id
    order_id = Column(String(128), index=True)
    timestamp = Column(BigInteger, index=True)  # ms
    datetime = Column(String(64))
    symbol = Column(String(64), index=True)
    type = Column(String(32))  # limit/market
    side = Column(String(8))   # buy/sell
    taker_or_maker = Column(String(16))

    price = Column(Numeric(20, 10))
    amount = Column(Numeric(28, 10))
    cost = Column(Numeric(28, 10))

    fee_cost = Column(Numeric(28, 10))
    fee_currency = Column(String(32))

    raw = Column(JSONB)  # original exchange payload if available

    # Bot-specific fields
    mode = Column(String(16))  # backtest/paper/live
    strategy = Column(String(64))
    is_open = Column(Boolean, nullable=False, default=True, server_default="true")
    stop_loss = Column(Numeric(20, 10))
    take_profit = Column(Numeric(20, 10))
