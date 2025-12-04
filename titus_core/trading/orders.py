"""Order and fill primitives for Titus."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def direction(self) -> int:
        return 1 if self is OrderSide.BUY else -1


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELED = "canceled"


class ExecutionTiming(str, Enum):
    NEXT_BAR_OPEN = "next_bar_open"
    CURRENT_BAR_CLOSE = "current_bar_close"


@dataclass
class OrderRequest:
    """Captured intent from strategy code."""

    order_id: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    reduce_only: bool = False
    comment: Optional[str] = None
    timing: ExecutionTiming = ExecutionTiming.NEXT_BAR_OPEN


@dataclass
class Order:
    """Concrete order tracked by the engine."""

    request: OrderRequest
    created_bar_index: int
    status: OrderStatus = OrderStatus.PENDING
    last_activation_price: Optional[float] = None
    reference_price: Optional[float] = None

    def activate(self, reference_price: float) -> None:
        self.status = OrderStatus.ACTIVE
        self.last_activation_price = reference_price


@dataclass
class Fill:
    """Result of executing an order."""

    order_id: str
    bar_index: int
    price: float
    quantity: float
    side: OrderSide
    timestamp: str
    commission: float = 0.0
    reduce_only: bool = False
    comment: Optional[str] = None


@dataclass
class Trade:
    """Closed trade summary."""

    entry_id: str
    exit_id: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    quantity: float
    direction: str
    pnl: float
    entry_bar: int
    exit_bar: int
    metadata: dict = field(default_factory=dict)
