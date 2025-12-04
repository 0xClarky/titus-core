"""Base strategy definitions mirroring Pine Script lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional

from titus_core.indicators.registry import IndicatorRegistry
from titus_core.trading.orders import ExecutionTiming, OrderRequest, OrderSide, OrderType


@dataclass(frozen=True)
class PositionSnapshot:
    """Immutable view of the current portfolio position."""

    size: float = 0.0
    avg_price: float = 0.0


@dataclass
class StrategyContext:
    """Runtime context exposed to strategies each bar."""

    timestamp: datetime
    bar_index: int
    bar: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    features: Mapping[str, Any] = field(default_factory=dict)
    indicator_registry: Optional[IndicatorRegistry] = None
    order_dispatcher: Optional[Callable[[OrderRequest], None]] = field(default=None, repr=False)
    cancel_dispatcher: Optional[Callable[[str], None]] = field(default=None, repr=False)
    default_timing: ExecutionTiming = ExecutionTiming.NEXT_BAR_OPEN
    process_orders_on_close: bool = False
    default_qty_type: str = "fixed"
    default_qty_value: float = 1.0
    allow_long: bool = True
    allow_short: bool = True
    close_entries_rule: str = "none"
    position: PositionSnapshot = field(default_factory=PositionSnapshot)
    equity: float = 0.0

    def indicator(self, name: str, **params: Any) -> Any:
        """Fetch indicator output via shared registry."""
        if self.indicator_registry is None:
            raise RuntimeError("Indicator registry not attached to context.")
        return self.indicator_registry.compute(name, **params)

    def feature(self, key: str, default: Any = None) -> Any:
        """Return a precomputed feature value if present."""
        return self.features.get(key, default)

    @property
    def position_size(self) -> float:
        return self.position.size

    @property
    def position_avg_price(self) -> float:
        return self.position.avg_price

    def submit_order(
        self,
        *,
        order_id: str,
        side: OrderSide,
        quantity: float | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        timing: ExecutionTiming | None = None,
        reduce_only: bool = False,
        comment: str | None = None,
    ) -> None:
        if self.order_dispatcher is None:
            raise RuntimeError("Order dispatcher not attached to strategy context.")
        effective_timing = timing or self.default_timing
        if (
            effective_timing is ExecutionTiming.CURRENT_BAR_CLOSE
            and not self.process_orders_on_close
        ):
            effective_timing = ExecutionTiming.NEXT_BAR_OPEN
        resolved_qty, resolved_reduce_only = self._resolve_quantity_and_rule(
            quantity, side, reduce_only
        )
        request = OrderRequest(
            order_id=order_id,
            side=side,
            quantity=resolved_qty,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            reduce_only=resolved_reduce_only,
            comment=comment,
            timing=effective_timing,
        )
        self.order_dispatcher(request)

    def buy(
        self,
        *,
        order_id: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        timing: ExecutionTiming | None = None,
        comment: str | None = None,
    ) -> None:
        self.submit_order(
            order_id=order_id,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            timing=timing,
            comment=comment,
        )

    def sell(
        self,
        *,
        order_id: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        timing: ExecutionTiming | None = None,
        comment: str | None = None,
    ) -> None:
        self.submit_order(
            order_id=order_id,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            timing=timing,
            comment=comment,
        )

    def close_position(
        self,
        *,
        order_id: str,
        side: OrderSide,
        quantity: float | None = None,
        qty_percent: float | None = None,
        timing: ExecutionTiming | None = None,
        comment: str | None = None,
        order_type: OrderType = OrderType.MARKET,
        stop_price: float | None = None,
        limit_price: float | None = None,
    ) -> None:
        """Mimics Pine's strategy.close by issuing reduce-only orders."""
        resolved_qty = self._resolve_close_quantity(side, quantity, qty_percent)
        if resolved_qty <= 0:
            return
        self.submit_order(
            order_id=order_id,
            side=side,
            quantity=resolved_qty,
            order_type=order_type,
            stop_price=stop_price,
            limit_price=limit_price,
            timing=timing,
            comment=comment,
            reduce_only=True,
        )

    def cancel_order(self, order_id: str) -> None:
        if self.cancel_dispatcher is None:
            raise RuntimeError("Cancel dispatcher not attached to strategy context.")
        self.cancel_dispatcher(order_id)

    def _resolve_quantity_and_rule(
        self, quantity: float | None, side: OrderSide, reduce_only: bool
    ) -> tuple[float, bool]:
        if side is OrderSide.BUY and not self.allow_long:
            raise ValueError("Long entries disabled via engine config.")
        if side is OrderSide.SELL and not self.allow_short:
            raise ValueError("Short entries disabled via engine config.")
        qty = quantity
        if qty is None:
            qty = self._default_quantity(side)
        qty = max(qty, 0.0)
        final_reduce_only = reduce_only
        current_size = self.position_size
        opposite_position = current_size * side.direction < 0
        if self.close_entries_rule == "opposite" and opposite_position:
            qty = min(abs(current_size), qty)
            final_reduce_only = True
        elif opposite_position and not reduce_only:
            qty += abs(current_size)
        return qty, final_reduce_only

    def _resolve_close_quantity(
        self, side: OrderSide, quantity: float | None, qty_percent: float | None
    ) -> float:
        size = self.position_size
        base_qty = quantity if quantity is not None else abs(size)
        if qty_percent is not None:
            base_qty = abs(size) * (qty_percent / 100.0)
        if side is OrderSide.SELL:
            return min(max(size, 0.0), base_qty)
        return min(max(-size, 0.0), base_qty)

    def _default_close_quantity(self, side: OrderSide) -> float:
        size = self.position_size
        if side is OrderSide.SELL:
            return max(size, 0.0)
        return max(-size, 0.0)

    def _default_quantity(self, side: OrderSide) -> float:
        if self.default_qty_type == "fixed":
            return self.default_qty_value
        price = self.bar.get("close") or self.bar.get("open")
        if price <= 0:
            raise ValueError("Invalid price for percent_equity sizing.")
        position_value = (self.default_qty_value / 100.0) * self.equity
        qty = position_value / price
        return abs(qty)


class BaseStrategy(ABC):
    """All Titus strategies conform to this interface."""

    name: str = "Unnamed Strategy"

    @abstractmethod
    def prepare(self) -> None:
        """Called once before bar iteration begins."""

    @abstractmethod
    def on_bar(self, context: StrategyContext) -> None:
        """Main strategy hook, executed for every bar."""

    @abstractmethod
    def finalize(self) -> Dict[str, Any]:
        """Called after the final bar to emit summary stats."""
