"""Execution engine contracts focused on TradingView parity."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import math

import pandas as pd

from titus_core.indicators.registry import IndicatorRegistry
from titus_core.strategies.base import BaseStrategy, PositionSnapshot, StrategyContext
from titus_core.trading.orders import (
    ExecutionTiming,
    Fill,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Trade,
)
from titus_core.trading.portfolio import Portfolio
from titus_core.utils.config import EngineConfig


@dataclass
class BacktestArtifact:
    """Placeholder structure for engine outputs."""

    equity_curve: Iterable[float]
    fills: List[Fill]
    trades: List[Trade]
    metadata: dict[str, Any]


class ExecutionEngine(ABC):
    """Drives the bar iteration loop and order handling."""

    @abstractmethod
    def run(self, strategy: BaseStrategy, bars: pd.DataFrame) -> BacktestArtifact:
        """Execute the strategy and return deterministic artifacts."""


class BarExecutionEngine(ExecutionEngine):
    """Bar-by-bar executor matching TradingView broker emulator semantics."""

    def __init__(self, config: EngineConfig) -> None:
        if config.calc_on_every_tick:
            raise NotImplementedError(
                "calc_on_every_tick=true is not supported in Titus v0. "
                "Strategies execute once per bar, matching TradingView defaults."
            )
        self.config = config

    def run(self, strategy: BaseStrategy, bars: pd.DataFrame) -> BacktestArtifact:
        if bars.empty:
            raise ValueError("No bars provided to execution engine.")

        strategy.prepare()

        total_bars = len(bars)
        portfolio = Portfolio(
            cash=self.config.initial_capital,
            commission_rate=self.config.commission,
            use_leverage=self.config.use_leverage,
            leverage=self.config.leverage,
            margin_long=self.config.margin_long,
            margin_short=self.config.margin_short,
            initial_capital=self.config.initial_capital,
        )
        indicator_registry = IndicatorRegistry(bars)
        equity_curve: List[float] = []
        fills: List[Fill] = []
        pending_by_bar: Dict[int, List[Order]] = defaultdict(list)
        active_orders: List[Order] = []
        debug_messages: List[str] = []

        self._pending_long_entries = 0
        self._pending_short_entries = 0

        for bar_index, (timestamp, row) in enumerate(bars.iterrows()):
            ts = pd.Timestamp(timestamp)
            bar = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 0.0),
            }

            # Activate orders scheduled for this bar's open.
            due_orders = pending_by_bar.pop(bar_index, [])
            newly_active: List[Order] = []
            entry_adjustments: List[tuple[float, float, float]] = []
            for order in due_orders:
                order.activate(bar["open"])
                self._release_pending(order)
                if order.request.order_type == OrderType.MARKET:
                    # For entry orders (non-reduce-only), check if position is flat
                    # This matches Pine's behavior where entry orders don't fill if position_size != 0
                    if not order.request.reduce_only and portfolio.position.size != 0:
                        # Position is not flat, skip this entry order (matches Pine behavior)
                        order.status = OrderStatus.CANCELED
                        continue
                    fill = self._fill_market_order(order, bar, bar_index, ts, portfolio)
                    if fill:
                        # Log entry fill with equity and margin info
                        if not order.request.reduce_only:
                            import logging
                            logger = logging.getLogger(__name__)
                            margin_req = fill.quantity * fill.price * 0.20 if self.config.use_leverage else fill.quantity * fill.price
                            logger.info(
                                f"ENTRY_FILL: time={ts.isoformat()} entry_id={order.request.order_id} "
                                f"fill_price={fill.price:.2f} size={fill.quantity:.3f} avg_price={fill.price:.2f} "
                                f"equity_at_fill={portfolio.equity(bar['open']):.2f} margin_req={margin_req:.2f}"
                            )
                        fills.append(fill)
                        previous_size = portfolio.position.size
                        portfolio.apply_fill(fill, bar_index, ts.isoformat())
                        if not order.request.reduce_only:
                            # Entry fill
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.info(
                                f"ENTRY_FILL: time={ts.isoformat()} entry_id={order.request.order_id} "
                                f"fill_price={fill.price:.2f} size={fill.quantity:.3f} "
                                f"avg_price={portfolio.position.avg_price:.2f}"
                            )
                            entry_adjustments.append(
                                (previous_size, portfolio.position.size, fill.price)
                            )
                else:
                    newly_active.append(order)
            if entry_adjustments:
                candidates = active_orders + newly_active
                for prev_size, new_size, fill_price in entry_adjustments:
                    self._realign_reduce_only_orders_after_entry(
                        prev_size, new_size, fill_price, candidates
                    )
            active_orders.extend(newly_active)

            # Evaluate active stop/limit orders on this bar.
            # CRITICAL: Use TradingView's O→H→L→C price path logic when both stop and limit can fill.
            # If both are touched, only the one hit first (based on bar direction) should fill.
            stop_limit_logs: list[str] = []
            carry_over: List[Order] = []
            limit_queue = [
                order
                for order in active_orders
                if order.request.reduce_only and order.request.order_type == OrderType.LIMIT
            ]
            stop_queue = [
                order
                for order in active_orders
                if order.request.reduce_only and order.request.order_type == OrderType.STOP
            ]
            remaining_queue = [
                order
                for order in active_orders
                if order not in limit_queue and order not in stop_queue
            ]
            
            # TradingView execution priority: STOP orders before LIMIT orders
            # This matches TV's broker emulator behavior where stops have priority
            # See: https://www.tradingview.com/pine-script-docs/language/execution-model/
            evaluation_queue = stop_queue + limit_queue + remaining_queue
            
            exit_filled = False  # Only allow ONE exit per bar (TradingView behavior)
            for order in evaluation_queue:
                # Skip remaining exit orders if one already filled
                if exit_filled and order.request.reduce_only:
                    carry_over.append(order)
                    continue
                    
                fill = self._evaluate_stop_limit_order(order, bar, bar_index, ts, portfolio, stop_limit_logs)
                if fill:
                    fills.append(fill)
                    # Log exit fill before position is updated
                    if order.request.reduce_only:
                        import logging
                        logger = logging.getLogger(__name__)
                        entry_price = portfolio.position.avg_price
                        entry_size = abs(portfolio.position.size)
                        direction_mult = 1.0 if portfolio.position.size > 0 else -1.0
                        profit = (fill.price - entry_price) * fill.quantity * direction_mult
                        logger.info(
                            f"EXIT_FILL: exit_time={ts.isoformat()} exit_price={fill.price:.2f} "
                            f"entry_price={entry_price:.2f} entry_time={ts.isoformat()} "
                            f"entry_id={order.request.order_id} size={fill.quantity:.3f} profit={profit:.3f}"
                        )
                        exit_filled = True  # Mark that exit occurred
                    portfolio.apply_fill(fill, bar_index, ts.isoformat())
                    order.status = OrderStatus.FILLED
                    self._release_pending(order)
                else:
                    carry_over.append(order)
            active_orders = carry_over

            current_equity = portfolio.equity(bar["close"])
            equity_curve.append(current_equity)

            # Build strategy context and collect new orders.
            collected_requests: List[OrderRequest] = []
            cancel_requests: List[str] = []
            new_active_orders: List[Order] = []

            ctx_metadata = {
                "config": self.config.model_dump(),
                "position_size": portfolio.position.size,
                "position_avg_price": portfolio.position.avg_price,
                "is_last_bar": bar_index == total_bars - 1,
                "bars_total": total_bars,
            }
            position_snapshot = PositionSnapshot(
                size=portfolio.position.size,
                avg_price=portfolio.position.avg_price,
            )
            ctx = StrategyContext(
                timestamp=ts,
                bar_index=bar_index,
                bar=bar,
                metadata={**ctx_metadata, "stop_limit_logs": stop_limit_logs},
                features={},
                indicator_registry=indicator_registry,
                order_dispatcher=collected_requests.append,
                cancel_dispatcher=cancel_requests.append,
                default_timing=ExecutionTiming.NEXT_BAR_OPEN,
                process_orders_on_close=self.config.process_orders_on_close,
                default_qty_type=self.config.default_qty_type,
                default_qty_value=self.config.default_qty_value,
                allow_long=self.config.allow_long,
                allow_short=self.config.allow_short,
                close_entries_rule=self.config.close_entries_rule,
                position=position_snapshot,
                equity=current_equity,
            )

            strategy.on_bar(ctx)
            debug_messages.extend(stop_limit_logs)

            for cancel_id in cancel_requests:
                self._cancel_order(cancel_id, pending_by_bar, active_orders)
            cancel_requests.clear()

            for request in collected_requests:
                order = Order(
                    request=request,
                    created_bar_index=bar_index,
                    reference_price=bar["close"],
                )
                self._validate_pyramiding(order, portfolio)
                if request.order_type in {OrderType.STOP, OrderType.LIMIT}:
                    self._cancel_order(order.request.order_id, pending_by_bar, active_orders)
                    order.activate(bar["close"])
                    new_active_orders.append(order)
                    self._register_pending(order)
                elif (
                    request.timing is ExecutionTiming.CURRENT_BAR_CLOSE
                    and self.config.process_orders_on_close
                ):
                    price = self._apply_slippage(bar["close"], order.request.side)
                    fill = Fill(
                        order_id=order.request.order_id,
                        bar_index=bar_index,
                        price=price,
                        quantity=order.request.quantity,
                        side=order.request.side,
                        timestamp=ts.isoformat(),
                        reduce_only=order.request.reduce_only,
                    )
                    fills.append(fill)
                    portfolio.apply_fill(fill, bar_index, ts.isoformat())
                else:
                    # For MARKET orders with NEXT_BAR_OPEN timing, cancel existing order with same ID
                    # This matches Pine's behavior where strategy.entry("Long", ...) replaces previous orders
                    if request.order_type == OrderType.MARKET and not request.reduce_only:
                        self._cancel_order(order.request.order_id, pending_by_bar, active_orders)
                    pending_by_bar[bar_index + 1].append(order)
                    self._register_pending(order)

            if new_active_orders:
                limit_queue = [
                    order
                    for order in new_active_orders
                    if order.request.reduce_only and order.request.order_type == OrderType.LIMIT
                ]
                stop_queue = [
                    order
                    for order in new_active_orders
                    if order.request.reduce_only and order.request.order_type == OrderType.STOP
                ]
                remaining_queue = [
                    order
                    for order in new_active_orders
                    if order not in limit_queue and order not in stop_queue
                ]
                evaluation_queue = limit_queue + stop_queue + remaining_queue
                carry_new: List[Order] = []
                for order in evaluation_queue:
                    fill = self._evaluate_stop_limit_order(order, bar, bar_index, ts, portfolio, stop_limit_logs)
                    if fill:
                        fills.append(fill)
                        portfolio.apply_fill(fill, bar_index, ts.isoformat())
                        order.status = OrderStatus.FILLED
                        self._release_pending(order)
                    else:
                        carry_new.append(order)
                active_orders.extend(carry_new)

        if portfolio.position.size != 0:
            self._flatten_open_position(portfolio, bars, fills)
            equity_curve[-1] = portfolio.equity(bars.iloc[-1]["close"])

        summary = strategy.finalize()
        metadata = {
            "engine_config": self.config.model_dump(),
            "strategy_summary": summary,
            "bars": len(bars),
            "stop_limit_logs": debug_messages,
        }
        return BacktestArtifact(
            equity_curve=equity_curve,
            fills=fills,
            trades=portfolio.trades,
            metadata=metadata,
        )

    def _flatten_open_position(
        self, portfolio: Portfolio, bars: pd.DataFrame, fills: list[Fill]
    ) -> None:
        last_index = len(bars) - 1
        timestamp = pd.Timestamp(bars.index[-1]).isoformat()
        price = bars.iloc[-1]["close"]
        side = OrderSide.SELL if portfolio.position.size > 0 else OrderSide.BUY
        quantity = abs(portfolio.position.size)
        fill = Fill(
            order_id="final_flatten",
            bar_index=last_index,
            price=price,
            quantity=quantity,
            side=side,
            timestamp=timestamp,
            reduce_only=True,
        )
        fills.append(fill)
        portfolio.apply_fill(fill, last_index, timestamp)

    def _fill_market_order(
        self, order: Order, bar: Dict[str, float], bar_index: int, timestamp: pd.Timestamp, portfolio: Portfolio
    ) -> Optional[Fill]:
        if order.request.reduce_only:
            # Only allow if position is opposite; enforcement handled upstream for now.
            pass
        
        # Calculate fill price with slippage
        fill_price = self._apply_slippage(bar["open"], order.request.side)
        
        # For entry orders, validate that we have sufficient equity (matches Pine behavior)
        # Pine Script reduces quantity when margin requirement exceeds available equity
        # When leverage is enabled, check margin requirement instead of full notional
        quantity = order.request.quantity
        if not order.request.reduce_only:
            current_equity = portfolio.equity(bar["open"])
            
            if self.config.use_leverage:
                # Margin-based validation for leveraged positions
                # Calculate margin requirement based on leverage
                margin_pct = 100.0 / self.config.leverage if self.config.leverage > 0 else 100.0
                # Use explicit margin if provided, otherwise calculate from leverage
                if order.request.side is OrderSide.BUY and self.config.margin_long is not None:
                    margin_pct = self.config.margin_long
                elif order.request.side is OrderSide.SELL and self.config.margin_short is not None:
                    margin_pct = self.config.margin_short
                
                # Calculate maximum quantity that can be filled given available equity
                # margin_required = quantity * price * margin_pct / 100
                # If margin_required > equity, reduce quantity to: equity / (price * margin_pct / 100)
                max_margin_available = current_equity
                max_qty_by_margin = (max_margin_available * 100.0) / (fill_price * margin_pct) if fill_price > 0 and margin_pct > 0 else 0.0
                
                # Round down to avoid exceeding margin (matches Pine's behavior of reducing quantity)
                # Pine Script reduces quantity when margin required > available equity
                # Always cap quantity to max affordable by margin (Pine Script does this)
                if max_qty_by_margin > 0:
                    # Always cap to max quantity that can be afforded (even if it equals requested)
                    # Pine Script caps even when max_qty <= requested to account for rounding
                    if max_qty_by_margin < quantity:
                        # Cap quantity to maximum affordable by margin
                        quantity = max_qty_by_margin
                    
                    # Always round down to step size (get from strategy metadata if available)
                    # For now, use default step_size (0.001 for BTCUSDT.P)
                    # This matches the strategy's step_size parameter
                    step_size = 0.001  # Default step size for BTCUSDT.P
                    
                    # Round down to step size to avoid exceeding margin
                    if step_size > 0:
                        quantity = math.floor(quantity / step_size) * step_size
                    
                    # If quantity is too small after reduction, reject the order
                    if quantity <= 0 or quantity < step_size:
                        order.status = OrderStatus.CANCELED
                        return None
            else:
                # Original behavior: reject if full notional cost exceeds equity
                cost = fill_price * quantity
                if cost > current_equity:
                    order.status = OrderStatus.CANCELED
                    return None
            
        order.status = OrderStatus.FILLED
        return Fill(
            order_id=order.request.order_id,
            bar_index=bar_index,
            price=fill_price,
            quantity=quantity,
            side=order.request.side,
            timestamp=timestamp.isoformat(),
            reduce_only=order.request.reduce_only,
        )

    def _evaluate_stop_limit_order(
        self,
        order: Order,
        bar: Dict[str, float],
        bar_index: int,
        timestamp: pd.Timestamp,
        portfolio: Portfolio,
        debug_log: list[str],
    ) -> Optional[Fill]:
        req = order.request
        price = None
        filled_via_jump = False
        if req.order_type == OrderType.LIMIT and req.limit_price is not None:
            price, _ = self._simulate_limit_price(req, bar, bar_index, timestamp, debug_log)
        elif req.order_type == OrderType.STOP and req.stop_price is not None:
            price = self._simulate_stop_price(req, bar, bar_index, timestamp, debug_log)
        if price is None:
            return None

        quantity = req.quantity
        if req.reduce_only:
            available = (
                max(portfolio.position.size, 0.0)
                if req.side is OrderSide.SELL
                else max(-portfolio.position.size, 0.0)
            )
            if available <= 0:
                return None
            quantity = min(quantity, available)
        if quantity <= 0:
            return None

        price = self._apply_slippage(price, req.side)
        return Fill(
            order_id=req.order_id,
            bar_index=bar_index,
            price=price,
            quantity=quantity,
            side=req.side,
            timestamp=timestamp.isoformat(),
            reduce_only=req.reduce_only,
        )

    def _simulate_limit_price(
        self,
        request: OrderRequest,
        bar: Dict[str, float],
        bar_index: int,
        timestamp: pd.Timestamp,
        debug_log: list[str],
    ) -> tuple[Optional[float], bool]:
        limit = request.limit_price
        if limit is None:
            return None, False
        open_price = bar["open"]
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]
        filled_via_jump = False
        if request.side is OrderSide.BUY:
            if open_price <= limit:
                # Bar opens at or below limit: fill at limit price (TradingView behavior)
                # Slippage will be applied later
                return limit, False
            if low <= limit:
                filled_via_jump = open_price > limit
                return limit, filled_via_jump
        else:
            if open_price >= limit:
                # Bar opens at or above limit: fill at limit price (TradingView behavior)
                # Slippage will be applied later
                return limit, False
            if high >= limit:
                filled_via_jump = open_price < limit
                return limit, filled_via_jump
        debug_log.append(
            f"MISSED_LIMIT bar={bar_index} ts={timestamp.isoformat()} order_id={request.order_id} "
            f"side={request.side} limit={limit} open={open_price} high={high} low={low} close={close}"
        )
        return None, False

    def _simulate_stop_price(
        self,
        request: OrderRequest,
        bar: Dict[str, float],
        bar_index: int,
        timestamp: pd.Timestamp,
        debug_log: list[str],
    ) -> Optional[float]:
        stop = request.stop_price
        if stop is None:
            return None
        open_price = bar["open"]
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]
        if request.side is OrderSide.BUY:
            if open_price >= stop:
                # Bar opens at or above stop: fill at stop price (TradingView behavior)
                # Slippage will be applied later
                return stop
            if high >= stop:
                return stop
        else:
            if open_price <= stop:
                # Bar opens at or below stop: fill at stop price (TradingView behavior)
                # Slippage will be applied later
                return stop
            if low <= stop:
                return stop
        debug_log.append(
            f"MISSED_STOP bar={bar_index} ts={timestamp.isoformat()} order_id={request.order_id} "
            f"side={request.side} stop={stop} open={open_price} high={high} low={low} close={close}"
        )
        return None

    def _register_pending(self, order: Order) -> None:
        if order.request.reduce_only:
            return
        if order.request.side is OrderSide.BUY:
            self._pending_long_entries += 1
        else:
            self._pending_short_entries += 1

    def _release_pending(self, order: Order) -> None:
        if order.request.reduce_only:
            return
        if order.request.side is OrderSide.BUY:
            self._pending_long_entries = max(0, self._pending_long_entries - 1)
        else:
            self._pending_short_entries = max(0, self._pending_short_entries - 1)

    def _cancel_order(
        self,
        order_id: str,
        pending_by_bar: Dict[int, List[Order]],
        active_orders: List[Order],
    ) -> None:
        for orders in pending_by_bar.values():
            for order in list(orders):
                if order.request.order_id == order_id:
                    orders.remove(order)
                    order.status = OrderStatus.CANCELED
                    self._release_pending(order)
        for order in list(active_orders):
            if order.request.order_id == order_id:
                active_orders.remove(order)
                order.status = OrderStatus.CANCELED
                self._release_pending(order)

    def _validate_pyramiding(self, order: Order, portfolio: Portfolio) -> None:
        if order.request.reduce_only:
            return
        allowed = self.config.pyramiding
        if order.request.side is OrderSide.BUY:
            current_entries = portfolio.entries_count(OrderSide.BUY)
            future = current_entries + self._pending_long_entries + 1
            if future > allowed:
                raise ValueError(
                    f"Pyramiding limit ({allowed}) exceeded for long exposure."
                )
        else:
            current_entries = portfolio.entries_count(OrderSide.SELL)
            future = current_entries + self._pending_short_entries + 1
            if future > allowed:
                raise ValueError(
                    f"Pyramiding limit ({allowed}) exceeded for short exposure."
                )

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        slip = self.config.slippage
        if slip <= 0:
            return price
        stype = self.config.slippage_type
        if stype == "percent":
            amount = price * slip
        elif stype == "ticks":
            if not self.config.tick_size or self.config.tick_size <= 0:
                raise ValueError("tick_size must be > 0 when slippage_type='ticks'.")
            amount = slip * self.config.tick_size
        else:
            amount = slip
        result = price + amount * side.direction
        return result


    def _realign_reduce_only_orders_after_entry(
        self,
        previous_size: float,
        new_size: float,
        fill_price: float,
        active_orders: List[Order],
    ) -> None:
        """Shift reduce-only stop/limit prices to track the actual fill price."""

        if previous_size != 0:
            return
        if new_size == 0:
            return
        
        for active in active_orders:
            if not active.request.reduce_only:
                continue
            anchor = active.reference_price or active.last_activation_price
            if anchor is None:
                continue
            delta = fill_price - anchor
            if delta == 0:
                continue
            if active.request.order_type == OrderType.STOP and active.request.stop_price is not None:
                active.request.stop_price += delta
            if active.request.order_type == OrderType.LIMIT and active.request.limit_price is not None:
                active.request.limit_price += delta
            active.reference_price = fill_price
