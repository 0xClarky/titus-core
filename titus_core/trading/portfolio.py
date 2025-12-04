"""Portfolio and trade accounting matching TradingView semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from titus_core.trading.orders import Fill, OrderSide, Trade


@dataclass
class Lot:
    quantity: float
    price: float
    timestamp: str
    order_id: str
    side: OrderSide
    entry_commission_per_unit: float
    bar_index: int

    @property
    def direction(self) -> int:
        return 1 if self.side is OrderSide.BUY else -1


@dataclass
class Position:
    size: float = 0.0
    avg_price: float = 0.0


@dataclass
class Portfolio:
    cash: float
    commission_rate: float = 0.0
    position: Position = field(default_factory=Position)
    trades: List[Trade] = field(default_factory=list)
    _lots: List[Lot] = field(default_factory=list)
    # Leverage support
    use_leverage: bool = False
    leverage: float = 1.0
    margin_long: float | None = None
    margin_short: float | None = None
    initial_capital: float = 0.0  # Track initial capital for equity calculation with leverage

    def apply_fill(self, fill: Fill, bar_index: int, timestamp: str) -> None:
        commission = abs(fill.price * fill.quantity) * self.commission_rate
        
        if self.use_leverage:
            # Margin-based accounting for leveraged positions
            # Calculate margin requirement
            margin_pct = 100.0 / self.leverage if self.leverage > 0 else 100.0
            # Use explicit margin if provided
            if fill.side is OrderSide.BUY and self.margin_long is not None:
                margin_pct = self.margin_long
            elif fill.side is OrderSide.SELL and self.margin_short is not None:
                margin_pct = self.margin_short
            
            # Deduct only margin requirement, not full notional
            margin_required = abs(fill.price * fill.quantity) * (margin_pct / 100.0)
            self.cash -= margin_required * fill.side.direction
        else:
            # Original behavior: deduct full notional value
            self.cash -= fill.price * fill.quantity * fill.side.direction
        
        self.cash -= commission

        qty_remaining = fill.quantity
        commission_per_unit = commission / fill.quantity if fill.quantity else 0.0

        if self._is_entry(fill):
            entry_commission_per_unit = commission_per_unit
            self._add_lot(
                fill,
                timestamp,
                qty_remaining,
                entry_commission_per_unit,
                bar_index,
            )
            self._refresh_position()
            return

        idx = 0
        while qty_remaining > 0 and idx < len(self._lots):
            lot = self._lots[idx]
            match_qty = min(lot.quantity, qty_remaining)
            pnl = self._compute_lot_pnl(lot, fill.price, match_qty)
            entry_commission = lot.entry_commission_per_unit * match_qty
            exit_commission = commission_per_unit * match_qty
            pnl -= entry_commission + exit_commission
            direction = "long" if lot.direction > 0 else "short"
            self.trades.append(
                Trade(
                    entry_id=lot.order_id,
                    exit_id=fill.order_id,
                    entry_price=lot.price,
                    exit_price=fill.price,
                    entry_time=lot.timestamp,
                    exit_time=timestamp,
                    quantity=match_qty,
                    direction=direction,
                    pnl=pnl,
                    entry_bar=lot.bar_index,
                    exit_bar=bar_index,
                )
            )
            lot.quantity -= match_qty
            qty_remaining -= match_qty
            if lot.quantity == 0:
                self._lots.pop(idx)
            else:
                idx += 1

        if qty_remaining > 0 and not fill.reduce_only:
            rollover_fill = Fill(
                order_id=fill.order_id,
                bar_index=bar_index,
                price=fill.price,
                quantity=qty_remaining,
                side=fill.side,
                timestamp=timestamp,
            )
            self._add_lot(
                rollover_fill,
                timestamp,
                qty_remaining,
                commission_per_unit,
                bar_index,
            )

        self._refresh_position()

    def equity(self, price: float) -> float:
        if self.use_leverage:
            # With leverage: Equity = Initial Capital + Realized PnL + Unrealized PnL
            # Calculate realized PnL from trades
            realized_pnl = sum(trade.pnl for trade in self.trades)
            # Calculate unrealized PnL from open position
            unrealized_pnl = 0.0
            if self.position.size != 0:
                # Calculate unrealized PnL: (current_price - avg_entry_price) * position_size
                if self.position.size > 0:  # Long position
                    unrealized_pnl = (price - self.position.avg_price) * self.position.size
                else:  # Short position
                    unrealized_pnl = (self.position.avg_price - price) * abs(self.position.size)
            return self.initial_capital + realized_pnl + unrealized_pnl
        else:
            # Without leverage: Equity = Cash + Position Value (spot-style)
            return self.cash + self.position.size * price

    def entries_count(self, side: OrderSide) -> int:
        if side is OrderSide.BUY:
            return sum(1 for lot in self._lots if lot.direction > 0)
        return sum(1 for lot in self._lots if lot.direction < 0)

    def _is_entry(self, fill: Fill) -> bool:
        if fill.reduce_only:
            return False
        current_size = self.position.size
        if current_size == 0:
            return True
        return current_size * fill.side.direction >= 0

    def _add_lot(
        self,
        fill: Fill,
        timestamp: str,
        quantity: float,
        entry_commission_per_unit: float,
        bar_index: int,
    ) -> None:
        self._lots.append(
            Lot(
                quantity=quantity,
                price=fill.price,
                timestamp=timestamp,
                order_id=fill.order_id,
                side=fill.side,
                entry_commission_per_unit=entry_commission_per_unit,
                bar_index=bar_index,
            )
        )

    def _compute_lot_pnl(self, lot: Lot, exit_price: float, quantity: float) -> float:
        if lot.direction > 0:
            return (exit_price - lot.price) * quantity
        return (lot.price - exit_price) * quantity

    def _refresh_position(self) -> None:
        size = sum(lot.direction * lot.quantity for lot in self._lots)
        self.position.size = size
        if size == 0:
            self.position.avg_price = 0.0
            return
        total_cost = sum(lot.price * lot.quantity for lot in self._lots)
        total_qty = sum(lot.quantity for lot in self._lots)
        self.position.avg_price = total_cost / total_qty if total_qty else 0.0
