"""Reporting helpers for Titus backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from titus_core.trading.orders import Trade
from titus_core.utils.config import EngineConfig

TRADINGVIEW_COLUMNS = [
    "Entry time",
    "Entry price",
    "Exit time",
    "Exit price",
    "Direction",
    "Qty",
    "Profit",
    "Leverage",
]


@dataclass
class BacktestReport:
    equity_curve: Iterable[float]
    trades: List[Trade]
    engine_config: EngineConfig | None = None

    def _calculate_leverage(self, trade: Trade) -> float:
        """Calculate actual leverage used for a trade based on notional value and equity at entry."""
        if self.engine_config is None:
            return 1.0  # Default to 1x if no config provided
        
        if not self.engine_config.use_leverage:
            return 1.0  # Spot trading, no leverage
        
        # Calculate notional value of the trade
        notional = trade.entry_price * trade.quantity
        
        # Get equity at entry time from equity curve
        # entry_bar is the bar index when entry occurred
        # equity_curve[entry_bar - 1] is equity at end of previous bar (available when order placed)
        # For first bar (entry_bar=0), use initial_capital
        equity_at_entry: float
        if trade.entry_bar == 0:
            equity_at_entry = self.engine_config.initial_capital
        elif trade.entry_bar <= len(self.equity_curve):
            # Use equity from previous bar (what was available when order was placed)
            equity_at_entry = self.equity_curve[trade.entry_bar - 1]
        else:
            # Fallback: use last known equity or initial capital
            equity_at_entry = self.equity_curve[-1] if self.equity_curve else self.engine_config.initial_capital
        
        # Calculate actual leverage: notional / equity
        if equity_at_entry > 0:
            actual_leverage = notional / equity_at_entry
        else:
            actual_leverage = 1.0
        
        return round(actual_leverage, 2)  # Round to 2 decimal places

    def metrics(self) -> Dict[str, float]:
        if not self.equity_curve:
            return {
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "avg_trade": 0.0,
                "profit_factor": 0.0,
                "linearity_r2": 0.0,
                "ulcer_index": 0.0,
            }
        curve = pd.Series(self.equity_curve)
        total_return = (curve.iloc[-1] - curve.iloc[0]) / curve.iloc[0]
        roll_max = curve.cummax()
        drawdown = (curve - roll_max) / roll_max
        max_dd = drawdown.min()
        drawdown_pct = (roll_max - curve) / roll_max
        ulcer_index = float(np.sqrt(np.nanmean(np.square(drawdown_pct.fillna(0.0)))))
        if len(curve) >= 2:
            x = np.arange(len(curve), dtype=float)
            y = curve.to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r_squared = 0.0
        wins = [trade.pnl for trade in self.trades if trade.pnl > 0]
        losses = [abs(trade.pnl) for trade in self.trades if trade.pnl < 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        total_win = sum(wins)
        total_loss = sum(losses)
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf") if total_win > 0 else 0.0
        avg_trade = (total_win - total_loss) / len(self.trades) if self.trades else 0.0
        return {
            "total_return": float(total_return),
            "max_drawdown": float(max_dd),
            "trades": len(self.trades),
            "win_rate": win_rate,
            "avg_trade": avg_trade,
            "profit_factor": profit_factor,
            "linearity_r2": max(0.0, min(1.0, r_squared)),
            "ulcer_index": float(ulcer_index),
        }

    def trades_dataframe(self) -> pd.DataFrame:
        rows = []
        for trade in self.trades:
            leverage = self._calculate_leverage(trade)
            rows.append(
                {
                    "Entry time": trade.entry_time,
                    "Entry price": trade.entry_price,
                    "Exit time": trade.exit_time,
                    "Exit price": trade.exit_price,
                    "Direction": trade.direction,
                    "Qty": trade.quantity,
                    "Profit": trade.pnl,
                    "Leverage": leverage,
                }
            )
        return pd.DataFrame(rows, columns=TRADINGVIEW_COLUMNS)

    def export_trades_csv(self, path: Path) -> None:
        df = self.trades_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
