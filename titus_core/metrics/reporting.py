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
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "k_ratio": 0.0,
            }
        curve = pd.Series(self.equity_curve)
        total_return = (curve.iloc[-1] - curve.iloc[0]) / curve.iloc[0]
        roll_max = curve.cummax()
        drawdown = (curve - roll_max) / roll_max
        max_dd = drawdown.min()
        
        # Calculate returns from equity curve
        returns = curve.pct_change().dropna()
        initial_equity = curve.iloc[0]
        
        # Sharpe ratio: (mean return / std return) * sqrt(annualization_factor)
        # Assuming daily bars for now (252 trading days per year)
        # For 4h bars: 6 bars per day * 252 = 1512 bars per year
        # For 1h bars: 24 bars per day * 252 = 6048 bars per year
        # We'll use a conservative estimate based on data length
        periods_per_year = 252  # Default to daily, will be approximate for other resolutions
        sharpe = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        
        # Sortino ratio: (mean return / downside std) * sqrt(annualization_factor)
        # Downside deviation only considers negative returns
        downside_returns = returns[returns < 0]
        sortino = 0.0
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
        elif returns.mean() > 0:
            # If no downside returns, sortino is infinity-like, cap it
            sortino = 100.0  # Cap at high value to avoid infinity
        
        # K-ratio: (slope / std_error_of_slope)
        # Calculate regression slope and standard error
        # Normalize slope by initial equity to make it comparable across different capital sizes
        k_ratio = 0.0
        if len(curve) >= 2 and initial_equity > 0:
            x = np.arange(len(curve), dtype=float)
            y = curve.to_numpy(dtype=float)
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            residuals = y - y_pred
            # Standard error of slope
            ss_res = np.sum(residuals ** 2)
            if len(curve) > 2:
                mse = ss_res / (len(curve) - 2)  # Mean squared error
                x_mean = np.mean(x)
                ss_x = np.sum((x - x_mean) ** 2)
                if ss_x > 0:
                    std_error_slope = np.sqrt(mse / ss_x)
                    if std_error_slope > 0:
                        # Normalize slope by initial equity to make K-ratio comparable
                        normalized_slope = slope / initial_equity
                        normalized_std_error = std_error_slope / initial_equity
                        k_ratio = normalized_slope / normalized_std_error if normalized_std_error > 0 else 0.0
        
        wins = [trade.pnl for trade in self.trades if trade.pnl > 0]
        losses = [abs(trade.pnl) for trade in self.trades if trade.pnl < 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        total_win = sum(wins)
        total_loss = sum(losses)
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf") if total_win > 0 else 0.0
        
        return {
            "total_return": float(total_return),
            "max_drawdown": float(max_dd),
            "trades": len(self.trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "k_ratio": float(k_ratio),
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
