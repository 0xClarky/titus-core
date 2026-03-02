"""Reporting helpers for Titus backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from titus_core.trading.orders import Trade
from titus_core.utils.config import EngineConfig

try:  # optional plotting support
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
except Exception:  # pragma: no cover - matplotlib optional
    plt = None
    mdates = None
    sns = None

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
        if len(self.equity_curve) == 0:
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

    def plot_equity_enhanced(self, path: Path, bars: pd.DataFrame = None) -> None:
        """Generate enhanced equity curve with drawdown overlay.
        
        Args:
            path: Path to save the PNG file
            bars: Optional price data for buy & hold comparison
        """
        if plt is None:
            print("Warning: matplotlib not available, skipping equity_enhanced plot")
            return
        
        # Convert equity curve to pandas Series with datetime index
        if not self.equity_curve:
            print("Warning: Empty equity curve, skipping equity_enhanced plot")
            return
        
        # Create equity series - if bars provided, use its index; otherwise use integer index
        if bars is not None and len(bars) == len(self.equity_curve):
            equity = pd.Series(self.equity_curve, index=bars.index, name='equity')
        else:
            # Create a simple datetime index
            equity = pd.Series(self.equity_curve, name='equity')
            equity.index = pd.date_range(start='2024-01-01', periods=len(equity), freq='D')
        
        dates = equity.index
        
        # Create figure with 2 subplots: equity curve and drawdown
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Calculate drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        
        # Plot equity curve
        ax1.plot(dates, equity, label='Strategy', linewidth=2, color='#2E86AB')
        
        # Add drawdown shading
        ax1.fill_between(dates, peak, equity, alpha=0.3, color='red', 
                        where=(drawdown < 0), label='Drawdown')
        
        # Add S&P500 benchmark (10% annual return)
        initial_value = equity.iloc[0]
        if len(dates) > 1:
            # Calculate S&P500 10% annual compound return
            sp500_annual_rate = 0.10
            
            # For each point, calculate compound return based on time elapsed
            days_elapsed = pd.Series((dates - dates[0]).days, index=dates)
            years_at_point = days_elapsed / 365.25
            sp500_equity = initial_value * np.power(1 + sp500_annual_rate, years_at_point)
            
            # Calculate total return for label
            sp500_total_return = (sp500_equity.iloc[-1] - initial_value) / initial_value
            
            ax1.plot(dates, sp500_equity, '--', label=f'S&P500 10% Annual ({sp500_total_return:.1%} total)', 
                    color='gray', alpha=0.7)
        else:
            # Fallback for single data point
            ax1.plot(dates, [initial_value], '--', label='S&P500 10% Annual', 
                    color='gray', alpha=0.7)
        
        ax1.set_title('Enhanced Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown in bottom subplot
        ax2.fill_between(dates, 0, drawdown * 100, color='red', alpha=0.7)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        if mdates is not None:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved equity enhanced curve to {path}")
