"""Live execution engine for real-time bar-by-bar strategy execution.

This engine polls exchanges for new bars and executes strategies in real-time,
maintaining the same interface as the backtest engine but with live order submission.

Supports: HyperLiquid, Bybit
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import pandas as pd

from titus_core.indicators.registry import IndicatorRegistry
from titus_core.strategies.base import BaseStrategy, PositionSnapshot, StrategyContext
from titus_core.trading.orders import ExecutionTiming, OrderRequest, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class LiveEngineConfig:
    """Configuration for live execution engine.
    
    Supports both single-symbol and multi-symbol execution across multiple exchanges.
    
    Note: Live equity is fetched from exchange API before each bar execution.
    We don't use a static initial_capital - the strategy always uses live wallet balance.
    """

    # Strategy parameters
    commission: float = 0.00055  # 0.055%
    
    # Execution settings
    bar_resolution: str = "4h"  # Bar resolution to trade on
    symbols: list[str] = None  # Trading symbols (e.g., ["BTC", "ETH", "SOL"])
    exchange: str = "hyperliquid"  # Exchange: "hyperliquid" or "bybit"
    
    # Safety settings
    dry_run: bool = True  # If True, log signals but don't execute
    max_position_size: float = 0.0  # Max notional position per symbol (0 = no limit)
    max_leverage: float = 5.0  # Max leverage multiplier
    
    # Polling settings
    poll_interval: int = 60  # Seconds between bar checks
    lookback_bars: int = 200  # Number of historical bars for indicators
    
    def __post_init__(self):
        """Validate symbols list and exchange after initialization."""
        if self.symbols is None:
            raise ValueError("symbols must be provided")
        if not isinstance(self.symbols, list) or len(self.symbols) == 0:
            raise ValueError("symbols must be a non-empty list")
        if self.exchange not in ("hyperliquid", "bybit"):
            raise ValueError(f"Unsupported exchange: {self.exchange}")


class LiveExecutionEngine:
    """Real-time bar-by-bar strategy executor for multiple exchanges.
    
    Supports both single-symbol and multi-symbol execution.
    Each symbol runs independently with its own strategy instance and state.
    
    Polls exchange for new bars and executes strategy on bar completion,
    submitting orders directly to the exchange (or logging in dry-run mode).
    
    Supported exchanges: HyperLiquid, Bybit
    """

    def __init__(
        self,
        config: LiveEngineConfig,
        exchange_client: Union[Any, Any],
    ) -> None:
        """Initialize live execution engine.

        Args:
            config: Live engine configuration
            exchange_client: Exchange client (HyperLiquidClient or BybitClient)
        """
        self.config = config
        self.exchange_client = exchange_client
        # Keep backward compatibility alias
        self.hl_client = exchange_client
        self.running: bool = False
        
        # Multi-symbol state tracking
        self.symbols = config.symbols
        
        # Per-symbol state (independent for each symbol)
        self.strategies: dict[str, BaseStrategy] = {}
        self.historical_bars: dict[str, pd.DataFrame] = {}
        self.indicator_registries: dict[str, IndicatorRegistry] = {}
        self.last_processed_bar_time: dict[str, Optional[datetime]] = {}
        self.bars_processed: dict[str, int] = {}
        
        # Global tracking
        self.total_bars_processed: int = 0
        
        logger.info(
            f"Live execution engine initialized - "
            f"Exchange: {config.exchange.upper()}, "
            f"Symbols: {len(self.symbols)} ({', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}), "
            f"Resolution: {config.bar_resolution}, "
            f"Mode: {'DRY RUN' if config.dry_run else 'LIVE'}"
        )

    def _fetch_historical_bars(self, symbol: str, lookback: int) -> pd.DataFrame:
        """Fetch historical bars from exchange for indicator calculation.

        Args:
            symbol: Trading symbol
            lookback: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {lookback} historical bars for {symbol} from {self.config.exchange}")
        
        # Use HL Info API to fetch candles
        try:
            # Convert resolution to HL format (e.g., "4h" -> "4h")
            interval = self.config.bar_resolution
            
            # Fetch candles using HL SDK
            # Note: HL uses startTime and endTime in milliseconds
            end_time = int(time.time() * 1000)
            
            # Calculate start time based on resolution and lookback
            resolution_minutes = self._resolution_to_minutes(interval)
            start_time = end_time - (lookback * resolution_minutes * 60 * 1000)
            
            # Note: Currently only HyperLiquid live bar fetching is supported
            # Bybit will need to use titus_core.data.bybit.BybitMarketDataFeed
            if not hasattr(self.exchange_client, 'info'):
                raise NotImplementedError(
                    f"Live bar fetching not yet implemented for {self.config.exchange}. "
                    "Use HyperLiquid for now, or implement Bybit candles API."
                )
            
            candles = self.exchange_client.info.candles_snapshot(
                name=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
            )
            
            if not candles:
                logger.warning("No historical bars returned from HyperLiquid")
                return pd.DataFrame()
            
            # Convert to DataFrame
            # HL returns list of dicts with keys: t, T, s, i, o, c, h, l, v, n
            df = pd.DataFrame(candles)
            
            # Extract and rename columns we need
            # t = start timestamp (ms), o = open, h = high, l = low, c = close, v = volume
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df["open"] = pd.to_numeric(df["o"], errors="coerce")
            df["high"] = pd.to_numeric(df["h"], errors="coerce")
            df["low"] = pd.to_numeric(df["l"], errors="coerce")
            df["close"] = pd.to_numeric(df["c"], errors="coerce")
            df["volume"] = pd.to_numeric(df["v"], errors="coerce")
            
            # Keep only needed columns and set index
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.set_index("timestamp")
            
            logger.info(
                f"Historical bars fetched - Count: {len(df)}, "
                f"Period: {df.index[0]} to {df.index[-1]}"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical bars: {e}", exc_info=True)
            return pd.DataFrame()

    def _resolution_to_minutes(self, resolution: str) -> int:
        """Convert resolution string to minutes.

        Args:
            resolution: Resolution string (e.g., "1h", "4h", "1d")

        Returns:
            Minutes per bar
        """
        resolution = resolution.lower()
        
        if resolution.endswith("m"):
            return int(resolution[:-1])
        elif resolution.endswith("h"):
            return int(resolution[:-1]) * 60
        elif resolution.endswith("d"):
            return int(resolution[:-1]) * 60 * 24
        else:
            # Default to 1 hour
            logger.warning(f"Unknown resolution format '{resolution}', defaulting to 60 minutes")
            return 60

    def _get_latest_bar(self, symbol: str) -> Optional[dict[str, Any]]:
        """Fetch the latest completed bar from HyperLiquid.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with bar data or None if unavailable
        """
        try:
            # Fetch last 2 bars (current incomplete + previous complete)
            if not hasattr(self.exchange_client, 'info'):
                raise NotImplementedError(
                    f"Live bar fetching not yet implemented for {self.config.exchange}"
                )
            
            candles = self.exchange_client.info.candles_snapshot(
                name=symbol,
                interval=self.config.bar_resolution,
                startTime=int((time.time() - 2 * self._resolution_to_minutes(self.config.bar_resolution) * 60) * 1000),
                endTime=int(time.time() * 1000),
            )
            
            if not candles or len(candles) < 2:
                return None
            
            # Take second-to-last bar (most recent complete bar)
            bar = candles[-2]
            
            # Parse bar data (dict format: t, T, s, i, o, c, h, l, v, n)
            timestamp = pd.to_datetime(bar["t"], unit="ms")
            
            return {
                "timestamp": timestamp,
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": float(bar["v"]),
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch latest bar: {e}")
            return None

    def _build_strategy_context(
        self,
        symbol: str,
        timestamp: datetime,
        bar_index: int,
        bar: dict[str, Any],
    ) -> StrategyContext:
        """Build strategy context for the current bar.

        Args:
            symbol: Trading symbol
            timestamp: Bar timestamp
            bar_index: Bar index in historical data
            bar: Current bar data

        Returns:
            StrategyContext for strategy execution
        """
        # Get current position from exchange
        position = self.exchange_client.get_position(symbol)
        
        if position:
            position_snapshot = PositionSnapshot(
                size=position.get("size", 0.0),
                avg_price=position.get("entry_price", 0.0),
            )
        else:
            position_snapshot = PositionSnapshot(size=0.0, avg_price=0.0)
        
        # Get current equity
        equity = self.exchange_client.get_equity()
        
        # Build context
        context = StrategyContext(
            timestamp=timestamp,
            bar_index=bar_index,
            bar=bar,
            metadata={"config": self.config.__dict__, "symbol": symbol},
            indicator_registry=self.indicator_registries[symbol],
            order_dispatcher=lambda order_req: self._order_dispatcher(symbol, order_req),
            cancel_dispatcher=lambda order_id: self._cancel_dispatcher(symbol, order_id),
            default_timing=ExecutionTiming.NEXT_BAR_OPEN,
            process_orders_on_close=False,
            position=position_snapshot,
            equity=equity,
        )
        
        return context

    def _order_dispatcher(self, symbol: str, order_request: OrderRequest) -> None:
        """Dispatch order to HyperLiquid or log in dry-run mode.

        Args:
            symbol: Trading symbol for this order
            order_request: Order request from strategy
        """
        logger.info(
            f"Order signal received - Symbol: {symbol}, ID: {order_request.order_id}, "
            f"Side: {order_request.side.name}, Qty: {order_request.quantity}, "
            f"Type: {order_request.order_type.name}"
        )
        
        if self.config.dry_run:
            logger.info(
                f"[DRY RUN] Order logged (not executed) - "
                f"Symbol: {symbol}, ID: {order_request.order_id}, Side: {order_request.side.name}, "
                f"Qty: {order_request.quantity}, Type: {order_request.order_type.name}, "
                f"Limit: {order_request.limit_price}, Stop: {order_request.stop_price}, "
                f"ReduceOnly: {order_request.reduce_only}"
            )
            return
        
        # Execute order on HyperLiquid
        try:
            # Execute as individual orders
            # TODO: Detect bracket patterns and use place_bracket_order()
            
            if order_request.order_type.name == "MARKET":
                result = self.exchange_client.place_market_order(
                    symbol=symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    reduce_only=order_request.reduce_only,
                    leverage=self.config.max_leverage if not order_request.reduce_only else None,
                )
            elif order_request.order_type.name == "LIMIT":
                result = self.exchange_client.place_limit_order(
                    symbol=symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    limit_price=order_request.limit_price,
                    reduce_only=order_request.reduce_only,
                )
            elif order_request.order_type.name == "STOP":
                result = self.exchange_client.place_stop_order(
                    symbol=symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    stop_price=order_request.stop_price,
                    reduce_only=order_request.reduce_only,
                )
            else:
                logger.error(f"Unsupported order type: {order_request.order_type.name}")
                return
            
            if result.get("success"):
                logger.info(f"Order executed successfully - Symbol: {symbol}, ID: {order_request.order_id}")
                
                # Store HL order ID for future cancellation
                hl_oid = result.get("hl_order_id")
                if hl_oid is not None:
                    self.exchange_client._store_order_id(
                        symbol,
                        order_request.order_id,
                        hl_oid,
                    )
            else:
                logger.error(
                    f"Order execution failed - Symbol: {symbol}, ID: {order_request.order_id}, "
                    f"Error: {result.get('error')}"
                )
                
        except Exception as e:
            logger.error(
                f"Exception executing order - Symbol: {symbol}, ID: {order_request.order_id}, Error: {e}",
                exc_info=True,
            )

    def _cancel_dispatcher(self, symbol: str, order_id: str) -> None:
        """Cancel order dispatcher.

        Args:
            symbol: Trading symbol
            order_id: Strategy's order ID to cancel
        """
        logger.info(f"Order cancel requested - Symbol: {symbol}, ID: {order_id} (dry_run={self.config.dry_run})")
        
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Order cancel logged (not executed) - Symbol: {symbol}, ID: {order_id}")
            return
        
        # Cancel via HL client
        success = self.exchange_client.cancel_order(
            symbol=symbol,
            strategy_order_id=order_id,
        )
        
        if success:
            logger.info(f"Order cancelled successfully - Symbol: {symbol}, ID: {order_id}")
        else:
            logger.debug(f"Order cancel had no effect (may not exist) - Symbol: {symbol}, ID: {order_id}")

    def _process_bar(self, symbol: str, bar_data: dict[str, Any]) -> None:
        """Process a new bar through the strategy for a specific symbol.

        Args:
            symbol: Trading symbol
            bar_data: Bar data with timestamp, OHLCV
        """
        timestamp = bar_data["timestamp"]
        
        # Check if we've already processed this bar for this symbol
        last_time = self.last_processed_bar_time.get(symbol)
        if last_time and timestamp <= last_time:
            logger.debug(f"Bar already processed, skipping: {symbol} {timestamp}")
            return
        
        # Append new bar to this symbol's historical data
        new_bar = pd.DataFrame([{
            "open": bar_data["open"],
            "high": bar_data["high"],
            "low": bar_data["low"],
            "close": bar_data["close"],
            "volume": bar_data["volume"],
        }], index=[timestamp])
        
        self.historical_bars[symbol] = pd.concat([self.historical_bars[symbol], new_bar])
        
        # Keep only lookback window
        if len(self.historical_bars[symbol]) > self.config.lookback_bars:
            self.historical_bars[symbol] = self.historical_bars[symbol].iloc[-self.config.lookback_bars:]
        
        # Update indicator registry with new data
        self.indicator_registries[symbol] = IndicatorRegistry(self.historical_bars[symbol])
        
        # Build strategy context
        bar_index = len(self.historical_bars[symbol]) - 1
        context = self._build_strategy_context(
            symbol=symbol,
            timestamp=timestamp,
            bar_index=bar_index,
            bar=bar_data,
        )
        
        # Execute strategy for this symbol
        logger.info(
            f"Executing strategy on new bar - "
            f"Symbol: {symbol}, Time: {timestamp}, Index: {bar_index}, "
            f"Position: {context.position_size}, Equity: ${context.equity:,.2f}"
        )
        
        try:
            self.strategies[symbol].on_bar(context)
            
            logger.info(
                f"Strategy executed successfully - "
                f"Symbol: {symbol}, Time: {timestamp}, Position: {context.position_size}"
            )
            
        except Exception as e:
            logger.error(
                f"Strategy execution failed - Symbol: {symbol}, Time: {timestamp}, Error: {e}",
                exc_info=True,
            )
            # In production, this should trigger kill switch
            raise
        
        # Update tracking for this symbol
        self.last_processed_bar_time[symbol] = timestamp
        self.bars_processed[symbol] = self.bars_processed.get(symbol, 0) + 1
        self.total_bars_processed += 1

    def run(self, strategy_cls: type[BaseStrategy], strategy_params: dict[str, Any]) -> None:
        """Run the live execution engine for multiple symbols.

        Args:
            strategy_cls: Strategy class to instantiate per symbol
            strategy_params: Parameters to pass to strategy constructor
        """
        self.running = True
        
        logger.info(
            f"Starting multi-symbol live execution engine - "
            f"Strategy: {strategy_cls.name}, Symbols: {len(self.symbols)}, "
            f"Resolution: {self.config.bar_resolution}, "
            f"Mode: {'DRY RUN' if self.config.dry_run else 'LIVE'}"
        )
        
        # Initialize each symbol independently
        for symbol in self.symbols:
            logger.info(f"Initializing {symbol}...")
            
            # Create independent strategy instance for this symbol
            self.strategies[symbol] = strategy_cls(**strategy_params)
            self.strategies[symbol].prepare()
            
            # Fetch historical bars for this symbol
            bars = self._fetch_historical_bars(symbol, self.config.lookback_bars)
            
            if bars.empty:
                logger.error(f"Failed to fetch historical bars for {symbol}, skipping")
                continue
            
            self.historical_bars[symbol] = bars
            self.indicator_registries[symbol] = IndicatorRegistry(bars)
            self.last_processed_bar_time[symbol] = bars.index[-1]
            self.bars_processed[symbol] = 0
            
            logger.info(
                f"{symbol} initialized - Bars: {len(bars)}, "
                f"Last bar: {self.last_processed_bar_time[symbol]}"
            )
        
        # Verify at least one symbol initialized
        if not self.strategies:
            logger.error("No symbols initialized successfully, cannot start")
            raise RuntimeError("No symbols available")
        
        # Position reconciliation on startup
        self._reconcile_startup_positions()
        
        # Main execution loop - poll all symbols
        try:
            while self.running:
                # Poll each symbol for new bars
                for symbol in self.symbols:
                    if symbol not in self.strategies:
                        continue  # Skip if failed to initialize
                    
                    latest_bar = self._get_latest_bar(symbol)
                    
                    if latest_bar:
                        self._process_bar(symbol, latest_bar)
                
                # Sleep once per full iteration (not per symbol)
                logger.debug(f"Polled {len(self.symbols)} symbols, sleeping for {self.config.poll_interval}s")
                time.sleep(self.config.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping gracefully")
            self.stop()
        except Exception as e:
            logger.error(f"Fatal error in execution loop: {e}", exc_info=True)
            self.stop()
            raise

    def _reconcile_startup_positions(self) -> None:
        """Check for existing positions on startup and reconcile for all symbols.
        
        Warns if positions exist (could be from previous crashed session).
        """
        if self.config.dry_run:
            # Skip reconciliation in dry-run mode
            return
        
        logger.info(f"Checking for existing positions on {len(self.symbols)} symbols")
        
        positions_found = []
        
        for symbol in self.symbols:
            if symbol not in self.strategies:
                continue  # Skip if failed to initialize
            
            recon = self.exchange_client.reconcile_position(
                symbol=symbol,
                expected_size=0.0,  # Expect flat on startup
                action_on_mismatch="warn",  # Just warn, don't auto-flatten
            )
            
            if not recon["match"]:
                positions_found.append((symbol, recon['hl_position']))
        
        if positions_found:
            logger.warning(
                f"\n{'='*60}\n"
                f"⚠️  POSITIONS EXIST ON STARTUP\n" +
                "\n".join([f"   {sym}: {size}" for sym, size in positions_found]) +
                f"\n   These may be from a previous session.\n"
                f"   Strategies will manage from current state.\n"
                f"   To flatten, use: python -m apps.live kill-switch <address> --symbol <SYMBOL>\n"
                f"{'='*60}\n"
            )

    def stop(self) -> None:
        """Stop the live execution engine."""
        logger.info("Stopping live execution engine")
        self.running = False
        
        # Finalize all strategies
        for symbol, strategy in self.strategies.items():
            summary = strategy.finalize()
            bars_count = self.bars_processed.get(symbol, 0)
            logger.info(f"{symbol} finalized - Bars: {bars_count}, Summary: {summary}")
        
        logger.info(
            f"Live execution engine stopped - "
            f"Total bars processed: {self.total_bars_processed} across {len(self.symbols)} symbols"
        )

