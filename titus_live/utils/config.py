"""Configuration models for live execution."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from titus_core.utils.config import StrategyConfig


class LiveExecutionConfig(BaseModel):
    """Configuration for live execution on HyperLiquid.
    
    Supports both single-symbol and multi-symbol execution:
    - Single: symbol: "BTC"
    - Multi: symbols: ["BTC", "ETH", "SOL"]
    """

    id: str = Field(description="Unique identifier for this live config")
    
    # Strategy configuration
    strategy: StrategyConfig = Field(description="Strategy configuration")
    
    # Trading parameters (support both single and multi-symbol)
    symbol: Optional[str] = Field(
        default=None,
        description="Single trading symbol (e.g., 'BTC') - for backward compatibility"
    )
    symbols: Optional[list[str]] = Field(
        default=None,
        description="Multiple trading symbols (e.g., ['BTC', 'ETH', 'SOL']) - for multi-asset"
    )
    bar_resolution: str = Field(default="4h", description="Bar resolution (e.g., '1h', '4h', '1d')")
    
    # Exchange selection
    exchange: Optional[str] = Field(
        default=None,
        description="Exchange to use: 'hyperliquid' or 'bybit' (default: hyperliquid if not specified)"
    )
    
    # Capital and risk
    # Note: Live equity is fetched from exchange on each bar
    # This is just for reference/fallback
    commission: float = Field(default=0.00055, description="Commission rate (e.g., 0.00055 = 0.055%)")
    max_position_size: float = Field(
        default=0.0,
        description="Maximum position size in USD notional per symbol (0 = no limit)",
    )
    max_leverage: float = Field(default=5.0, description="Maximum leverage multiplier")
    size_multiplier: float = Field(
        default=1.0,
        description="Global position size multiplier (0.5=half size, 2.0=double size)"
    )
    
    # Engine settings
    poll_interval: int = Field(
        default=60,
        description="Seconds between bar checks (default: 60)",
    )
    lookback_bars: int = Field(
        default=200,
        description="Number of historical bars to load for indicators",
    )
    
    @model_validator(mode="after")
    def validate_symbols(self) -> "LiveExecutionConfig":
        """Ensure either symbol or symbols is provided, normalize to symbols list."""
        if self.symbol is None and self.symbols is None:
            raise ValueError("Must provide either 'symbol' or 'symbols'")
        
        if self.symbol is not None and self.symbols is not None:
            raise ValueError("Cannot provide both 'symbol' and 'symbols' - use one or the other")
        
        # Normalize single symbol to list for internal consistency
        if self.symbol is not None:
            self.symbols = [self.symbol]
            self.symbol = None  # Clear to avoid confusion
        
        return self
    
    def get_symbols(self) -> list[str]:
        """Get list of symbols (normalized accessor)."""
        return self.symbols or []

