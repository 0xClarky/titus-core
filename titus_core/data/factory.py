"""Factory helpers to build data requests/feeds from config."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from titus_core.data.cache import ParquetDataCache
from titus_core.data.feed import BarDataRequest, MarketDataFeed, OIDataRequest, OIDataFeed
from titus_core.data.bybit import BybitClient, BybitMarketDataFeed, BybitOIDataFeed
from titus_core.utils.config import BacktestConfig, DataConfig, EngineConfig, AuxiliaryDataConfig

CACHE_ROOT = Path("results/cache")
OI_CACHE_ROOT = Path("results/cache/oi")
logger = logging.getLogger(__name__)


def build_bar_request(config: DataConfig) -> BarDataRequest:
    return BarDataRequest(
        symbol=config.symbol,
        exchange=config.exchange,
        start=config.start,
        end=config.end,
        resolution=config.resolution,
        use_cache=config.use_cache,
        force_refresh=config.force_refresh,
    )


def build_oi_request(data_config: DataConfig, aux_config: AuxiliaryDataConfig) -> OIDataRequest:
    """Build OI data request from config."""
    resolution = aux_config.oi_resolution or data_config.resolution
    return OIDataRequest(
        symbol=data_config.symbol,
        exchange=data_config.exchange,
        start=data_config.start,
        end=data_config.end,
        resolution=resolution,
        use_cache=data_config.use_cache,
        force_refresh=data_config.force_refresh,
    )


def build_market_data_feed(config: BacktestConfig) -> MarketDataFeed:
    """Build market data feed from config.

    Args:
        config: Backtest configuration

    Returns:
        MarketDataFeed instance

    Raises:
        ValueError: If data source is unsupported
    """
    source = config.data.source.lower()
    
    if source == "bybit":
        return BybitMarketDataFeed(cache=ParquetDataCache(CACHE_ROOT))
    elif source == "binance":
        from titus_core.data.binance import BinanceMarketDataFeed
        return BinanceMarketDataFeed(cache=ParquetDataCache(CACHE_ROOT))
    elif source == "hyperliquid":
        # Lazy import to avoid requiring hyperliquid package when using Bybit
        from titus_core.data.hyperliquid import HyperLiquidMarketDataFeed
        testnet = config.data.exchange.upper() == "HYPERLIQUID_TESTNET"
        return HyperLiquidMarketDataFeed(
            cache=ParquetDataCache(CACHE_ROOT),
            testnet=testnet,
        )
    else:
        raise ValueError(
            f"Unsupported data source '{config.data.source}'. "
            f"Supported sources: 'bybit', 'binance', 'hyperliquid'"
        )


def build_oi_data_feed(aux_config: AuxiliaryDataConfig) -> OIDataFeed:
    """Build OI data feed from config.

    Args:
        aux_config: Auxiliary data configuration

    Returns:
        OIDataFeed instance

    Raises:
        ValueError: If OI source is unsupported
    """
    source = aux_config.oi_source.lower()
    
    if source == "bybit":
        return BybitOIDataFeed(cache=ParquetDataCache(OI_CACHE_ROOT))
    elif source == "binance":
        from titus_core.data.binance import BinanceOIDataFeed
        return BinanceOIDataFeed(cache=ParquetDataCache(OI_CACHE_ROOT))
    else:
        raise ValueError(
            f"Unsupported OI data source '{aux_config.oi_source}'. "
            f"Supported sources: 'bybit', 'binance'"
        )


def load_merged_bars_with_oi(
    config: BacktestConfig,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Merge OI data with price bars if auxiliary_data.oi_enabled is True.

    Args:
        config: Backtest configuration
        bars: Price DataFrame (OHLCV)

    Returns:
        DataFrame with 'open_interest' column added if OI enabled, else original bars
    """
    if config.auxiliary_data is None or not config.auxiliary_data.oi_enabled:
        return bars

    aux_config = config.auxiliary_data
    oi_feed = build_oi_data_feed(aux_config)
    oi_request = build_oi_request(config.data, aux_config)
    
    try:
        oi_data = oi_feed.get_oi(oi_request)
        if oi_data.empty:
            logger.warning("OI data is empty, proceeding without OI")
            bars["open_interest"] = float("nan")
            return bars
        
        # Merge OI with bars
        # OI index may not align perfectly with bars, use nearest or forward-fill
        merged = bars.copy()
        merged = merged.join(oi_data, how="left")
        merged["open_interest"] = merged["open_interest"].ffill()
        
        logger.info(f"Merged {len(oi_data)} OI bars with {len(bars)} price bars")
        return merged
        
    except Exception as e:
        logger.warning(f"Failed to fetch OI data: {e}. Proceeding without OI.")
        bars["open_interest"] = float("nan")
        return bars



def auto_populate_tick_size(engine_config: EngineConfig, symbol: str, exchange: str = "BYBIT") -> EngineConfig:
    """Auto-populate tick_size from exchange API if not set in config.
    
    Args:
        engine_config: Engine configuration (may have tick_size=None)
        symbol: Trading symbol (e.g., "BTCUSDT.P" for Bybit, "BTC" for HL)
        exchange: Exchange name (BYBIT or HYPERLIQUID)
        
    Returns:
        EngineConfig with tick_size populated if available from exchange
        
    Raises:
        ValueError: If tick_size is required (slippage_type='ticks') but cannot be fetched
    """
    # If tick_size is already set, don't override
    if engine_config.tick_size is not None:
        return engine_config
    
    # Check if tick_size is required
    requires_tick_size = engine_config.slippage_type == "ticks" and engine_config.slippage > 0
    
    exchange_upper = exchange.upper()
    
    # Try to auto-detect from exchange API
    if exchange_upper == "BYBIT":
        try:
            client = BybitClient()
            tick_size = client.get_tick_size(symbol)
            if tick_size is not None:
                config_dict = engine_config.model_dump()
                config_dict["tick_size"] = tick_size
                logger.info(f"Auto-detected tick_size={tick_size} for {symbol} from Bybit API")
                return EngineConfig(**config_dict)
            else:
                if requires_tick_size:
                    raise ValueError(
                        f"Could not fetch tick_size for {symbol} from Bybit API, "
                        f"but tick_size is required when slippage_type='ticks' and slippage > 0. "
                        f"Please set tick_size manually in config."
                    )
                logger.warning(f"Could not fetch tick_size for {symbol} from Bybit API. Using default.")
        except ValueError:
            raise
        except Exception as e:
            if requires_tick_size:
                raise ValueError(
                    f"Failed to auto-detect tick_size for {symbol}: {e}. "
                    f"tick_size is required when slippage_type='ticks' and slippage > 0. "
                    f"Please set tick_size manually in config."
                ) from e
            logger.warning(f"Failed to auto-detect tick_size for {symbol}: {e}. Using default.")
    
    elif exchange_upper == "BINANCE":
        try:
            from titus_core.data.binance import BinanceClient
            client = BinanceClient()
            tick_size = client.get_tick_size(symbol)
            if tick_size is not None:
                config_dict = engine_config.model_dump()
                config_dict["tick_size"] = tick_size
                logger.info(f"Auto-detected tick_size={tick_size} for {symbol} from Binance API")
                return EngineConfig(**config_dict)
            else:
                if requires_tick_size:
                    raise ValueError(
                        f"Could not fetch tick_size for {symbol} from Binance API, "
                        f"but tick_size is required when slippage_type='ticks' and slippage > 0. "
                        f"Please set tick_size manually in config."
                    )
                logger.warning(f"Could not fetch tick_size for {symbol} from Binance API. Using default.")
        except ValueError:
            raise
        except Exception as e:
            if requires_tick_size:
                raise ValueError(
                    f"Failed to auto-detect tick_size for {symbol}: {e}. "
                    f"tick_size is required when slippage_type='ticks' and slippage > 0. "
                    f"Please set tick_size manually in config."
                ) from e
            logger.warning(f"Failed to auto-detect tick_size for {symbol}: {e}. Using default.")

    elif exchange_upper in ("HYPERLIQUID", "HYPERLIQUID_TESTNET"):
        try:
            # Lazy import to avoid requiring hyperliquid package when using Bybit
            from titus_core.data.hyperliquid import HyperLiquidClient
            testnet = exchange_upper == "HYPERLIQUID_TESTNET"
            client = HyperLiquidClient(testnet=testnet)
            tick_size = client.get_tick_size(symbol)
            if tick_size is not None:
                config_dict = engine_config.model_dump()
                config_dict["tick_size"] = tick_size
                logger.info(f"Auto-detected tick_size={tick_size} for {symbol} from HyperLiquid API")
                return EngineConfig(**config_dict)
            else:
                if requires_tick_size:
                    raise ValueError(
                        f"Could not fetch tick_size for {symbol} from HyperLiquid API, "
                        f"but tick_size is required when slippage_type='ticks' and slippage > 0. "
                        f"Please set tick_size manually in config."
                    )
                logger.warning(f"Could not fetch tick_size for {symbol} from HyperLiquid API. Using default.")
        except ValueError:
            raise
        except Exception as e:
            if requires_tick_size:
                raise ValueError(
                    f"Failed to auto-detect tick_size for {symbol}: {e}. "
                    f"tick_size is required when slippage_type='ticks' and slippage > 0. "
                    f"Please set tick_size manually in config."
                ) from e
            logger.warning(f"Failed to auto-detect tick_size for {symbol}: {e}. Using default.")
    
    else:
        if requires_tick_size:
            raise ValueError(
                f"Auto-detection of tick_size not supported for exchange '{exchange}', "
                f"but tick_size is required when slippage_type='ticks' and slippage > 0. "
                f"Please set tick_size manually in config."
            )
        logger.warning(f"Auto-detection of tick_size not supported for exchange '{exchange}'. Using default.")
    
    return engine_config


def get_lot_size_for_symbol(symbol: str, exchange: str = "BYBIT") -> float:
    """Get the lot size (quantity step) for a symbol from the exchange API.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT.P" for Bybit)
        exchange: Exchange name (BYBIT, BINANCE, or HYPERLIQUID)
        
    Returns:
        Lot size as float, defaults to 0.001 if unavailable
    """
    exchange_upper = exchange.upper()
    
    if exchange_upper == "BYBIT":
        try:
            from titus_core.data.bybit import BybitClient
            client = BybitClient()
            lot_size = client.get_lot_size(symbol)
            if lot_size is not None:
                return lot_size
        except Exception:
            pass
    
    elif exchange_upper == "BINANCE":
        try:
            from titus_core.data.binance import BinanceClient
            client = BinanceClient()
            lot_size = client.get_lot_size(symbol)
            if lot_size is not None:
                return lot_size
        except Exception:
            pass
    
    # Default fallback (BTC-sized assets)
    return 0.001
