"""Factory helpers to build data requests/feeds from config."""

from __future__ import annotations

import logging
from pathlib import Path

from titus_core.data.cache import ParquetDataCache
from titus_core.data.feed import BarDataRequest, MarketDataFeed
from titus_core.data.bybit import BybitClient, BybitMarketDataFeed
from titus_core.data.hyperliquid import HyperLiquidClient, HyperLiquidMarketDataFeed
from titus_core.utils.config import BacktestConfig, DataConfig, EngineConfig

CACHE_ROOT = Path("results/cache")
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
    elif source == "hyperliquid":
        testnet = config.data.exchange.upper() == "HYPERLIQUID_TESTNET"
        return HyperLiquidMarketDataFeed(
            cache=ParquetDataCache(CACHE_ROOT),
            testnet=testnet,
        )
    else:
        raise ValueError(
            f"Unsupported data source '{config.data.source}'. "
            f"Supported sources: 'bybit', 'hyperliquid'"
        )


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
    
    elif exchange_upper in ("HYPERLIQUID", "HYPERLIQUID_TESTNET"):
        try:
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
