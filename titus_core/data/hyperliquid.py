"""HyperLiquid market data feed with caching."""

from __future__ import annotations

import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils import constants

from titus_core.data.cache import ParquetDataCache
from titus_core.data.feed import BarDataRequest, MarketDataFeed
from titus_core.data.validation import validate_bars

# HyperLiquid API limits
MAX_CANDLES_BATCH = 5000  # HL can return large batches


class HyperLiquidClient:
    """Thin wrapper around HyperLiquid Info API for market data."""

    def __init__(self, testnet: bool = False) -> None:
        """Initialize HyperLiquid client.

        Args:
            testnet: Use testnet API (default: False for mainnet)
        """
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(base_url, skip_ws=True)
        self.testnet = testnet
        self._tick_size_cache: dict[str, float] = {}

    def get_candles(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
    ) -> pd.DataFrame:
        """Fetch historical candles from HyperLiquid.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            interval: Bar resolution (e.g., "1h", "4h", "1d")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        try:
            candles = self.info.candles_snapshot(
                name=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
            )
        except Exception as e:
            raise RuntimeError(f"HyperLiquid API error: {e}") from e

        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert to DataFrame
        # HL returns list of dicts: {t, T, s, i, o, c, h, l, v, n}
        df = pd.DataFrame(candles)

        # Extract and rename columns
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df["open"] = pd.to_numeric(df["o"], errors="coerce")
        df["high"] = pd.to_numeric(df["h"], errors="coerce")
        df["low"] = pd.to_numeric(df["l"], errors="coerce")
        df["close"] = pd.to_numeric(df["c"], errors="coerce")
        df["volume"] = pd.to_numeric(df["v"], errors="coerce")

        # Keep only needed columns and set index
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.set_index("timestamp")
        df = df.sort_index()

        return df

    def get_meta(self) -> dict:
        """Fetch exchange metadata (symbols, tick sizes, etc.).

        Returns:
            Meta dict from HyperLiquid API
        """
        return self.info.meta()

    def get_tick_size(self, symbol: str) -> Optional[float]:
        """Get tick size for a symbol from HyperLiquid metadata.

        Args:
            symbol: Trading symbol (e.g., "BTC")

        Returns:
            Tick size or None if not available
        """
        if symbol in self._tick_size_cache:
            return self._tick_size_cache[symbol]

        try:
            meta = self.get_meta()
            universe = meta.get("universe", [])

            for asset in universe:
                if asset.get("name") == symbol:
                    # HL provides szDecimals for size precision
                    # For price, we'll use a default based on asset type
                    # TODO: Find where HL exposes price decimals
                    tick_size = 0.1  # Default for BTC-like assets

                    self._tick_size_cache[symbol] = tick_size
                    return tick_size

            return None

        except Exception:
            return None


class HyperLiquidMarketDataFeed(MarketDataFeed):
    """Market data feed backed by HyperLiquid API with Parquet caching."""

    def __init__(
        self,
        cache: Optional[ParquetDataCache] = None,
        client: Optional[HyperLiquidClient] = None,
        testnet: bool = False,
    ) -> None:
        """Initialize HyperLiquid data feed.

        Args:
            cache: Optional cache instance
            client: Optional HyperLiquid client instance
            testnet: Use testnet API (default: False)
        """
        self.cache = cache or ParquetDataCache(root=Path("results/cache"))
        self.client = client or HyperLiquidClient(testnet=testnet)

    def get_bars(self, request: BarDataRequest) -> pd.DataFrame:
        """Fetch OHLCV bars with caching.

        Args:
            request: Bar data request

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = self._cache_key(request)
        cached = None
        read_cache = request.use_cache and not request.force_refresh

        if read_cache:
            cached = self.cache.load(cache_key)
            if cached is not None and not cached.empty:
                cached = validate_bars(cached, request.resolution)
                # If cache covers the full range, return cached data
                if cached.index[0] <= request.start and cached.index[-1] >= request.end:
                    return cached.loc[request.start : request.end]

        # Fetch from HyperLiquid
        fetched = self._fetch_remote(request)

        # Merge with cache if available
        frames = [fetched]
        if cached is not None and not cached.empty:
            frames.append(cached)

        stacked = pd.concat(frames).sort_index()
        combined = stacked.loc[~stacked.index.duplicated(keep="last")]
        combined = validate_bars(combined, request.resolution)

        # Update cache
        if request.use_cache and not combined.empty:
            self.cache.store(cache_key, combined)

        return combined.loc[request.start : request.end]

    def _fetch_remote(self, request: BarDataRequest) -> pd.DataFrame:
        """Fetch data from HyperLiquid API.

        Args:
            request: Bar data request

        Returns:
            DataFrame with fetched bars
        """
        symbol = self._normalize_symbol(request.symbol)
        interval = self._map_interval(request.resolution)

        # Convert to milliseconds
        start_ms = int(request.start.timestamp() * 1000)
        end_ms = int(request.end.timestamp() * 1000)

        # HyperLiquid can handle large ranges in one call (up to 5000 bars)
        # For safety, we'll paginate if the range exceeds that
        delta = self._interval_delta(request.resolution)
        total_bars = int((request.end - request.start) / delta)

        if total_bars <= MAX_CANDLES_BATCH:
            # Fetch in one call
            chunk = self.client.get_candles(
                symbol=symbol,
                interval=interval,
                start_time=start_ms,
                end_time=end_ms,
            )
            if chunk.empty:
                raise RuntimeError("HyperLiquid returned no data for requested range")
            return chunk
        else:
            # Paginate for very large ranges
            rows = []
            current_start = request.start

            while current_start < request.end:
                batch_end = min(current_start + delta * MAX_CANDLES_BATCH, request.end)

                chunk = self.client.get_candles(
                    symbol=symbol,
                    interval=interval,
                    start_time=int(current_start.timestamp() * 1000),
                    end_time=int(batch_end.timestamp() * 1000),
                )

                if not chunk.empty:
                    rows.append(chunk)

                current_start = batch_end

            if not rows:
                raise RuntimeError("HyperLiquid returned no data for requested range")

            frame = pd.concat(rows).sort_index()
            frame = frame.loc[~frame.index.duplicated(keep="last")]

            if frame.index[0] > request.start:
                warnings.warn(
                    f"HyperLiquid data begins at {frame.index[0].isoformat()} "
                    f"which is later than requested start {request.start.isoformat()}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            return frame

    @staticmethod
    def _interval_delta(resolution: str) -> timedelta:
        """Convert resolution to timedelta.

        Args:
            resolution: Resolution string (e.g., "1h", "4h", "1d")

        Returns:
            Timedelta representing one bar
        """
        res = resolution.lower()
        if res.endswith("m"):
            return timedelta(minutes=int(res[:-1]))
        if res.endswith("h"):
            return timedelta(hours=int(res[:-1]))
        if res.endswith("d"):
            return timedelta(days=int(res[:-1]))
        if res.isdigit():
            return timedelta(minutes=int(res))
        raise ValueError(f"Unsupported resolution: {resolution}")

    @staticmethod
    def _map_interval(resolution: str) -> str:
        """Map Titus resolution to HyperLiquid interval format.

        Args:
            resolution: Titus resolution (e.g., "4h", "1d")

        Returns:
            HyperLiquid interval string
        """
        # HyperLiquid uses same format: "1m", "1h", "4h", "1d"
        res = resolution.lower()
        
        # Normalize common formats
        if res in {"1d", "1day", "d"}:
            return "1d"
        
        # HyperLiquid accepts: 1m, 15m, 1h, 4h, 1d
        # Pass through as-is
        return res

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol for HyperLiquid.

        Args:
            symbol: Raw symbol (e.g., "BTCUSDT.P" or "BTC")

        Returns:
            HyperLiquid symbol format (e.g., "BTC")
        """
        # Remove .P suffix if present (Bybit perpetual notation)
        symbol = symbol.upper().replace(".P", "")
        
        # Remove USDT suffix (HL uses just "BTC" not "BTCUSDT")
        symbol = symbol.replace("USDT", "")
        
        return symbol

    @staticmethod
    def _cache_key(request: BarDataRequest) -> str:
        """Generate cache key for request.

        Args:
            request: Bar data request

        Returns:
            Cache key string
        """
        return f"hyperliquid_{request.symbol}_{request.resolution}".lower()

