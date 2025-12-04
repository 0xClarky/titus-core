"""Bybit market data feed with caching + pagination."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import warnings

from titus_core.data.cache import ParquetDataCache
from titus_core.data.feed import BarDataRequest, MarketDataFeed
from titus_core.data.validation import validate_bars

MAX_KLINE_BATCH = 1000


class BybitClient:
    """Thin wrapper around Bybit v5 market endpoints."""

    BASE_URL = "https://api.bybit.com/v5/market"
    KLINE_URL = f"{BASE_URL}/kline"
    INSTRUMENTS_URL = f"{BASE_URL}/instruments-info"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None) -> None:
        self.session = requests.Session()
        self.api_key = api_key
        self.api_secret = api_secret
        self._tick_size_cache: dict[str, float] = {}

    def get_kline(
        self,
        symbol: str,
        interval: str,
        start_ms: Optional[int],
        end_ms: Optional[int],
        limit: int = 200,
    ) -> pd.DataFrame:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["start"] = start_ms
        if end_ms is not None:
            params["end"] = end_ms

        resp = self.session.get(self.KLINE_URL, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error: {payload}")

        data = payload.get("result", {}).get("list", [])
        if not data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rows = []
        for entry in data:
            ts = datetime.fromtimestamp(float(entry[0]) / 1000.0, tz=timezone.utc)
            rows.append(
                {
                    "timestamp": ts,
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                }
            )

        frame = pd.DataFrame(rows)
        frame = frame.sort_values("timestamp")
        frame.set_index("timestamp", inplace=True)
        return frame

    def get_instrument_info(self, symbol: str, category: str = "linear") -> dict | None:
        """Get instrument metadata including tick size from Bybit API.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            category: Instrument category, default "linear" for futures
            
        Returns:
            Instrument info dict with 'tickSize', or None if not found
        """
        # Check cache first
        cache_key = f"{category}:{symbol.upper()}"
        if cache_key in self._tick_size_cache:
            return {"tickSize": self._tick_size_cache[cache_key]}
        
        params = {
            "category": category,
            "symbol": symbol.upper(),
        }
        
        try:
            resp = self.session.get(self.INSTRUMENTS_URL, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            
            if payload.get("retCode") != 0:
                return None
                
            result = payload.get("result", {})
            instruments = result.get("list", [])
            
            if not instruments:
                return None
                
            # Get first matching instrument
            instrument = instruments[0]
            
            # Extract tick size from priceFilter (direct field, not in filters array)
            price_filter = instrument.get("priceFilter")
            if price_filter and "tickSize" in price_filter:
                tick_size = price_filter.get("tickSize")
                if tick_size:
                    tick_size_float = float(tick_size)
                    # Cache it
                    self._tick_size_cache[cache_key] = tick_size_float
                    return {"tickSize": tick_size_float}
                        
            return None
        except Exception:
            # Return None on any error (network, parsing, etc.)
            return None

    def get_tick_size(self, symbol: str, category: str = "linear") -> float | None:
        """Get tick size for a symbol. Returns None if unavailable.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT" or "BTCUSDT.P")
            category: Instrument category, default "linear" for futures
            
        Returns:
            Tick size as float, or None if not available
        """
        # Normalize symbol (remove .P suffix if present)
        normalized = symbol.upper().replace(".P", "")
        
        info = self.get_instrument_info(normalized, category)
        if info and "tickSize" in info:
            return info["tickSize"]
        return None

    def get_instrument_info(self, symbol: str, category: str = "linear") -> dict | None:
        """Get instrument metadata including tick size from Bybit API.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            category: Instrument category, default "linear" for futures
            
        Returns:
            Instrument info dict with 'tickSize' in filters, or None if not found
        """
        # Check cache first
        cache_key = f"{category}:{symbol.upper()}"
        if cache_key in self._tick_size_cache:
            return {"tickSize": self._tick_size_cache[cache_key]}
        
        params = {
            "category": category,
            "symbol": symbol.upper(),
        }
        
        try:
            resp = self.session.get(self.INSTRUMENTS_URL, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            
            if payload.get("retCode") != 0:
                return None
                
            result = payload.get("result", {})
            instruments = result.get("list", [])
            
            if not instruments:
                return None
                
            # Get first matching instrument
            instrument = instruments[0]
            
            # Extract tick size from priceFilter (direct field, not in filters array)
            price_filter = instrument.get("priceFilter")
            if price_filter and "tickSize" in price_filter:
                tick_size = price_filter.get("tickSize")
                if tick_size:
                    tick_size_float = float(tick_size)
                    # Cache it
                    self._tick_size_cache[cache_key] = tick_size_float
                    return {"tickSize": tick_size_float}
                        
            return None
        except Exception:
            # Return None on any error (network, parsing, etc.)
            return None

    def get_tick_size(self, symbol: str, category: str = "linear") -> float | None:
        """Get tick size for a symbol. Returns None if unavailable.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT" or "BTCUSDT.P")
            category: Instrument category, default "linear" for futures
            
        Returns:
            Tick size as float, or None if not available
        """
        # Normalize symbol (remove .P suffix if present)
        normalized = symbol.upper().replace(".P", "")
        
        info = self.get_instrument_info(normalized, category)
        if info and "tickSize" in info:
            return info["tickSize"]
        return None


class BybitMarketDataFeed(MarketDataFeed):
    """Market data feed backed by Bybit klines with Parquet caching."""

    def __init__(
        self,
        cache: Optional[ParquetDataCache] = None,
        client: Optional[BybitClient] = None,
    ) -> None:
        self.cache = cache or ParquetDataCache(root=Path("results/cache"))
        self.client = client or BybitClient()

    def get_bars(self, request: BarDataRequest) -> pd.DataFrame:
        cache_key = self._cache_key(request)
        cached = None
        read_cache = request.use_cache and not request.force_refresh
        if read_cache:
            cached = self.cache.load(cache_key)
            if cached is not None and not cached.empty:
                cached = validate_bars(cached, request.resolution)
                if cached.index[0] <= request.start and cached.index[-1] >= request.end:
                    return cached.loc[request.start : request.end]

        fetched = self._fetch_remote(request)
        frames = [fetched]
        if cached is not None and not cached.empty:
            frames.append(cached)
        stacked = pd.concat(frames).sort_index()
        combined = stacked.loc[~stacked.index.duplicated(keep="last")]
        combined = validate_bars(combined, request.resolution)

        if request.use_cache and not combined.empty:
            self.cache.store(cache_key, combined)

        return combined.loc[request.start : request.end]

    def _fetch_remote(self, request: BarDataRequest) -> pd.DataFrame:
        interval = self._map_interval(request.resolution)
        delta = self._interval_delta(request.resolution)
        rows = []
        current_end = request.end
        symbol = self._normalize_symbol(request.symbol)
        last_earliest = None
        while current_end > request.start:
            start_batch = max(request.start, current_end - delta * MAX_KLINE_BATCH)
            chunk = self.client.get_kline(
                symbol=symbol,
                interval=interval,
                start_ms=int(start_batch.timestamp() * 1000),
                end_ms=int(current_end.timestamp() * 1000),
                limit=MAX_KLINE_BATCH,
            )
            if chunk.empty:
                current_end = start_batch - delta
                continue
            rows.append(chunk)
            earliest = chunk.index[0]
            if last_earliest is not None and earliest >= last_earliest:
                raise RuntimeError("Bybit pagination stalled (no progress toward requested start).")
            last_earliest = earliest
            if earliest <= request.start:
                break
            current_end = earliest - delta
        if not rows:
            raise RuntimeError("Bybit returned no data for requested range")
        frame = pd.concat(rows).sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        if frame.index[0] > request.start:
            warnings.warn(
                f"Bybit data begins at {frame.index[0].isoformat()} which is later than requested start {request.start.isoformat()}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return frame

    @staticmethod
    def _interval_delta(resolution: str) -> timedelta:
        if resolution.endswith("m"):
            return timedelta(minutes=int(resolution[:-1]))
        if resolution.endswith("h"):
            return timedelta(hours=int(resolution[:-1]))
        if resolution.endswith("d"):
            return timedelta(days=int(resolution[:-1]))
        if resolution.isdigit():
            return timedelta(minutes=int(resolution))
        raise ValueError(f"Unsupported resolution for delta: {resolution}")

    @staticmethod
    def _map_interval(resolution: str) -> str:
        res = resolution.lower()
        if res in {"1d", "1day", "d"}:
            return "D"
        if res.endswith("h"):
            return str(int(res[:-1]) * 60)
        if res.endswith("m"):
            return res[:-1]
        if res.isdigit():
            return res
        raise ValueError(f"Unsupported Bybit interval for resolution '{resolution}'")

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if symbol.upper().endswith(".P"):
            return symbol.upper().replace(".P", "")
        return symbol.upper()

    @staticmethod
    def _cache_key(request: BarDataRequest) -> str:
        return f"bybit_{request.symbol}_{request.resolution}".lower()
