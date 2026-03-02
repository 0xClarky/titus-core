"""Binance market data feed with caching + pagination."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import warnings

from titus_core.data.cache import ParquetDataCache
from titus_core.data.feed import BarDataRequest, MarketDataFeed, OIDataRequest, OIDataFeed
from titus_core.data.validation import validate_bars

MAX_KLINE_BATCH = 1500  # Binance max limit for klines


class BinanceClient:
    """Thin wrapper around Binance Futures API."""

    BASE_URL = "https://fapi.binance.com/fapi/v1"
    KLINE_URL = f"{BASE_URL}/klines"
    OI_URL = f"{BASE_URL}/openInterestHist"
    EXCHANGE_INFO_URL = f"{BASE_URL}/exchangeInfo"

    def __init__(self) -> None:
        self.session = requests.Session()
        self._tick_size_cache: dict[str, float] = {}

    def get_kline(
        self,
        symbol: str,
        interval: str,
        start_ms: Optional[int],
        end_ms: Optional[int],
        limit: int = 1500,
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = self.session.get(self.KLINE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            # Binance errors usually return a dict with "code" and "msg"
            raise RuntimeError(f"Binance API error: {data}")

        if not data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rows = []
        for entry in data:
            # [Opentime, Open, High, Low, Close, Volume, CloseTime, ...]
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

    def get_open_interest(
        self,
        symbol: str,
        period: str,
        start_ms: Optional[int],
        end_ms: Optional[int],
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch open interest data from Binance."""
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = self.session.get(self.OI_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            raise RuntimeError(f"Binance OI API error: {data}")

        if not data:
            return pd.DataFrame(columns=["timestamp", "open_interest"])

        rows = []
        for entry in data:
            # {"symbol": "BTCUSDT", "sumOpenInterest": "...", "sumOpenInterestValue": "...", "timestamp": ...}
            ts = datetime.fromtimestamp(float(entry["timestamp"]) / 1000.0, tz=timezone.utc)
            rows.append(
                {
                    "timestamp": ts,
                    "open_interest": float(entry["sumOpenInterest"]),
                }
            )

        frame = pd.DataFrame(rows)
        frame = frame.sort_values("timestamp")
        frame.set_index("timestamp", inplace=True)
        return frame

    def get_tick_size(self, symbol: str) -> float | None:
        """Get tick size for a symbol from exchange info."""
        if symbol in self._tick_size_cache:
            return self._tick_size_cache[symbol]

        try:
            resp = self.session.get(self.EXCHANGE_INFO_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            symbols = data.get("symbols", [])
            for s in symbols:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "PRICE_FILTER":
                            tick_size = float(f["tickSize"])
                            self._tick_size_cache[symbol] = tick_size
                            return tick_size
            return None
        except Exception:
            return None


class BinanceMarketDataFeed(MarketDataFeed):
    """Market data feed backed by Binance klines with Parquet caching."""

    def __init__(
        self,
        cache: Optional[ParquetDataCache] = None,
        client: Optional[BinanceClient] = None,
    ) -> None:
        self.cache = cache or ParquetDataCache(root=Path("results/cache"))
        self.client = client or BinanceClient()

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
        
        # Binance often has small gaps due to maintenance. 
        # We reindex to the full expected grid and forward-fill to prevent validation errors.
        if not combined.empty:
            freq = self._map_pandas_freq(request.resolution)
            if freq:
                full_idx = pd.date_range(start=combined.index[0], end=combined.index[-1], freq=freq)
                combined = combined.reindex(full_idx)
                # Forward fill close, then use close to fill others (flat bar)
                combined['close'] = combined['close'].ffill()
                combined['open'] = combined['open'].fillna(combined['close'])
                combined['high'] = combined['high'].fillna(combined['close'])
                combined['low'] = combined['low'].fillna(combined['close'])
                combined['volume'] = combined['volume'].fillna(0)

        combined = validate_bars(combined, request.resolution)

        if request.use_cache and not combined.empty:
            self.cache.store(cache_key, combined)

        return combined.loc[request.start : request.end]
        
    @staticmethod
    def _map_pandas_freq(resolution: str) -> str | None:
        """Map resolution to pandas frequency string."""
        if resolution.endswith("h"):
            return f"{resolution[:-1]}h"  # 'h' is the new standard, 'H' is deprecated
        if resolution.endswith("m"):
            return f"{resolution[:-1]}min"
        if resolution.endswith("d"):
            return f"{resolution[:-1]}D"
        return None

    def _fetch_remote(self, request: BarDataRequest) -> pd.DataFrame:
        interval = self._map_interval(request.resolution)
        delta = self._interval_delta(request.resolution)
        rows = []
        
        # Clamp end time to now to avoid API errors for future data
        now = datetime.now(timezone.utc)
        current_end = min(request.end, now)
        
        symbol = self._normalize_symbol(request.symbol)
        last_earliest = None
        
        try:
            while current_end > request.start:
                start_batch = max(request.start, current_end - delta * MAX_KLINE_BATCH)
                
                try:
                    chunk = self.client.get_kline(
                        symbol=symbol,
                        interval=interval,
                        start_ms=int(start_batch.timestamp() * 1000),
                        end_ms=int(current_end.timestamp() * 1000),
                        limit=MAX_KLINE_BATCH,
                    )
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code in [400, 404]:
                        warnings.warn(f"Binance API stopped returning data at {current_end}: {e}")
                        break
                    raise
                    
                if chunk.empty:
                    current_end = start_batch - delta
                    continue
                rows.append(chunk)
                earliest = chunk.index[0]
                if last_earliest is not None and earliest >= last_earliest:
                    raise RuntimeError("Binance pagination stalled.")
                last_earliest = earliest
                if earliest <= request.start:
                    break
                current_end = earliest - delta
        except Exception as e:
            if not rows and not request.use_cache:
                # Only raise if we have absolutely no data
                raise RuntimeError(f"Failed to fetch Binance data: {e}") from e
            warnings.warn(f"Binance fetch interrupted: {e}. Returning partial data.")
            
        if not rows:
            # Try to return empty frame compliant with schema
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")

        frame = pd.concat(rows).sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        
        # Binance often has small gaps due to maintenance. 
        # We reindex to the full expected grid and forward-fill to prevent validation errors.
        if not frame.empty:
            freq = self._map_pandas_freq(request.resolution)
            if freq:
                full_idx = pd.date_range(start=frame.index[0], end=frame.index[-1], freq=freq)
                frame = frame.reindex(full_idx)
                # Forward fill close, then use close to fill others (flat bar)
                frame['close'] = frame['close'].ffill()
                frame['open'] = frame['open'].fillna(frame['close'])
                frame['high'] = frame['high'].fillna(frame['close'])
                frame['low'] = frame['low'].fillna(frame['close'])
                frame['volume'] = frame['volume'].fillna(0)

        # Allow partial data if validation fails due to missing head/tail, relying on backtest engine to handle
        # But here we call validate which checks gaps inside
        frame = validate_bars(frame, request.resolution)
        
        if frame.index[0] > request.start:
            warnings.warn(
                f"Binance data begins at {frame.index[0].isoformat()} which is later than requested start {request.start.isoformat()}.",
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
            return "1d"
        if res.endswith("h"):
            return res  # Binance supports 1h, 2h, 4h directly
        if res.endswith("m"):
            return res
        if res.isdigit():
            return f"{res}m"
        raise ValueError(f"Unsupported Binance interval for resolution '{resolution}'")

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        # Binance expects BTCUSDT, remove suffixes if any
        return symbol.upper().replace(".P", "").replace("/", "")

    @staticmethod
    def _cache_key(request: BarDataRequest) -> str:
        return f"binance_{request.symbol}_{request.resolution}".lower()


MAX_OI_BATCH = 500  # Binance OI endpoint max limit


class BinanceOIDataFeed(OIDataFeed):
    """Open Interest data feed backed by Binance with Parquet caching."""

    def __init__(
        self,
        cache: Optional[ParquetDataCache] = None,
        client: Optional[BinanceClient] = None,
    ) -> None:
        self.cache = cache or ParquetDataCache(root=Path("results/cache/oi"))
        self.client = client or BinanceClient()

    def get_oi(self, request: OIDataRequest) -> pd.DataFrame:
        cache_key = self._cache_key(request)
        cached = None
        read_cache = request.use_cache and not request.force_refresh
        
        if read_cache:
            cached = self.cache.load(cache_key)
            if cached is not None and not cached.empty:
                if cached.index[0] <= request.start and cached.index[-1] >= request.end:
                    return cached.loc[request.start : request.end]

        fetched = self._fetch_remote(request)
        frames = [fetched]
        if cached is not None and not cached.empty:
            frames.append(cached)
        stacked = pd.concat(frames).sort_index()
        combined = stacked.loc[~stacked.index.duplicated(keep="last")]

        if request.use_cache and not combined.empty:
            self.cache.store(cache_key, combined)

        return combined.loc[request.start : request.end]

    def _fetch_remote(self, request: OIDataRequest) -> pd.DataFrame:
        interval = self._map_interval(request.resolution)
        delta = self._interval_delta(request.resolution)
        rows = []
        
        # Use Forward Pagination to avoid "future date" 404 errors destroying the whole fetch
        current_start = request.start
        target_end = request.end
        
        symbol = self._normalize_symbol(request.symbol)
        last_latest = None
        
        while current_start < target_end:
            # Calculate batch end
            # Binance limits returned items to 'limit' (500). 
            # We estimate the time window covered by 500 items to avoid asking for too much
            end_batch = min(target_end, current_start + delta * MAX_OI_BATCH)
            
            try:
                chunk = self.client.get_open_interest(
                    symbol=symbol,
                    period=interval,
                    start_ms=int(current_start.timestamp() * 1000),
                    end_ms=int(end_batch.timestamp() * 1000),
                    limit=MAX_OI_BATCH,
                )
            except requests.exceptions.HTTPError as e:
                # 404 or 400 usually means we hit the edge of available data (or future)
                if e.response.status_code in [400, 404]:
                    warnings.warn(f"Binance OI API stopped at {current_start}: {e}")
                    break
                raise
                
            if chunk.empty:
                # If valid range returns empty, we might have a gap or reached the end
                # Try advancing purely by time
                current_start = end_batch
                continue
                
            rows.append(chunk)
            latest = chunk.index[-1]
            
            if last_latest is not None and latest <= last_latest:
                # Stalled
                break
            last_latest = latest
            
            # Next batch starts after the last received item
            # We add a small buffer or just use the next expected slot
            # Safe way: start from latest + 1ms effectively
            if latest >= target_end:
                break
                
            # Advance start pointer
            # Note: Binance might include the start time item again if we are not careful,
            # but we remove duplicates at the end.
            current_start = max(latest + timedelta(milliseconds=1), current_start + delta)

        if not rows:
             warnings.warn(
                 f"Binance OI returned no data for {request.symbol} in requested range.",
                 RuntimeWarning,
                 stacklevel=2,
             )
             return pd.DataFrame(columns=["open_interest"])
             
        frame = pd.concat(rows).sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        
        if frame.index[0] > request.start:
            warnings.warn(
                f"Binance OI data begins at {frame.index[0].isoformat()} which is later than requested start {request.start.isoformat()}.",
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
            return "1d"
        if res.endswith("h"):
            return res
        if res.endswith("m"):
            return res
        if res.isdigit():
            return f"{res}m"
        raise ValueError(f"Unsupported Binance OI interval for resolution '{resolution}'")

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.upper().replace(".P", "").replace("/", "")

    @staticmethod
    def _cache_key(request: OIDataRequest) -> str:
        return f"binance_oi_{request.symbol}_{request.resolution}".lower()
