"""Core data-layer interfaces for Titus."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class BarDataRequest:
    """Normalized request for OHLCV data."""

    symbol: str
    exchange: str
    start: datetime
    end: datetime
    resolution: str
    use_cache: bool = True
    force_refresh: bool = False


class MarketDataFeed(ABC):
    """Abstract TradingView-compatible data source."""

    @abstractmethod
    def get_bars(self, request: BarDataRequest) -> pd.DataFrame:
        """Return OHLCV bars indexed by timezone-aware timestamps."""


class DataCache(Protocol):
    """Protocol for cache implementations used by data feeds."""

    def load(self, key: str) -> pd.DataFrame | None:  # pragma: no cover - interface only
        ...

    def store(self, key: str, frame: pd.DataFrame) -> None:  # pragma: no cover - interface only
        ...
