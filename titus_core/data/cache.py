"""Cache implementations for market data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from titus_core.data.feed import DataCache


class ParquetDataCache(DataCache):
    """Simple Parquet-backed cache rooted under results/cache."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        return self.root / f"{key}.parquet"

    def load(self, key: str) -> Optional[pd.DataFrame]:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"Cached frame for {key} must have DatetimeIndex.")
        return frame

    def store(self, key: str, frame: pd.DataFrame) -> None:
        path = self._path_for_key(key)
        frame.to_parquet(path)
