"""Validation helpers for OHLCV data."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

import pandas as pd


class DataValidationError(ValueError):
    """Raised when market data fails deterministic validation."""


def resolution_to_timedelta(resolution: str) -> Optional[pd.Timedelta]:
    if resolution.isdigit():
        return pd.to_timedelta(int(resolution), unit="m")
    match = re.fullmatch(r"(?i)(\d+)([mhd])", resolution.strip())
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    unit_map = {"m": "m", "h": "h", "d": "d"}
    return pd.to_timedelta(value, unit=unit_map[unit])


def validate_bars(frame: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Ensure dataframe meets TradingView parity requirements."""

    required_columns = {"open", "high", "low", "close"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(sorted(missing))}")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError("Bars must be indexed by pandas.DatetimeIndex.")
    if frame.index.tz is None:
        raise DataValidationError("Bars must use timezone-aware timestamps (UTC).")
    if not frame.index.is_monotonic_increasing:
        raise DataValidationError("Timestamps must be strictly increasing.")
    if frame.index.has_duplicates:
        raise DataValidationError("Duplicate timestamps detected.")

    high_cap = frame[["open", "close"]].max(axis=1)
    low_cap = frame[["open", "close"]].min(axis=1)
    if (frame["high"] < high_cap).any():
        raise DataValidationError("High price must be >= max(open, close).")
    if (frame["low"] > low_cap).any():
        raise DataValidationError("Low price must be <= min(open, close).")
    if "volume" in frame.columns and (frame["volume"] < 0).any():
        raise DataValidationError("Volume cannot be negative.")

    expected_delta = resolution_to_timedelta(resolution)
    if expected_delta is not None and len(frame) > 1:
        deltas = frame.index.to_series().diff().dropna()
        missing = deltas[deltas > expected_delta * 1.5]
        if not missing.empty:
            raise DataValidationError(
                f"Detected gaps larger than expected delta ({expected_delta})."
            )

    if frame.isna().any().any():
        raise DataValidationError("Detected NA values inside OHLCV frame.")

    return frame
