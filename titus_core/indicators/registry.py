"""Indicator registry mirroring Pine's ta.* semantics."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import pandas as pd

IndicatorFunc = Callable[["IndicatorRegistry", dict], pd.Series]


class IndicatorRegistry:
    """Caches indicator outputs so they are computed once per run."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars
        self._registry: Dict[str, IndicatorFunc] = {}
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], pd.Series] = {}
        self._metadata: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], dict] = {}
        self._register_builtin_indicators()

    def _register_builtin_indicators(self) -> None:
        self.register("sma", _sma)
        self.register("ema", _ema)
        self.register("rsi", _rsi)
        self.register("atr", _atr)
        self.register("adx", _adx)
        self.register("donchian", _donchian)
        self.register("highest", _highest)
        self.register("lowest", _lowest)

    def register(self, name: str, func: IndicatorFunc) -> None:
        self._registry[name] = func

    def compute(self, name: str, **params) -> pd.Series:
        if name not in self._registry:
            raise KeyError(f"Indicator '{name}' is not registered.")
        normalized_params = self._normalize_params(params)
        frozen = _freeze_params(normalized_params)
        key = (name, frozen)
        if key in self._cache:
            return self._cache[key]
        series = self._registry[name](self, normalized_params)
        self._cache[key] = series
        self._metadata[key] = normalized_params
        return series

    def metadata(self, name: str, **params) -> dict | None:
        normalized_params = self._normalize_params(params)
        key = (name, _freeze_params(normalized_params))
        return self._metadata.get(key)

    @staticmethod
    def _normalize_params(params: dict) -> dict:
        normalized = dict(params)
        source = normalized.get("source")
        if isinstance(source, str):
            normalized["source"] = source.strip()
        channel = normalized.get("channel")
        if isinstance(channel, str):
            normalized["channel"] = channel.strip().lower()
        return normalized


def _select_source(registry: IndicatorRegistry, params: dict) -> pd.Series:
    bars = registry._bars
    source = params.get("source", "close")
    if isinstance(source, dict):
        indicator_name = source.get("indicator")
        if not indicator_name:
            raise ValueError("source dict requires an 'indicator' key.")
        indicator_params = source.get("params", {})
        return registry.compute(indicator_name, **indicator_params)
    if isinstance(source, str):
        if source in bars.columns:
            return bars[source]
        lower = source.lower()
        if lower in bars.columns:
            return bars[lower]
        if lower == "hl2":
            return (bars["high"] + bars["low"]) / 2.0
        if lower == "hlc3":
            return (bars["high"] + bars["low"] + bars["close"]) / 3.0
        if lower == "ohlc4":
            return (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4.0
    if source in bars.columns:
        return bars[source]
    raise KeyError(f"Column '{source}' not present in bars or unsupported synthetic source.")


def _sma(registry: IndicatorRegistry, params: dict) -> pd.Series:
    length = params.get("length")
    if length is None:
        raise ValueError("sma requires a 'length' parameter.")
    source = _select_source(registry, params)
    return source.rolling(int(length)).mean()


def _ema(registry: IndicatorRegistry, params: dict) -> pd.Series:
    length = params.get("length")
    if length is None:
        raise ValueError("ema requires a 'length' parameter.")
    source = _select_source(registry, params)
    return _ema_series(source, int(length))


def _rma(values: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        raise ValueError("length must be positive for rma.")
    result = pd.Series(float("nan"), index=values.index, dtype=float)
    if len(values) < length:
        return result
    first_valid = length - 1
    seed = values.iloc[:length].mean()
    result.iloc[first_valid] = seed
    for i in range(first_valid + 1, len(values)):
        prev = result.iloc[i - 1]
        current = values.iloc[i]
        result.iloc[i] = prev + (current - prev) / length
    return result


def _ema_series(values: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        raise ValueError("length must be positive for ema.")
    result = pd.Series(float("nan"), index=values.index, dtype=float)
    if len(values) < length:
        return result
    first_valid = length - 1
    seed = values.iloc[:length].mean()
    result.iloc[first_valid] = seed
    alpha = 2.0 / (length + 1)
    prev = seed
    for i in range(first_valid + 1, len(values)):
        current = values.iloc[i]
        prev = prev + alpha * (current - prev)
        result.iloc[i] = prev
    return result


def _rsi(registry: IndicatorRegistry, params: dict) -> pd.Series:
    length = params.get("length")
    if length is None:
        raise ValueError("rsi requires a 'length' parameter.")
    length = int(length)
    source = _select_source(registry, params)
    delta = source.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = _rma(gains, length)
    avg_loss = _rma(losses, length)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    zero_gain = avg_gain.eq(0)
    zero_loss = avg_loss.eq(0)
    rsi = rsi.where(~zero_gain, 0.0)
    rsi = rsi.where(~zero_loss, 100.0)
    both_zero = zero_gain & zero_loss
    rsi = rsi.where(~both_zero, 50.0)
    return rsi


def _atr(registry: IndicatorRegistry, params: dict) -> pd.Series:
    length = params.get("length")
    if length is None:
        raise ValueError("atr requires a 'length' parameter.")
    length = int(length)
    bars = registry._bars
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return _rma(true_range, length)


def _adx(registry: IndicatorRegistry, params: dict) -> pd.Series:
    """
    Calculate ADX (Average Directional Index) using Welles Wilder's method.
    
    ADX measures trend strength regardless of direction.
    Formula matches TradingView's ta.dmi() implementation.
    
    Note: TradingView's ta.dmi() uses RMA (Wilder's smoothing) which is:
    - First value: Simple average of first N periods
    - Subsequent: RMA = previous + (current - previous) / N
    """
    length = params.get("length")
    if length is None:
        raise ValueError("adx requires a 'length' parameter.")
    length = int(length)
    bars = registry._bars
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    
    # Calculate True Range (same as ATR)
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    
    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    # +DM: up_move if it's greater than down_move and > 0, else 0
    # -DM: down_move if it's greater than up_move and > 0, else 0
    plus_dm = pd.Series(0.0, index=bars.index, dtype=float)
    minus_dm = pd.Series(0.0, index=bars.index, dtype=float)
    
    plus_condition = (up_move > down_move) & (up_move > 0)
    minus_condition = (down_move > up_move) & (down_move > 0)
    
    plus_dm.loc[plus_condition] = up_move[plus_condition]
    minus_dm.loc[minus_condition] = down_move[minus_condition]
    
    # Smooth +DM, -DM, and TR using RMA (Wilder's smoothing)
    # RMA: First value is average of first N periods, then RMA formula
    plus_dm_smooth = _rma(plus_dm, length)
    minus_dm_smooth = _rma(minus_dm, length)
    tr_smooth = _rma(tr, length)
    
    # Calculate +DI and -DI (Directional Indicators)
    plus_di = 100 * (plus_dm_smooth / tr_smooth).where(tr_smooth != 0, 0.0)
    minus_di = 100 * (minus_dm_smooth / tr_smooth).where(tr_smooth != 0, 0.0)
    
    # Calculate DX (Directional Index)
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * (di_diff / di_sum).where(di_sum != 0, 0.0)
    
    # ADX is RMA of DX (same smoothing method)
    adx = _rma(dx, length)
    
    return adx


def _donchian(registry: IndicatorRegistry, params: dict) -> pd.Series:
    length = params.get("length")
    if length is None:
        raise ValueError("donchian requires a 'length' parameter.")
    channel = params.get("channel", "upper").lower()
    bars = registry._bars
    highs = bars.get("high")
    lows = bars.get("low")
    if highs is None or lows is None:
        raise ValueError("Bars must contain 'high' and 'low' columns for Donchian channels.")
    upper = highs.rolling(int(length)).max()
    lower = lows.rolling(int(length)).min()
    if channel == "upper":
        return upper
    if channel == "lower":
        return lower
    if channel == "middle":
        return (upper + lower) / 2.0
    raise ValueError("channel must be one of: upper, lower, middle")


def _highest(registry: IndicatorRegistry, params: dict) -> pd.Series:
    length = params.get("length")
    if length is None:
        raise ValueError("highest requires a 'length' parameter.")
    source = _select_source(registry, params)
    return source.rolling(int(length)).max()


def _lowest(registry: IndicatorRegistry, params: dict) -> pd.Series:
    length = params.get("length")
    if length is None:
        raise ValueError("lowest requires a 'length' parameter.")
    source = _select_source(registry, params)
    return source.rolling(int(length)).min()


def _freeze_params(params: dict) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((key, _freeze_value(value)) for key, value in params.items()))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    return value
