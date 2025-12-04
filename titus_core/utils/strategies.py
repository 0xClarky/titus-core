"""Utilities for loading strategy classes."""

from __future__ import annotations

import importlib
from typing import Type

from titus_core.strategies.base import BaseStrategy
from titus_core.utils.config import StrategyConfig


def load_strategy_class(config: StrategyConfig) -> Type[BaseStrategy]:
    """Import and validate a strategy class from a config block."""
    module = importlib.import_module(config.module)
    try:
        cls = getattr(module, config.class_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ImportError(
            f"Strategy class '{config.class_name}' not found in {config.module}"
        ) from exc
    if not issubclass(cls, BaseStrategy):
        raise TypeError(f"{config.class_name} must inherit from BaseStrategy")
    return cls

