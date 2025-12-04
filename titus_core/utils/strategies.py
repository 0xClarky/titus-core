"""Utilities for loading strategy classes."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Type

from titus_core.strategies.base import BaseStrategy
from titus_core.utils.config import StrategyConfig


def load_strategy_class(config: StrategyConfig) -> Type[BaseStrategy]:
    """Import and validate a strategy class from a config block.
    
    Automatically adds current working directory to sys.path if not already present
    to allow importing local strategy modules.
    """
    # Add current working directory to sys.path if not already there
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    
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

