"""Configuration system backed by Pydantic + YAML."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel, Field, field_validator

T = TypeVar("T", bound=BaseModel)


class DataConfig(BaseModel):
    """Data sourcing parameters."""

    source: str = Field(default="bybit", description="Data provider identifier")
    symbol: str
    exchange: str
    resolution: str
    start: datetime
    end: datetime
    use_cache: bool = True
    force_refresh: bool = False

    @field_validator("start", "end", mode="before")
    @classmethod
    def ensure_datetime(cls, value: Any) -> datetime:
        if isinstance(value, datetime):
            dt_value = value
        else:
            dt_value = datetime.fromisoformat(str(value))
        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc)


class StrategyConfig(BaseModel):
    """Strategy module metadata."""

    module: str
    class_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class EngineConfig(BaseModel):
    """Execution engine knobs that must match Pine defaults."""

    initial_capital: float = 100_000
    commission: float = 0.0
    slippage: float = 0.0
    tick_size: Optional[float] = None
    pyramiding: int = 1
    calc_on_every_tick: bool = False
    process_orders_on_close: bool = False
    default_qty_type: str = Field(default="fixed", description="fixed | percent_equity")
    default_qty_value: float = 1.0
    allow_long: bool = True
    allow_short: bool = True
    close_entries_rule: str = Field(default="none", description="none | opposite")
    slippage_type: str = Field(default="absolute", description="absolute | percent | ticks")
    # Leverage support (optional, defaults maintain backward compatibility)
    use_leverage: bool = Field(default=False, description="Enable margin-based leverage accounting")
    leverage: float = Field(default=1.0, description="Leverage multiplier (e.g., 5.0 for 5x leverage)")
    margin_long: Optional[float] = Field(default=None, description="Margin requirement for long positions (percentage, e.g., 20 for 5x leverage)")
    margin_short: Optional[float] = Field(default=None, description="Margin requirement for short positions (percentage, e.g., 20 for 5x leverage)")

    @field_validator("pyramiding")
    @classmethod
    def validate_pyramiding(cls, value: int) -> int:
        if value < 1:
            raise ValueError("pyramiding must be >= 1")
        return value

    @field_validator("default_qty_type")
    @classmethod
    def validate_qty_type(cls, value: str) -> str:
        if value not in {"fixed", "percent_equity"}:
            raise ValueError("default_qty_type must be 'fixed' or 'percent_equity'")
        return value

    @field_validator("close_entries_rule")
    @classmethod
    def validate_close_rule(cls, value: str) -> str:
        if value not in {"none", "opposite"}:
            raise ValueError("close_entries_rule must be 'none' or 'opposite'")
        return value

    @field_validator("slippage_type")
    @classmethod
    def validate_slippage_type(cls, value: str) -> str:
        if value not in {"absolute", "percent", "ticks"}:
            raise ValueError("slippage_type must be 'absolute', 'percent', or 'ticks'")
        return value


class BacktestConfig(BaseModel):
    """Top-level configuration model for a single run."""

    id: str
    data: DataConfig
    strategy: StrategyConfig
    engine: EngineConfig = Field(default_factory=EngineConfig)


class OptimizationTarget(BaseModel):
    """Single data target used in optimization."""

    id: str
    data: Optional[DataConfig] = None
    symbol: Optional[str] = None

    def resolve_data(self, default: Optional[DataConfig]) -> DataConfig:
        if self.data:
            return self.data
        if default is None:
            raise ValueError(
                f"Target '{self.id}' requires 'data' or a top-level data_template."
            )
        merged = default.model_copy()
        if self.symbol:
            merged.symbol = self.symbol
        return merged


class OptimizationConfig(BaseModel):
    """Configuration for grid-search style optimization."""

    id: str
    strategy: StrategyConfig
    engine: EngineConfig = Field(default_factory=EngineConfig)
    parameter_grid: dict[str, list[Any]]
    targets: list[OptimizationTarget]
    output_dir: Path = Field(default=Path("titus_results/optimization"))
    data_template: Optional[DataConfig] = None

    @field_validator("parameter_grid")
    @classmethod
    def validate_parameter_grid(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not value:
            raise ValueError("parameter_grid cannot be empty.")
        for name, options in value.items():
            if not options:
                raise ValueError(f"parameter '{name}' must have at least one value.")
        return value

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, value: list[OptimizationTarget]) -> list[OptimizationTarget]:
        if not value:
            raise ValueError("targets cannot be empty.")
        return value

    @field_validator("output_dir", mode="before")
    @classmethod
    def coerce_output_dir(cls, value: Any) -> Path:
        return Path(value)


def load_yaml_config(path: Any, model: Type[T]) -> T:
    """Load YAML file and parse it into the provided Pydantic model."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file {file_path} does not exist.")
    with file_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return model.model_validate(payload)
