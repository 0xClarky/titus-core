# Titus Core

**Private proprietary trading system**  
Core execution engines and exchange adapters for algorithmic trading.

---

## Overview

Titus Core provides the foundational infrastructure for systematic trading:

- **Backtest Engine** - TradingView-parity backtesting with historical data
- **Live Execution Engine** - Real-time strategy execution on HyperLiquid
- **Exchange Adapters** - HyperLiquid order execution and data fetching
- **Strategy Framework** - Environment-agnostic strategy interface
- **Multi-Symbol Support** - Execute 30+ symbols in a single process

---

## Installation

### From Private GitHub Repository

```bash
pip install git+https://github.com/0xClarky/titus-core.git@v0.1.0
```

### For Development

```bash
git clone https://github.com/0xClarky/titus-core.git
cd titus-core
pip install -e .
```

---

## Quick Start

### Using the CLI

```bash
# Run live execution (dry-run mode)
titus-live run config.yaml --dry-run

# Test order placement  
titus-live test-order 0xYOUR_ADDRESS --symbol BTC --quantity 0.001

# Emergency kill switch
titus-live kill-switch 0xYOUR_ADDRESS --symbol BTC
```

### As a Library

```python
from titus_core.strategies.base import BaseStrategy, StrategyContext
from titus_live.execution.live_engine import LiveExecutionEngine, LiveEngineConfig
from titus_live.adapters.hyperliquid.client import HyperLiquidClient

# Create your strategy
class MyStrategy(BaseStrategy):
    name = "My Strategy"
    
    def on_bar(self, context: StrategyContext):
        # Your trading logic here
        if context.bar['close'] > some_condition:
            context.buy(order_id="Long", quantity=0.01)

# Run live
hl_client = HyperLiquidClient(private_key="0x...", account_address="0x...")
config = LiveEngineConfig(symbols=["BTC", "ETH"], bar_resolution="4h")
engine = LiveExecutionEngine(config, hl_client)
engine.run(MyStrategy, {})
```

---

## Components

### titus_core/

Core backtesting and strategy infrastructure:
- **data/** - Market data adapters (Bybit, HyperLiquid)
- **execution/** - Backtest engine with TradingView parity
- **indicators/** - Technical indicator registry
- **strategies/** - Base strategy classes
- **trading/** - Order and portfolio management

### titus_live/

Live execution and exchange adapters:
- **execution/** - Real-time bar polling and strategy execution
- **adapters/hyperliquid/** - HyperLiquid API client
- **cli/** - Command-line interface for live trading

---

## Features

- ✅ **Environment-Agnostic Strategies** - Same code for backtest and live
- ✅ **TradingView Parity** - Match Pine Script behavior exactly
- ✅ **Multi-Symbol Execution** - 30+ symbols in single process (6-10x efficient)
- ✅ **Robust Order Management** - Market, limit, stop, and bracket orders
- ✅ **Position Reconciliation** - Startup checks and state management
- ✅ **Dry-Run Mode** - Test strategies with real data, zero risk
- ✅ **Leverage Management** - Automatic enforcement and safety caps
- ✅ **Available Margin Sizing** - Prevents over-leveraging in multi-symbol

---

## Requirements

- Python 3.11+
- HyperLiquid account for live trading
- Private repository access

---

## Documentation

Full documentation available in the development repository.

---

## License

Proprietary - All Rights Reserved  
Copyright © 2025 Relic Labs

This software is private and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

---

## Version

Current: v0.1.0  
Status: Production-ready
