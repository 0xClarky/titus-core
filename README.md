# Titus Core

**Private proprietary trading system**  
Core execution engines and exchange adapters for systematic algorithmic trading.

---

## Overview

Titus Core provides production-ready infrastructure for automated trading strategies with a focus on reliability, performance, and safety.

**What it does:**
- Backtests strategies with TradingView parity
- Executes live trading on crypto derivatives exchanges
- Manages multi-symbol portfolios (30+ assets per instance)
- Provides environment-agnostic strategy framework (same code for backtest & live)
- Handles order execution, position management, and risk controls

**What it's for:**
- Systematic trading strategy development
- Live execution of algorithmic strategies
- Research and optimization workflows
- Production deployments (Railway, AWS, etc.)

**Currently supported exchanges:**
- ✅ **HyperLiquid** (perpetuals) - Full support
- ✅ **Bybit** (perpetuals) - Full support

---

## Key Features

- **Environment-Agnostic Strategies** - Same code for backtest and live execution
- **TradingView Parity** - Match Pine Script behavior exactly for validation
- **Multi-Symbol Execution** - 30+ symbols in single process (6-10x more efficient)
- **Exchange-Agnostic Architecture** - Switch between HyperLiquid/Bybit via config
- **Position Sizing** - Available margin-based sizing prevents over-leveraging
- **Position Reconciliation** - Startup checks synchronize strategy state with exchange
- **Dry-Run Mode** - Test strategies with real market data, zero risk
- **Leverage Management** - Automatic enforcement and safety caps
- **Environment Variable Overrides** - Control risk/sizing via env vars (Railway-friendly)
- **Robust Order Management** - Market, limit, stop, and bracket orders
- **Non-Interactive Execution** - TTY detection for cloud deployments

---

## Installation

### From Private GitHub Repository

```bash
pip install git+https://github.com/0xClarky/titus-core.git@v0.2.6
```

### For Development

```bash
git clone https://github.com/0xClarky/titus-core.git
cd titus-core
pip install -e .
```

---

## Quick Start

### CLI Usage

```bash
# Dry-run mode (no real orders)
titus-live run config.yaml --dry-run

# Live execution (requires ENABLE_TRADING=true env var or confirmation)
titus-live run config.yaml --live

# Test single order placement
titus-live test-order 0xYOUR_ADDRESS --symbol BTC --quantity 0.001

# Emergency kill switch (flatten position + cancel orders)
titus-live kill-switch 0xYOUR_ADDRESS --symbol BTC
```

### Configuration

Create a `config.yaml`:

```yaml
id: my_strategy_prod
exchange: hyperliquid  # or 'bybit'

strategy:
  module: strategy
  class_name: MyStrategy
  parameters:
    risk_pct: 2.0
    max_leverage: 5.0

symbols:
  - BTC
  - ETH
  - SOL

bar_resolution: 4h
lookback_bars: 200
poll_interval: 60

max_position_size: 500.0
commission: 0.00055
```

### Environment Variables (Optional Overrides)

```bash
# HyperLiquid credentials
export HYPERLIQUID_PRIVATE_KEY=0x...
export HYPERLIQUID_ACCOUNT_ADDRESS=0x...
export HYPERLIQUID_TESTNET=false

# Bybit credentials
export BYBIT_API_KEY=...
export BYBIT_API_SECRET=...
export BYBIT_TESTNET=false

# Trading control
export ENABLE_TRADING=true  # Skips confirmation prompt, enables live mode

# Risk/sizing overrides (optional - overrides config.yaml)
export RISK_PCT=1.5
export MAX_POSITION_SIZE=1000.0
export SIZE_MULTIPLIER=0.5  # Scale all orders by 50%
```

### As a Library

```python
from titus_core.strategies.base import BaseStrategy, StrategyContext
from titus_live.execution.live_engine import LiveExecutionEngine
from titus_live.adapters.hyperliquid.client import HyperLiquidClient

class MyStrategy(BaseStrategy):
    name = "My Strategy"
    
    def on_bar(self, context: StrategyContext):
        # Your trading logic
        if context.bar['close'] > some_condition:
            context.buy(order_id="Long", quantity=0.01)
        elif context.bar['close'] < exit_condition:
            context.sell(order_id="Exit", quantity=0.01)

# Initialize client and run
hl_client = HyperLiquidClient(private_key="0x...", account_address="0x...")
config = LiveExecutionConfig(
    symbols=["BTC", "ETH"],
    bar_resolution="4h",
    exchange="hyperliquid"
)
engine = LiveExecutionEngine(config, hl_client)
engine.run(MyStrategy, {})
```

---

## Architecture

### titus_core/

Core backtesting and strategy infrastructure:
- **data/** - Market data adapters (Bybit, HyperLiquid, cache layer)
- **execution/** - Backtest engine with TradingView parity
- **indicators/** - Technical indicator registry
- **strategies/** - Base strategy classes and framework
- **trading/** - Order and portfolio management
- **utils/** - Configuration, environment, strategy loading

### titus_live/

Live execution and exchange adapters:
- **execution/** - Real-time bar polling, multi-symbol orchestration
- **adapters/** - Exchange-specific clients (HyperLiquid, Bybit)
- **cli/** - Command-line interface for live trading
- **utils/** - Live configuration management

---

## Multi-Symbol Support

Run **30+ symbols in a single process:**

```yaml
symbols:
  - BTC
  - ETH
  - SOL
  - AVAX
  - MATIC
  # ... up to 30+ symbols
```

- Independent strategy instances per symbol
- No state cross-contamination
- Available margin-based position sizing
- 6-10x more efficient than separate processes

---

## Deployment

### Railway (Recommended)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["titus-live", "run", "config.yaml", "--dry-run"]
```

Set environment variables in Railway dashboard. Use `ENABLE_TRADING=true` to enable live mode without interactive confirmation.

### Local Development

```bash
cd your-strategy/
export HYPERLIQUID_PRIVATE_KEY=0x...
export HYPERLIQUID_ACCOUNT_ADDRESS=0x...
titus-live run config.yaml --dry-run
```

---

## Requirements

- **Python:** 3.11+
- **Exchange Account:** HyperLiquid or Bybit for live trading
- **Access:** Private repository access (GitHub token)

---

## Safety Features

- **Dry-Run Mode:** Test with real data, zero order execution
- **Position Reconciliation:** Startup checks prevent state mismatches
- **Leverage Caps:** Automatic enforcement of max leverage
- **Available Margin Sizing:** Prevents over-allocation in multi-symbol
- **Fail-Fast Parsing:** Position/order data parsing errors crash immediately (no silent failures)
- **Emergency Kill Switch:** Instant position flattening via CLI

---

## Documentation

Full documentation available in development repository:
- Strategy parity guide (TradingView ↔ Titus)
- Live execution path diagrams
- Production deployment guide
- Security best practices

---

## Version

**Current:** v0.2.6  
**Status:** Production-ready  
**Latest Updates:**
- Fix: Account address parameter passing in API retry logic
- Feature: Environment variable overrides for risk/sizing
- Feature: Non-interactive execution for cloud deployments
- Feature: Multi-asset support with independent state tracking

---

## License

**Proprietary - All Rights Reserved**  
Copyright © 2025 Relic Labs

This software is private and confidential. Unauthorized copying, distribution, or use is strictly prohibited.
