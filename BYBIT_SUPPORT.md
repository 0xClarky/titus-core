# Bybit Support in Titus Core

## Overview

`titus-core` now supports **both HyperLiquid and Bybit** for live trading execution. The Bybit adapter was ported from the battle-tested `bybit-relay` system.

## Architecture

### Exchange Clients

1. **HyperLiquidClient** (`titus_live/adapters/hyperliquid/client.py`)
   - Uses `hyperliquid-python-sdk` + `eth-account`
   - Credentials: `HYPERLIQUID_PRIVATE_KEY`, `HYPERLIQUID_ACCOUNT_ADDRESS`

2. **BybitClient** (`titus_live/adapters/bybit/client.py`) ✨ NEW
   - Uses `pybit` (Unified Trading API v5)
   - Credentials: `BYBIT_API_KEY`, `BYBIT_API_SECRET`
   - Ported from `bybit-relay` adapter

### Common Interface

Both clients implement the same interface:
- `get_equity()` - Fetch account balance
- `get_mark_price(symbol)` - Get current mark price
- `get_position(symbol)` - Query open position
- `place_market_order()` - Execute market order
- `place_limit_order()` - Place take profit
- `place_stop_order()` - Place stop loss
- `flatten_position(symbol)` - Emergency close
- `cancel_order()` - Cancel pending order
- `reconcile_position()` - Verify state sync

## Usage

### CLI Commands

```bash
# HyperLiquid (default)
export HYPERLIQUID_PRIVATE_KEY=0x...
export HYPERLIQUID_ACCOUNT_ADDRESS=0x...
titus-live run config.yaml --dry-run --exchange hyperliquid

# Bybit
export BYBIT_API_KEY=your_key
export BYBIT_API_SECRET=your_secret
titus-live run config.yaml --dry-run --exchange bybit
```

### Test Commands

```bash
# Test HyperLiquid order
titus-live test-order 0xYOUR_ADDRESS --symbol BTC --side buy --quantity 0.001 --dry-run

# For Bybit, use the same command but set BYBIT credentials first
```

### Environment Variables

**HyperLiquid:**
- `HYPERLIQUID_PRIVATE_KEY` - Wallet private key
- `HYPERLIQUID_ACCOUNT_ADDRESS` - Wallet address
- `HYPERLIQUID_TESTNET` - Set to `true` for testnet (default: `false`)

**Bybit:**
- `BYBIT_API_KEY` - API key from Bybit account
- `BYBIT_API_SECRET` - API secret (never logged)
- `BYBIT_TESTNET` - Set to `true` for testnet (default: `false`)

## Symbol Format Differences

### HyperLiquid
- Uses short symbols: `BTC`, `ETH`, `SOL`
- No suffix needed

### Bybit
- Uses full pairs: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`
- Linear perpetuals (Unified Trading Account)

**Note:** You'll need to adjust your config files depending on which exchange you're targeting.

## Implementation Details

### Bybit Adapter Features

✅ Market orders (entry)  
✅ Limit orders (take profit)  
✅ Stop market orders (stop loss)  
✅ Position reconciliation  
✅ Instrument constraints (tick size, lot size)  
✅ Leverage management  
✅ Order ID tracking  
✅ Retry logic with exponential backoff  
✅ Auth error detection (no retry on 401/signature errors)  

### Safety Features

1. **Dry-run mode** - Log signals without execution (default)
2. **Position reconciliation** - Verify exchange state matches strategy state
3. **Max position size** - Cap notional exposure per symbol
4. **Max leverage** - Prevent over-leveraging
5. **Order quantity rounding** - Automatic tick/lot size compliance

## Dependencies

Added to `pyproject.toml`:
- `pybit>=5.7.0` - Bybit Unified Trading API

## Next Steps

### For Production Deployment

1. **Test on testnet first**
   ```bash
   export BYBIT_TESTNET=true
   export BYBIT_API_KEY=testnet_key
   export BYBIT_API_SECRET=testnet_secret
   titus-live run config.yaml --dry-run --exchange bybit
   ```

2. **Verify symbol formats** - Update configs to use `BTCUSDT` format for Bybit

3. **Monitor dry-run for 24h+** - Ensure signals match expectations

4. **Deploy to Railway** - Same process as HyperLiquid, just change env vars

### Future Enhancements

- [ ] Add `status` command with `--exchange` flag
- [ ] Add `kill-switch` command with `--exchange` flag  
- [ ] Support `test-order` for Bybit explicitly
- [ ] Add exchange-specific config validation
- [ ] Support Bybit market data feed for live bars (currently uses HL only)

## Compatibility

✅ Backward compatible - existing HyperLiquid deployments work unchanged  
✅ Same strategy code works on both exchanges  
✅ Same config format (just adjust symbols)  
✅ CLI defaults to HyperLiquid if `--exchange` not specified  

---

**Version:** 0.1.1+  
**Author:** Relic Labs  
**Source:** Ported from `bybit-relay` production system

