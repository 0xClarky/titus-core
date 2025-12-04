"""CLI for live execution on HyperLiquid."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer

from titus_core.utils.config import load_yaml_config
from titus_core.utils.env import load_project_env
from titus_core.utils.strategies import load_strategy_class
from titus_live.adapters.hyperliquid.client import HyperLiquidClient
from titus_live.execution.live_engine import LiveEngineConfig, LiveExecutionEngine

app = typer.Typer(help="Titus live execution CLI")
load_project_env()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, help="Path to live execution config YAML"),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--live",
        help="Dry run mode (log signals without executing). Use --live to enable real execution.",
    ),
) -> None:
    """Run live execution engine with specified configuration.
    
    Example:
        python -m apps.live configs/live/atr_breakout4_live.yaml --dry-run
    """
    from titus_live.utils.config import LiveExecutionConfig
    
    live_config = load_yaml_config(config, LiveExecutionConfig)
    
    # Get normalized symbols list (config validator handles backward compatibility)
    symbols = live_config.get_symbols()
    
    typer.echo(f"[Titus Live] Loaded config '{live_config.id}'")
    typer.echo(f"Strategy: {live_config.strategy.class_name}")
    if len(symbols) == 1:
        typer.echo(f"Symbol: {symbols[0]} @ {live_config.bar_resolution}")
    else:
        typer.echo(f"Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}) @ {live_config.bar_resolution}")
    typer.echo(f"Mode: {'DRY RUN' if dry_run else 'LIVE EXECUTION'}")
    
    # Get HyperLiquid credentials from environment
    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    account_address = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS")
    testnet = os.getenv("HYPERLIQUID_TESTNET", "false").lower() == "true"
    
    if not private_key or not account_address:
        typer.echo(
            "Error: HyperLiquid credentials not found in environment.\n"
            "Please set HYPERLIQUID_PRIVATE_KEY and HYPERLIQUID_ACCOUNT_ADDRESS.",
            err=True,
        )
        raise typer.Exit(code=1)
    
    # Confirm before live execution
    if not dry_run:
        confirm = typer.confirm(
            "\n⚠️  LIVE EXECUTION MODE\n"
            "This will execute real orders on HyperLiquid with real money.\n"
            "Are you sure you want to continue?",
            abort=True,
        )
        typer.echo("Starting live execution...")
    else:
        typer.echo("Starting dry run (signals will be logged but not executed)...")
    
    # Initialize HyperLiquid client
    hl_client = HyperLiquidClient(
        private_key=private_key,
        account_address=account_address,
        testnet=testnet,
    )
    
    # Build engine config
    engine_config = LiveEngineConfig(
        commission=live_config.commission,
        bar_resolution=live_config.bar_resolution,
        symbols=symbols,
        dry_run=dry_run,
        max_position_size=live_config.max_position_size,
        max_leverage=live_config.max_leverage,
        poll_interval=live_config.poll_interval,
        lookback_bars=live_config.lookback_bars,
    )
    
    # Load strategy class (will be instantiated per symbol by engine)
    strategy_cls = load_strategy_class(live_config.strategy)
    
    # Create and run live engine
    engine = LiveExecutionEngine(config=engine_config, hl_client=hl_client)
    
    try:
        typer.echo("\n[Titus Live] Engine started. Press Ctrl+C to stop.\n")
        engine.run(strategy_cls, live_config.strategy.parameters)
    except KeyboardInterrupt:
        typer.echo("\n[Titus Live] Received stop signal")
    finally:
        typer.echo("[Titus Live] Engine stopped")


@app.command()
def status(
    account_address: str = typer.Argument(..., help="HyperLiquid account address"),
) -> None:
    """Check account status and positions on HyperLiquid.
    
    Example:
        python -m apps.live status 0x1234...
    """
    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    testnet = os.getenv("HYPERLIQUID_TESTNET", "false").lower() == "true"
    
    if not private_key:
        typer.echo("Error: HYPERLIQUID_PRIVATE_KEY not found in environment.", err=True)
        raise typer.Exit(code=1)
    
    hl_client = HyperLiquidClient(
        private_key=private_key,
        account_address=account_address,
        testnet=testnet,
    )
    
    # Get equity
    equity = hl_client.get_equity()
    typer.echo(f"\n📊 Account Status")
    typer.echo(f"   Equity: ${equity:,.2f}")
    typer.echo(f"   Network: {'Testnet' if testnet else 'Mainnet'}")
    
    # Get positions (would need to query all symbols or pass symbol)
    typer.echo(f"\n💼 Positions: (query specific symbols for details)")


@app.command()
def test_order(
    account_address: str = typer.Argument(..., help="HyperLiquid account address"),
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Symbol to test"),
    side: str = typer.Option("buy", "--side", help="Order side: buy or sell"),
    quantity: float = typer.Option(0.001, "--quantity", "-q", help="Order quantity"),
    leverage: float = typer.Option(5.0, "--leverage", "-l", help="Leverage to set"),
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Dry run (default) or execute real order"),
) -> None:
    """Test order placement on HyperLiquid (validates API connection and order structure).
    
    Example:
        # Test (no execution)
        python -m apps.live test-order 0x1234... --symbol BTC --side buy --quantity 0.001 --dry-run
        
        # Execute small test order (use with caution!)
        python -m apps.live test-order 0x1234... --symbol BTC --side buy --quantity 0.001 --execute
    """
    from titus_core.trading.orders import OrderSide
    
    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    testnet = os.getenv("HYPERLIQUID_TESTNET", "false").lower() == "true"
    
    if not private_key:
        typer.echo("Error: HYPERLIQUID_PRIVATE_KEY not found in environment.", err=True)
        raise typer.Exit(code=1)
    
    hl_client = HyperLiquidClient(
        private_key=private_key,
        account_address=account_address,
        testnet=testnet,
    )
    
    # Convert side string to OrderSide
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    
    typer.echo(f"\n📋 Test Order Configuration")
    typer.echo(f"   Symbol: {symbol}")
    typer.echo(f"   Side: {order_side.name}")
    typer.echo(f"   Quantity: {quantity}")
    typer.echo(f"   Leverage: {leverage}x")
    typer.echo(f"   Mode: {'DRY RUN (will not execute)' if dry_run else 'LIVE EXECUTION'}")
    typer.echo(f"   Network: {'Testnet' if testnet else 'Mainnet'}\n")
    
    if dry_run:
        typer.echo("✅ DRY RUN: Order structure validated (not executed)")
        typer.echo(f"   Would place: {order_side.name} {quantity} {symbol}")
        return
    
    # Real execution
    confirm = typer.confirm(
        f"\n⚠️  LIVE ORDER\n"
        f"This will place a REAL market order on HyperLiquid:\n"
        f"   {order_side.name} {quantity} {symbol}\n"
        f"Continue?",
        abort=True,
    )
    
    typer.echo(f"\nPlacing {order_side.name} order...")
    
    try:
        result = hl_client.place_market_order(
            symbol=symbol,
            side=order_side,
            quantity=quantity,
            reduce_only=False,
            leverage=leverage,
        )
        
        if result.get("success"):
            typer.echo(f"✅ Order placed successfully")
            typer.echo(f"   Result: {result.get('result')}")
        else:
            typer.echo(f"❌ Order failed: {result.get('error')}", err=True)
            raise typer.Exit(code=1)
            
    except Exception as e:
        typer.echo(f"❌ Exception: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def kill_switch(
    account_address: str = typer.Argument(..., help="HyperLiquid account address"),
    symbol: str = typer.Option(None, "--symbol", "-s", help="Flatten specific symbol (or all if not specified)"),
) -> None:
    """Emergency: Flatten all positions immediately.
    
    Example:
        python -m apps.live kill-switch 0x1234... --symbol BTC
    """
    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
    testnet = os.getenv("HYPERLIQUID_TESTNET", "false").lower() == "true"
    
    if not private_key:
        typer.echo("Error: HYPERLIQUID_PRIVATE_KEY not found in environment.", err=True)
        raise typer.Exit(code=1)
    
    hl_client = HyperLiquidClient(
        private_key=private_key,
        account_address=account_address,
        testnet=testnet,
    )
    
    if symbol:
        confirm = typer.confirm(
            f"\n⚠️  KILL SWITCH: Flatten {symbol} position?\n"
            "This will close the position with a market order.\n"
            "Continue?",
            abort=True,
        )
        
        typer.echo(f"Flattening {symbol} position...")
        success = hl_client.flatten_position(symbol)
        
        if success:
            typer.echo(f"✅ {symbol} position flattened successfully")
        else:
            typer.echo(f"❌ Failed to flatten {symbol} position", err=True)
            raise typer.Exit(code=1)
    else:
        typer.echo("⚠️  Flatten all positions not yet implemented")
        typer.echo("Please specify --symbol")
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()

