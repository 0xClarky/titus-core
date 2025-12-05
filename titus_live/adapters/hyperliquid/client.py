"""HyperLiquid exchange adapter for live order execution.

Adapted from bybit-relay patterns but using HyperLiquid Python SDK.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from titus_core.trading.orders import OrderRequest, OrderSide, OrderType

logger = logging.getLogger(__name__)


class HyperLiquidClient:
    """Client for executing orders on HyperLiquid DEX."""

    def __init__(
        self,
        private_key: str,
        account_address: str,
        testnet: bool = False,
    ) -> None:
        """Initialize HyperLiquid client.

        Args:
            private_key: Wallet private key for signing transactions (NEVER logged)
            account_address: Main wallet address
            testnet: Use testnet (default: False for mainnet)
        """
        if not private_key or not account_address:
            raise ValueError(
                "HyperLiquid credentials not found. "
                "Please provide private_key and account_address."
            )
        
        # SECURITY: Store but never log the private key
        self._private_key = private_key  # Keep reference but never expose

        self.account_address = account_address
        self.testnet = testnet
        
        # API base URL
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        
        # Create wallet object from private key
        # HyperLiquid SDK requires an eth_account LocalAccount object
        self.wallet = Account.from_key(self._private_key)
        
        # Verify the wallet address matches the provided account address
        if self.wallet.address.lower() != account_address.lower():
            raise ValueError(
                f"Private key does not match account address. "
                f"Wallet address: {self.wallet.address}, "
                f"Provided address: {account_address}"
            )
        
        # Initialize Info (read-only queries)
        self.info = Info(base_url, skip_ws=True)
        
        # Initialize Exchange (order execution)
        # SDK expects a wallet object, not raw credentials
        self.exchange = Exchange(
            wallet=self.wallet,
            base_url=base_url,
        )
        
        # Cache for instrument info
        self.instrument_cache: dict[str, dict[str, Any]] = {}
        self.leverage_cache: dict[str, float] = {}
        
        # Order ID tracking: strategy_order_id -> HL_numeric_id
        # Format: {symbol: {strategy_id: hl_oid}}
        self.order_id_map: dict[str, dict[str, int]] = {}
        
        logger.info(
            f"HyperLiquid client initialized - Account: {account_address[:6]}...{account_address[-4:]}, "
            f"Network: {'Testnet' if testnet else 'Mainnet'}"
        )

    def _retry_api_call(
        self,
        func: callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Retry API call with exponential backoff.

        Does not retry authentication errors or invalid signatures.
        
        Args:
            func: Function to call
            max_retries: Maximum retry attempts
            base_delay: Base delay between retries (seconds)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                
                # Check for error response format from HL
                if isinstance(result, dict) and "status" in result:
                    status = result.get("status")
                    if status == "ok":
                        return result
                    elif status == "err":
                        error_msg = result.get("response", "Unknown error")
                        # Don't retry auth/signature errors
                        if "signature" in str(error_msg).lower() or "auth" in str(error_msg).lower():
                            logger.error(
                                "Authentication error (not retrying)",
                                error=error_msg,
                            )
                            raise Exception(f"Authentication failed: {error_msg}")
                        
                        logger.warning(
                            "API returned error status",
                            error=error_msg,
                            attempt=attempt + 1,
                        )
                        # Fall through to retry logic
                    else:
                        # Unknown status, return as-is
                        return result
                else:
                    # Not a status dict, assume success
                    return result
                    
            except Exception as e:
                error_str = str(e)
                
                # Check for auth errors
                if "signature" in error_str.lower() or "unauthorized" in error_str.lower():
                    logger.error(
                        "Authentication error (not retrying)",
                        error=error_str,
                    )
                    raise Exception(
                        f"Authentication failed: {error_str}. "
                        "Check your private key and account address."
                    ) from e
                
                last_exception = e
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "API call failed, retrying",
                        error=error_str,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "API call failed after all retries",
                        error=error_str,
                        max_retries=max_retries,
                    )
        
        raise last_exception or Exception("API call failed")

    def get_equity(self) -> float:
        """Fetch available equity for position sizing.
        
        Prioritizes available margin over total equity to prevent over-leveraging
        in multi-symbol mode (mirrors Bybit-relay behavior).

        Returns:
            Available equity/margin in USD

        Raises:
            Exception: If equity cannot be fetched
        """
        logger.debug("Fetching equity from HyperLiquid")
        
        try:
            user_state = self._retry_api_call(
                self.info.user_state,
                3,  # max_retries
                1.0,  # base_delay
                self.account_address,
            )
            
            # Extract equity from user state
            # HL returns marginSummary with various equity/margin fields
            margin_summary = user_state.get("marginSummary", {})
            
            # Try fields in order of preference (like Bybit-relay)
            # Prefer available margin to prevent race condition in multi-symbol
            value = None
            chosen_field = None
            
            for key in (
                "withdrawable",       # Free margin (best for multi-symbol)
                "accountValue",       # Total equity (includes unrealized PnL)
                "totalMarginUsed",    # Could calculate available from this
            ):
                try:
                    v = float(margin_summary.get(key, 0.0) or 0.0)
                    if v > 0:
                        value = v
                        chosen_field = key
                        break
                except (ValueError, TypeError):
                    continue
            
            if value is None or value <= 0:
                logger.error(
                    f"Equity returned is zero or invalid - "
                    f"Available fields: {list(margin_summary.keys())}"
                )
                raise Exception(f"Invalid equity: {value}")
            
            logger.info(f"Equity fetched successfully: ${value:,.2f} (from {chosen_field})")
            return value
            
        except Exception as e:
            logger.error(f"Failed to fetch equity: {e}")
            raise

    def get_mark_price(self, symbol: str) -> Optional[float]:
        """Fetch current mark price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC")

        Returns:
            Mark price or None if unavailable
        """
        try:
            meta = self._retry_api_call(self.info.meta)
            
            # Find symbol in universe
            universe = meta.get("universe", [])
            for asset in universe:
                if asset.get("name") == symbol:
                    mark_price = float(asset.get("markPx", 0))
                    return mark_price if mark_price > 0 else None
            
            logger.warning(f"Symbol {symbol} not found in universe")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch mark price for {symbol}: {e}")
            return None

    def get_position(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get current position for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position dict with size, entry_price, etc., or None if flat
            
        Raises:
            Exception: If position data cannot be parsed (fail-fast for safety)
        """
        try:
            user_state = self._retry_api_call(
                self.info.user_state,
                3,  # max_retries
                1.0,  # base_delay
                self.account_address,
            )
            
            positions = user_state.get("assetPositions", [])
            for pos in positions:
                position_info = pos.get("position", {})
                coin = position_info.get("coin")
                
                # Parse size - fail fast if invalid type
                szi_raw = position_info.get("szi", 0)
                try:
                    size = float(szi_raw if szi_raw is not None else 0)
                except (ValueError, TypeError) as e:
                    logger.error(
                        f"Invalid position size data for {symbol}: "
                        f"szi={szi_raw} (type={type(szi_raw).__name__})"
                    )
                    raise TypeError(f"Invalid position size for {symbol}: {szi_raw}") from e
                
                if coin == symbol and size != 0:
                    # Parse leverage value - fail fast if invalid
                    lev_data = position_info.get("leverage", {})
                    try:
                        if isinstance(lev_data, dict):
                            leverage_val = float(lev_data.get("value", 1) if lev_data.get("value") is not None else 1)
                        else:
                            leverage_val = float(lev_data if lev_data is not None else 1)
                    except (ValueError, TypeError) as e:
                        logger.error(
                            f"Invalid leverage data for {symbol}: "
                            f"leverage={lev_data} (type={type(lev_data).__name__})"
                        )
                        raise TypeError(f"Invalid leverage for {symbol}: {lev_data}") from e
                    
                    # Parse other fields - fail fast if invalid
                    try:
                        entry_px = position_info.get("entryPx", 0)
                        position_val = position_info.get("positionValue", 0)
                        unrealized = position_info.get("unrealizedPnl", 0)
                        
                        return {
                            "symbol": symbol,
                            "size": size,
                            "entry_price": float(entry_px if entry_px is not None else 0),
                            "position_value": float(position_val if position_val is not None else 0),
                            "unrealized_pnl": float(unrealized if unrealized is not None else 0),
                            "leverage": leverage_val,
                        }
                    except (ValueError, TypeError) as e:
                        logger.error(
                            f"Invalid position data fields for {symbol}: "
                            f"entryPx={entry_px}, positionValue={position_val}, unrealizedPnl={unrealized}"
                        )
                        raise TypeError(f"Invalid position data for {symbol}") from e
            
            # No position found for this symbol
            return None
            
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}", exc_info=True)
            raise  # Fail fast - don't silently return None on errors

    def _get_instrument_info(self, symbol: str) -> dict[str, Any]:
        """Fetch instrument metadata (tick size, lot size).

        Args:
            symbol: Trading pair symbol

        Returns:
            Dict with tick_size, lot_size, min_order_qty
        """
        if symbol in self.instrument_cache:
            return self.instrument_cache[symbol]
        
        try:
            meta = self._retry_api_call(self.info.meta)
            universe = meta.get("universe", [])
            
            for asset in universe:
                if asset.get("name") == symbol:
                    sz_decimals = int(asset.get("szDecimals", 3))
                    lot_size = 10 ** (-sz_decimals)
                    
                    # Tick size from price decimals (not always available)
                    tick_size = 0.01  # Default fallback
                    
                    info = {
                        "tick_size": tick_size,
                        "lot_size": lot_size,
                        "min_order_qty": lot_size,  # Minimum is typically 1 lot
                    }
                    
                    self.instrument_cache[symbol] = info
                    logger.debug(f"Cached instrument info for {symbol}: {info}")
                    return info
            
            # Symbol not found, return defaults
            logger.warning(f"Symbol {symbol} not found in meta, using defaults")
            default_info = {
                "tick_size": 0.01,
                "lot_size": 0.001,
                "min_order_qty": 0.001,
            }
            self.instrument_cache[symbol] = default_info
            return default_info
            
        except Exception as e:
            logger.error(f"Failed to fetch instrument info for {symbol}: {e}")
            # Return safe defaults
            return {
                "tick_size": 0.01,
                "lot_size": 0.001,
                "min_order_qty": 0.001,
            }

    def _round_to_tick(self, price: float, tick_size: float) -> float:
        """Round price to tick size.

        Args:
            price: Raw price
            tick_size: Minimum price increment

        Returns:
            Rounded price
        """
        if tick_size <= 0:
            return price
        return round(price / tick_size) * tick_size

    def _apply_lot_constraints(self, symbol: str, quantity: float) -> float:
        """Round quantity to lot size and enforce minimums.

        Args:
            symbol: Trading pair
            quantity: Raw quantity

        Returns:
            Constrained quantity (0 if below minimum)
        """
        info = self._get_instrument_info(symbol)
        lot_size = info.get("lot_size", 0.001)
        min_qty = info.get("min_order_qty", 0.0)
        
        if lot_size > 0:
            quantity = (quantity // lot_size) * lot_size
        
        if quantity < min_qty:
            return 0.0
        
        return quantity

    def _ensure_leverage(self, symbol: str, leverage: float) -> None:
        """Set leverage for a symbol if not already set.

        Args:
            symbol: Trading pair
            leverage: Target leverage (e.g., 5.0)
        """
        if self.leverage_cache.get(symbol) == leverage:
            return
        
        try:
            # HL SDK: set leverage using update_leverage
            result = self._retry_api_call(
                self.exchange.update_leverage,
                leverage=int(leverage),
                name=symbol,
            )
            
            if result.get("status") == "ok":
                self.leverage_cache[symbol] = leverage
                logger.info(f"Leverage set for {symbol}: {leverage}x")
            else:
                logger.warning(f"Failed to set leverage for {symbol}: {result}")
        except Exception as e:
            logger.warning(f"Error setting leverage for {symbol}: {e}")

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        reduce_only: bool = False,
        leverage: Optional[float] = None,
    ) -> dict[str, Any]:
        """Place a market order.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order quantity
            reduce_only: If True, order only reduces position
            leverage: Optional leverage to set (e.g., 5.0)

        Returns:
            Order result dict

        Raises:
            Exception: If order fails
        """
        # Set leverage if specified and not reducing position
        if leverage is not None and not reduce_only:
            self._ensure_leverage(symbol, leverage)
        
        # Constrain quantity
        adj_qty = self._apply_lot_constraints(symbol, quantity)
        if adj_qty <= 0:
            raise ValueError(f"Quantity {quantity} below minimum after rounding")
        
        is_buy = side == OrderSide.BUY
        
        try:
            # HL SDK: market_open doesn't support reduce_only parameter
            # For reduce-only, we'll need to use order() with market type
            if reduce_only:
                # Use order() method for reduce-only market orders
                # Note: market_open is for opening positions, not closing
                logger.warning(
                    f"Reduce-only market order not fully implemented for {symbol} - "
                    f"using market_open (may fail if no position exists)"
                )
            
            result = self._retry_api_call(
                self.exchange.market_open,
                name=symbol,
                is_buy=is_buy,
                sz=adj_qty,
            )
            
            if result.get("status") == "ok":
                logger.info(
                    f"Market order placed - Symbol: {symbol}, Side: {side.name}, "
                    f"Qty: {adj_qty}, ReduceOnly: {reduce_only}"
                )
                return {
                    "success": True,
                    "result": result.get("response"),
                }
            else:
                error = result.get("response", "Unknown error")
                logger.error(
                    f"Market order failed - Symbol: {symbol}, Side: {side.name}, "
                    f"Error: {error}"
                )
                return {
                    "success": False,
                    "error": error,
                }
        except Exception as e:
            logger.error(
                f"Exception placing market order - Symbol: {symbol}, "
                f"Side: {side.name}, Error: {e}"
            )
            raise

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place a limit order (for take profit).

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order quantity
            limit_price: Limit price
            reduce_only: If True, order only reduces position

        Returns:
            Order result dict
        """
        # Constrain quantity and price
        adj_qty = self._apply_lot_constraints(symbol, quantity)
        if adj_qty <= 0:
            raise ValueError(f"Quantity {quantity} below minimum after rounding")
        
        info = self._get_instrument_info(symbol)
        adj_price = self._round_to_tick(limit_price, info.get("tick_size", 0.01))
        
        is_buy = side == OrderSide.BUY
        
        try:
            # HL SDK: limit order
            result = self._retry_api_call(
                self.exchange.order,
                name=symbol,
                is_buy=is_buy,
                sz=adj_qty,
                limit_px=adj_price,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=reduce_only,
            )
            
            if result.get("status") == "ok":
                response = result.get("response", {})
                logger.info(
                    f"Limit order placed - Symbol: {symbol}, Side: {side.name}, "
                    f"Qty: {adj_qty}, Limit: {adj_price}, ReduceOnly: {reduce_only}"
                )
                return {
                    "success": True,
                    "result": response,
                    "hl_order_id": self._extract_order_id(response),
                }
            else:
                error = result.get("response", "Unknown error")
                logger.error(
                    f"Limit order failed - Symbol: {symbol}, Side: {side.name}, Error: {error}"
                )
                return {
                    "success": False,
                    "error": error,
                }
        except Exception as e:
            logger.error(
                f"Exception placing limit order - Symbol: {symbol}, "
                f"Side: {side.name}, Error: {e}"
            )
            raise

    def place_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place a stop market order (for stop loss).

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order quantity
            stop_price: Trigger price
            reduce_only: If True, order only reduces position

        Returns:
            Order result dict
        """
        # Constrain quantity and price
        adj_qty = self._apply_lot_constraints(symbol, quantity)
        if adj_qty <= 0:
            raise ValueError(f"Quantity {quantity} below minimum after rounding")
        
        info = self._get_instrument_info(symbol)
        adj_price = self._round_to_tick(stop_price, info.get("tick_size", 0.01))
        
        is_buy = side == OrderSide.BUY
        
        try:
            # HL SDK: stop market order
            result = self._retry_api_call(
                self.exchange.order,
                name=symbol,
                is_buy=is_buy,
                sz=adj_qty,
                limit_px=adj_price,  # Trigger price
                order_type={"trigger": {"triggerPx": adj_price, "isMarket": True, "tpsl": "sl"}},
                reduce_only=reduce_only,
            )
            
            if result.get("status") == "ok":
                response = result.get("response", {})
                logger.info(
                    f"Stop order placed - Symbol: {symbol}, Side: {side.name}, "
                    f"Qty: {adj_qty}, Stop: {adj_price}, ReduceOnly: {reduce_only}"
                )
                return {
                    "success": True,
                    "result": response,
                    "hl_order_id": self._extract_order_id(response),
                }
            else:
                error = result.get("response", "Unknown error")
                logger.error(
                    f"Stop order failed - Symbol: {symbol}, Side: {side.name}, Error: {error}"
                )
                return {
                    "success": False,
                    "error": error,
                }
        except Exception as e:
            logger.error(
                f"Exception placing stop order - Symbol: {symbol}, "
                f"Side: {side.name}, Error: {e}"
            )
            raise

    def flatten_position(self, symbol: str) -> bool:
        """Emergency flatten of current position.

        Args:
            symbol: Trading pair

        Returns:
            True if successful, False otherwise
        """
        position = self.get_position(symbol)
        if not position or position["size"] == 0:
            logger.warning(f"No position to flatten for {symbol}")
            return False
        
        size = position["size"]
        # If size > 0 (long), we sell; if size < 0 (short), we buy
        side = OrderSide.SELL if size > 0 else OrderSide.BUY
        qty = abs(size)
        
        try:
            result = self.place_market_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                reduce_only=True,
            )
            
            if result.get("success"):
                logger.info(f"Position flattened for {symbol}: {qty} contracts")
                return True
            else:
                logger.error(f"Failed to flatten position for {symbol}: {result}")
                return False
        except Exception as e:
            logger.error(f"Exception flattening position for {symbol}: {e}")
            return False

    def cancel_order(
        self,
        symbol: str,
        strategy_order_id: str,
    ) -> bool:
        """Cancel an open order.

        Args:
            symbol: Trading pair
            strategy_order_id: Strategy's order ID (e.g., "Long Exit")

        Returns:
            True if cancelled successfully, False otherwise
        """
        # Get HL order ID from our mapping
        symbol_map = self.order_id_map.get(symbol, {})
        hl_order_id = symbol_map.get(strategy_order_id)
        
        if hl_order_id is None:
            logger.debug(
                f"No tracked order to cancel - Symbol: {symbol}, "
                f"Strategy ID: {strategy_order_id}"
            )
            return False
        
        try:
            result = self._retry_api_call(
                self.exchange.cancel,
                name=symbol,
                oid=hl_order_id,
            )
            
            if result.get("status") == "ok":
                logger.info(
                    f"Order cancelled - Symbol: {symbol}, "
                    f"Strategy ID: {strategy_order_id}, HL ID: {hl_order_id}"
                )
                # Remove from tracking
                del symbol_map[strategy_order_id]
                return True
            else:
                error = result.get("response", "Unknown error")
                logger.warning(
                    f"Order cancel failed - Symbol: {symbol}, "
                    f"Strategy ID: {strategy_order_id}, Error: {error}"
                )
                return False
                
        except Exception as e:
            logger.error(
                f"Exception cancelling order - Symbol: {symbol}, "
                f"Strategy ID: {strategy_order_id}, Error: {e}"
            )
            return False

    def _store_order_id(self, symbol: str, strategy_order_id: str, hl_order_id: int) -> None:
        """Store mapping between strategy order ID and HL order ID.

        Args:
            symbol: Trading symbol
            strategy_order_id: Strategy's order ID (e.g., "Long Exit")
            hl_order_id: HyperLiquid's numeric order ID
        """
        if symbol not in self.order_id_map:
            self.order_id_map[symbol] = {}
        
        self.order_id_map[symbol][strategy_order_id] = hl_order_id
        logger.debug(
            f"Stored order ID - Symbol: {symbol}, "
            f"Strategy ID: {strategy_order_id}, HL ID: {hl_order_id}"
        )

    def _clear_order_ids(self, symbol: str) -> None:
        """Clear all order ID mappings for a symbol (called when position closes).

        Args:
            symbol: Trading symbol
        """
        if symbol in self.order_id_map:
            count = len(self.order_id_map[symbol])
            self.order_id_map[symbol] = {}
            logger.debug(f"Cleared {count} order ID mappings for {symbol}")

    def _extract_order_id(self, response: dict) -> Optional[int]:
        """Extract HL order ID from API response.

        Args:
            response: HL API response dict

        Returns:
            Order ID if found, None otherwise
        """
        try:
            # HL response format: {data: {statuses: [{resting: {oid: 12345}}]}}
            if isinstance(response, dict):
                statuses = response.get("data", {}).get("statuses", [])
                if statuses and isinstance(statuses[0], dict):
                    resting = statuses[0].get("resting")
                    if resting and isinstance(resting, dict):
                        oid = resting.get("oid")
                        if oid is not None:
                            return int(oid)
            return None
        except (KeyError, IndexError, ValueError, TypeError):
            return None

    def reconcile_position(
        self,
        symbol: str,
        expected_size: Optional[float] = None,
        action_on_mismatch: str = "warn",
    ) -> dict[str, Any]:
        """Reconcile HyperLiquid position with expected state.

        Args:
            symbol: Trading symbol
            expected_size: Expected position size (None = just query, don't compare)
            action_on_mismatch: Action if mismatch: "warn", "flatten", "error"

        Returns:
            Dict with reconciliation result:
            {
                "hl_position": float,
                "expected": float,
                "match": bool,
                "action_taken": str,
            }
        """
        position = self.get_position(symbol)
        hl_size = position.get("size", 0.0) if position else 0.0
        
        result = {
            "hl_position": hl_size,
            "expected": expected_size,
            "match": True,
            "action_taken": "none",
        }
        
        # If no expected size provided, just return current state
        if expected_size is None:
            logger.info(f"Position query - Symbol: {symbol}, Size: {hl_size}")
            return result
        
        # Check for mismatch
        if abs(hl_size - expected_size) > 0.0001:  # Floating point tolerance
            result["match"] = False
            
            logger.warning(
                f"⚠️ Position mismatch - Symbol: {symbol}, "
                f"HL: {hl_size}, Expected: {expected_size}, "
                f"Diff: {hl_size - expected_size}"
            )
            
            if action_on_mismatch == "flatten":
                logger.warning(f"Flattening {symbol} position due to mismatch")
                flattened = self.flatten_position(symbol)
                result["action_taken"] = "flattened" if flattened else "flatten_failed"
            elif action_on_mismatch == "error":
                raise RuntimeError(
                    f"Position mismatch - Symbol: {symbol}, "
                    f"HL: {hl_size}, Expected: {expected_size}"
                )
            else:
                result["action_taken"] = "warned"
        else:
            logger.debug(f"Position reconciled - Symbol: {symbol}, Size: {hl_size}")
        
        return result

    def place_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: float,
        leverage: Optional[float] = None,
    ) -> dict[str, Any]:
        """Place entry order with stop loss and take profit (bracket order).

        Places three orders in sequence:
        1. Market entry order
        2. Stop loss order (reduce-only)
        3. Take profit order (reduce-only)

        If stop loss or take profit placement fails, the position is immediately
        flattened for safety. HyperLiquid automatically creates OCO relationship
        between TP and SL once both are placed.

        Args:
            symbol: Trading pair (e.g., "BTC")
            side: BUY for long, SELL for short
            quantity: Order quantity
            stop_price: Stop loss trigger price
            limit_price: Take profit limit price
            leverage: Optional leverage to set (e.g., 5.0)

        Returns:
            Dict with success status and order results:
            {
                "success": bool,
                "entry": dict,
                "stop_loss": dict,
                "take_profit": dict,
                "errors": list[str],
                "flattened": bool (only if protective flatten was needed),
            }
        """
        logger.info(
            "Placing bracket order",
            symbol=symbol,
            side=side.name,
            quantity=quantity,
            stop_price=stop_price,
            limit_price=limit_price,
        )

        results: dict[str, Any] = {
            "success": False,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "errors": [],
        }

        # Set leverage if specified
        if leverage is not None:
            self._ensure_leverage(symbol, leverage)

        # Check for existing position
        position = self.get_position(symbol)
        if position and position.get("size", 0) != 0:
            logger.warning(
                "Position already exists, skipping bracket order",
                symbol=symbol,
                position_size=position.get("size"),
            )
            results["errors"].append("Position already exists")
            return results

        # Step 1: Place entry order
        try:
            entry_result = self.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                reduce_only=False,
            )
            results["entry"] = entry_result

            if not entry_result.get("success"):
                error_msg = entry_result.get("error", "Unknown error")
                results["errors"].append(f"Entry order failed: {error_msg}")
                logger.error(
                    "Entry order failed",
                    symbol=symbol,
                    error=error_msg,
                )
                return results

            logger.info(f"Entry order placed successfully for {symbol}")

        except Exception as e:
            error_msg = str(e)
            results["errors"].append(f"Entry order exception: {error_msg}")
            logger.error(
                "Exception placing entry order",
                symbol=symbol,
                error=error_msg,
                exc_info=True,
            )
            return results

        # Step 2: Place stop loss order
        opposite_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        try:
            sl_result = self.place_stop_order(
                symbol=symbol,
                side=opposite_side,
                quantity=quantity,
                stop_price=stop_price,
                reduce_only=True,
            )
            results["stop_loss"] = sl_result

            if not sl_result.get("success"):
                error_msg = sl_result.get("error", "Unknown error")
                results["errors"].append(f"Stop loss failed: {error_msg}")
                logger.error(
                    "Stop loss order failed, flattening position",
                    symbol=symbol,
                    error=error_msg,
                )
                # Protective flatten
                flattened = self.flatten_position(symbol)
                results["flattened"] = flattened
                if not flattened:
                    results["errors"].append("Failed to flatten after SL failure"                )
                return results

            logger.info(f"Stop loss order placed successfully for {symbol}")

        except Exception as e:
            error_msg = str(e)
            results["errors"].append(f"Stop loss exception: {error_msg}")
            logger.error(
                "Exception placing stop loss, flattening position",
                symbol=symbol,
                error=error_msg,
                exc_info=True,
            )
            # Protective flatten
            flattened = self.flatten_position(symbol)
            results["flattened"] = flattened
            if not flattened:
                results["errors"].append("Failed to flatten after SL exception")
            return results

        # Step 3: Place take profit order
        try:
            tp_result = self.place_limit_order(
                symbol=symbol,
                side=opposite_side,
                quantity=quantity,
                limit_price=limit_price,
                reduce_only=True,
            )
            results["take_profit"] = tp_result

            if not tp_result.get("success"):
                error_msg = tp_result.get("error", "Unknown error")
                results["errors"].append(f"Take profit failed: {error_msg}")
                logger.error(
                    "Take profit order failed, flattening position",
                    symbol=symbol,
                    error=error_msg,
                )
                # Protective flatten
                flattened = self.flatten_position(symbol)
                results["flattened"] = flattened
                if not flattened:
                    results["errors"].append("Failed to flatten after TP failure"                )
                return results

            logger.info(f"Take profit order placed successfully for {symbol}")

        except Exception as e:
            error_msg = str(e)
            results["errors"].append(f"Take profit exception: {error_msg}")
            logger.error(
                "Exception placing take profit, flattening position",
                symbol=symbol,
                error=error_msg,
                exc_info=True,
            )
            # Protective flatten
            flattened = self.flatten_position(symbol)
            results["flattened"] = flattened
            if not flattened:
                results["errors"].append("Failed to flatten after TP exception")
            return results

        # All orders placed successfully
        # HyperLiquid automatically creates OCO between TP and SL
        results["success"] = True
        logger.info(
            "Bracket order completed successfully (OCO active)",
            symbol=symbol,
            side=side.name,
            quantity=quantity,
        )
        return results

