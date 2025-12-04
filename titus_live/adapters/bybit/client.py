"""Bybit exchange adapter for live order execution.

Adapted from bybit-relay with interface matching HyperLiquidClient.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from pybit.unified_trading import HTTP

from titus_core.trading.orders import OrderSide, OrderType

logger = logging.getLogger(__name__)


class BybitClient:
    """Client for executing orders on Bybit exchange."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ) -> None:
        """Initialize Bybit client.

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret (NEVER logged)
            testnet: Use testnet (default: False for mainnet)
        """
        if not api_key or not api_secret:
            raise ValueError(
                "Bybit credentials not found. "
                "Please provide api_key and api_secret."
            )
        
        # SECURITY: Store but never log the API secret
        self._api_key = api_key.strip()
        self._api_secret = api_secret.strip()
        self.testnet = testnet
        
        # Validate credential format
        if len(self._api_key) < 10 or len(self._api_secret) < 10:
            logger.warning(
                "API credentials seem short - check for truncation",
                api_key_length=len(self._api_key),
                api_secret_length=len(self._api_secret),
            )
        
        # Initialize Bybit HTTP client
        self.client = HTTP(
            testnet=testnet,
            api_key=self._api_key,
            api_secret=self._api_secret,
        )
        
        # Cache for instrument info and leverage
        self.instrument_cache: dict[str, dict[str, Any]] = {}
        self.leverage_cache: dict[str, float] = {}
        
        # Order ID tracking: strategy_order_id -> bybit_order_id
        # Format: {symbol: {strategy_id: bybit_oid}}
        self.order_id_map: dict[str, dict[str, str]] = {}
        
        logger.info(
            f"Bybit client initialized - "
            f"API Key: {self._api_key[:6]}...{self._api_key[-4:]}, "
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
                response = func(*args, **kwargs)
                
                # Check for successful response
                ret_code = response.get("retCode", 0)
                if ret_code == 0:
                    return response
                
                # Check for auth/permission errors (don't retry)
                # Bybit codes: 10001 (invalid key), 10002 (invalid sig), etc.
                ret_msg = response.get("retMsg", "")
                if ret_code in (10001, 10002, 10003, 10004, 10005, 10006):
                    logger.error(
                        "Authentication/permission error (not retrying)",
                        ret_code=ret_code,
                        ret_msg=ret_msg,
                    )
                    raise Exception(f"Authentication failed: {ret_msg}")
                
                logger.warning(
                    "API returned non-zero retCode",
                    ret_code=ret_code,
                    ret_msg=ret_msg,
                    attempt=attempt + 1,
                )
                return response
                
            except Exception as e:
                error_str = str(e)
                
                # Check for HTTP 401 auth errors
                if "401" in error_str or "Unauthorized" in error_str:
                    logger.error(
                        "HTTP 401 Authentication error (not retrying)",
                        error=error_str,
                    )
                    raise Exception(
                        f"Authentication failed: {error_str}. "
                        "Check your API key, API secret, and permissions."
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
        
        Prioritizes available balance over total equity to prevent over-leveraging
        in multi-symbol mode (mirrors HyperLiquid behavior).

        Returns:
            Available equity/margin in USD

        Raises:
            Exception: If equity cannot be fetched
        """
        logger.debug("Fetching equity from Bybit")
        
        try:
            response = self._retry_api_call(
                self.client.get_wallet_balance,
                accountType="UNIFIED",  # Unified Trading Account
            )
            
            ret_code = response.get("retCode", 0)
            ret_msg = response.get("retMsg", "")
            
            if ret_code != 0:
                logger.error(
                    f"Failed to fetch equity - retCode={ret_code}, retMsg={ret_msg}"
                )
                raise Exception(f"Failed to fetch equity: {ret_msg}")
            
            balances = response.get("result", {}).get("list", [])
            if not balances:
                logger.error("No balance data returned from Bybit")
                raise Exception("No balance data returned from Bybit")
            
            entry = balances[0]
            value = None
            chosen_field = None
            
            # Prefer availableBalance to prevent race conditions in multi-symbol
            for key in (
                "availableBalance",
                "availableToWithdraw",
                "totalEquity",
                "walletBalance",
                "equity",
            ):
                try:
                    v = float(entry.get(key, 0.0) or 0.0)
                    if v > 0:
                        value = v
                        chosen_field = key
                        break
                except (ValueError, TypeError):
                    continue
            
            if value is None or value <= 0:
                logger.error(
                    f"Equity returned is zero or invalid - "
                    f"Available fields: {list(entry.keys())}"
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
            symbol: Trading pair symbol (e.g., "BTCUSDT")

        Returns:
            Mark price or None if unavailable
        """
        try:
            response = self._retry_api_call(
                self.client.get_tickers,
                category="linear",
                symbol=symbol,
            )
            
            if response.get("retCode") == 0:
                tickers = response.get("result", {}).get("list", [])
                if tickers:
                    mark_price = float(tickers[0].get("markPrice", 0))
                    return mark_price if mark_price > 0 else None
            
            logger.warning(f"Symbol {symbol} not found in tickers")
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
        """
        try:
            response = self._retry_api_call(
                self.client.get_positions,
                category="linear",
                symbol=symbol,
            )
            
            if response.get("retCode") == 0:
                positions = response.get("result", {}).get("list", [])
                for pos in positions:
                    try:
                        size = float(pos.get("size", 0))
                        if size != 0:
                            return {
                                "symbol": symbol,
                                "size": size,
                                "entry_price": float(pos.get("avgPrice", 0)),
                                "position_value": float(pos.get("positionValue", 0)),
                                "unrealized_pnl": float(pos.get("unrealisedPnl", 0)),
                                "leverage": float(pos.get("leverage", 1)),
                            }
                    except (ValueError, TypeError):
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

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
            response = self._retry_api_call(
                self.client.get_instruments_info,
                category="linear",
                symbol=symbol,
            )
            
            tick_size = 0.01
            lot_size = 0.001
            min_order_qty = 0.0
            
            if response.get("retCode") == 0:
                instruments = response.get("result", {}).get("list", [])
                if instruments:
                    info = instruments[0]
                    lot_filter = info.get("lotSizeFilter", {})
                    price_filter = info.get("priceFilter", {})
                    
                    lot_size = float(lot_filter.get("qtyStep", lot_size))
                    min_order_qty = float(lot_filter.get("minOrderQty", 0.0))
                    tick_size = float(price_filter.get("tickSize", tick_size))
            
            info_dict = {
                "tick_size": tick_size,
                "lot_size": lot_size,
                "min_order_qty": min_order_qty,
            }
            
            self.instrument_cache[symbol] = info_dict
            logger.debug(f"Cached instrument info for {symbol}: {info_dict}")
            return info_dict
            
        except Exception as e:
            logger.error(f"Failed to fetch instrument info for {symbol}: {e}")
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
            response = self._retry_api_call(
                self.client.set_leverage,
                category="linear",
                symbol=symbol,
                buyLeverage=str(int(leverage)),
                sellLeverage=str(int(leverage)),
            )
            
            if response.get("retCode") == 0:
                self.leverage_cache[symbol] = leverage
                logger.info(f"Leverage set for {symbol}: {leverage}x")
            else:
                logger.warning(f"Failed to set leverage for {symbol}: {response}")
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
        
        bybit_side = "Buy" if side == OrderSide.BUY else "Sell"
        
        try:
            response = self._retry_api_call(
                self.client.place_order,
                category="linear",
                symbol=symbol,
                side=bybit_side,
                orderType="Market",
                qty=str(adj_qty),
                reduceOnly=reduce_only,
            )
            
            if response.get("retCode") == 0:
                logger.info(
                    f"Market order placed - Symbol: {symbol}, Side: {side.name}, "
                    f"Qty: {adj_qty}, ReduceOnly: {reduce_only}"
                )
                return {
                    "success": True,
                    "result": response.get("result"),
                }
            else:
                error = response.get("retMsg", "Unknown error")
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
        
        bybit_side = "Buy" if side == OrderSide.BUY else "Sell"
        
        try:
            response = self._retry_api_call(
                self.client.place_order,
                category="linear",
                symbol=symbol,
                side=bybit_side,
                orderType="Limit",
                qty=str(adj_qty),
                price=str(adj_price),
                reduceOnly=reduce_only,
            )
            
            if response.get("retCode") == 0:
                result = response.get("result", {})
                order_id = result.get("orderId")
                logger.info(
                    f"Limit order placed - Symbol: {symbol}, Side: {side.name}, "
                    f"Qty: {adj_qty}, Limit: {adj_price}, ReduceOnly: {reduce_only}"
                )
                return {
                    "success": True,
                    "result": result,
                    "bybit_order_id": order_id,
                }
            else:
                error = response.get("retMsg", "Unknown error")
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
        
        bybit_side = "Buy" if side == OrderSide.BUY else "Sell"
        
        try:
            # Bybit uses Market order type with stopLoss parameter
            response = self._retry_api_call(
                self.client.place_order,
                category="linear",
                symbol=symbol,
                side=bybit_side,
                orderType="Market",
                qty=str(adj_qty),
                triggerPrice=str(adj_price),
                triggerBy="MarkPrice",
                reduceOnly=reduce_only,
            )
            
            if response.get("retCode") == 0:
                result = response.get("result", {})
                order_id = result.get("orderId")
                logger.info(
                    f"Stop order placed - Symbol: {symbol}, Side: {side.name}, "
                    f"Qty: {adj_qty}, Stop: {adj_price}, ReduceOnly: {reduce_only}"
                )
                return {
                    "success": True,
                    "result": result,
                    "bybit_order_id": order_id,
                }
            else:
                error = response.get("retMsg", "Unknown error")
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
        # Get Bybit order ID from our mapping
        symbol_map = self.order_id_map.get(symbol, {})
        bybit_order_id = symbol_map.get(strategy_order_id)
        
        if bybit_order_id is None:
            logger.debug(
                f"No tracked order to cancel - Symbol: {symbol}, "
                f"Strategy ID: {strategy_order_id}"
            )
            return False
        
        try:
            response = self._retry_api_call(
                self.client.cancel_order,
                category="linear",
                symbol=symbol,
                orderId=bybit_order_id,
            )
            
            if response.get("retCode") == 0:
                logger.info(
                    f"Order cancelled - Symbol: {symbol}, "
                    f"Strategy ID: {strategy_order_id}, Bybit ID: {bybit_order_id}"
                )
                # Remove from tracking
                del symbol_map[strategy_order_id]
                return True
            else:
                error = response.get("retMsg", "Unknown error")
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

    def _store_order_id(self, symbol: str, strategy_order_id: str, bybit_order_id: str) -> None:
        """Store mapping between strategy order ID and Bybit order ID.

        Args:
            symbol: Trading symbol
            strategy_order_id: Strategy's order ID (e.g., "Long Exit")
            bybit_order_id: Bybit's order ID
        """
        if symbol not in self.order_id_map:
            self.order_id_map[symbol] = {}
        
        self.order_id_map[symbol][strategy_order_id] = bybit_order_id
        logger.debug(
            f"Stored order ID - Symbol: {symbol}, "
            f"Strategy ID: {strategy_order_id}, Bybit ID: {bybit_order_id}"
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

    def reconcile_position(
        self,
        symbol: str,
        expected_size: Optional[float] = None,
        action_on_mismatch: str = "warn",
    ) -> dict[str, Any]:
        """Reconcile Bybit position with expected state.

        Args:
            symbol: Trading symbol
            expected_size: Expected position size (None = just query, don't compare)
            action_on_mismatch: Action if mismatch: "warn", "flatten", "error"

        Returns:
            Dict with reconciliation result:
            {
                "bybit_position": float,
                "expected": float,
                "match": bool,
                "action_taken": str,
            }
        """
        position = self.get_position(symbol)
        bybit_size = position.get("size", 0.0) if position else 0.0
        
        result = {
            "bybit_position": bybit_size,
            "expected": expected_size,
            "match": True,
            "action_taken": "none",
        }
        
        # If no expected size provided, just return current state
        if expected_size is None:
            logger.info(f"Position query - Symbol: {symbol}, Size: {bybit_size}")
            return result
        
        # Check for mismatch
        if abs(bybit_size - expected_size) > 0.0001:  # Floating point tolerance
            result["match"] = False
            
            logger.warning(
                f"⚠️ Position mismatch - Symbol: {symbol}, "
                f"Bybit: {bybit_size}, Expected: {expected_size}, "
                f"Diff: {bybit_size - expected_size}"
            )
            
            if action_on_mismatch == "flatten":
                logger.warning(f"Flattening {symbol} position due to mismatch")
                flattened = self.flatten_position(symbol)
                result["action_taken"] = "flattened" if flattened else "flatten_failed"
            elif action_on_mismatch == "error":
                raise RuntimeError(
                    f"Position mismatch - Symbol: {symbol}, "
                    f"Bybit: {bybit_size}, Expected: {expected_size}"
                )
            else:
                result["action_taken"] = "warned"
        else:
            logger.debug(f"Position reconciled - Symbol: {symbol}, Size: {bybit_size}")
        
        return result

