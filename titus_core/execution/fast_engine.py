
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from titus_core.utils.config import EngineConfig

# Helper to convert to numpy array (handles both pandas Series and numpy arrays)
# Defined at module level for performance - only created once, not per-backtest
def _to_numpy(arr):
    if hasattr(arr, 'values'):
        return arr.values
    return np.asarray(arr)

@dataclass
class FastBacktestResult:
    final_equity: float
    trades_count: int
    win_rate: float
    total_return: float
    equity_curve: np.ndarray
    expectancy: float
    sqn: float
    profit_factor: float

def run_fast_backtest(
    arrays: Dict[str, np.ndarray],
    config: EngineConfig,
    initial_capital: float = 10000.0,
) -> FastBacktestResult:
    """
    Run a backtest using primitive variables and numpy arrays for maximum speed.
    Mimics BarExecutionEngine's 'entry on signal, exit on bracket' logic.
    """
    # 1. Unpack Arrays (convert to numpy for uniform indexing)
    opens = _to_numpy(arrays["open"])
    highs = _to_numpy(arrays["high"])
    lows = _to_numpy(arrays["low"])
    closes = _to_numpy(arrays["close"])
    
    # Strategy Signals (bool arrays)
    long_entries = arrays.get("long_entries")
    short_entries = arrays.get("short_entries")
    
    # ATR for dynamic bracketing
    atrs = arrays.get("atr")
    
    # Check if we have necessary data
    if long_entries is None or short_entries is None or atrs is None:
        raise ValueError("Missing 'long_entries', 'short_entries', or 'atr' arrays.")
    
    # Convert signals and ATR to numpy as well
    long_entries = _to_numpy(long_entries)
    short_entries = _to_numpy(short_entries)
    atrs = _to_numpy(atrs)

    count = len(opens)
    
    # Config Unpacking (Local vars are faster)
    commission = config.commission
    slippage = config.slippage
    slippage_type = config.slippage_type
    tick_size = config.tick_size if config.tick_size else 0.01
    use_leverage = config.use_leverage
    leverage = config.leverage if use_leverage else 1.0
    initial_cap = initial_capital
    pyramiding = config.pyramiding
    
    # Strategy settings from metadata (ATR Multipliers)
    # CRITICAL FIX: Convert all metadata values to native Python floats
    metadata = arrays.get("metadata", {})
    atr_sl_mult = float(metadata.get("atr_sl_mult", 1.0))
    atr_tp_mult = float(metadata.get("atr_tp_mult", 4.0))
    risk_pct = float(metadata.get("risk_pct", 2.0))
    max_leverage = float(metadata.get("max_leverage", 5.0))
    step_size = float(metadata.get("step_size", 0.001))


    
    # State Variables
    cash = initial_cap
    position_size = 0.0
    position_avg_price = 0.0

    
    # Bracket State
    pending_stop_dist = 0.0
    pending_target_dist = 0.0
    
    # Pending Entry (for NEXT_BAR_OPEN timing)
    pending_entry_side = 0
    pending_entry_qty = 0.0
    pending_entry_stop_dist = 0.0
    pending_entry_target_dist = 0.0

    
    # Metrics
    equity_curve = np.zeros(count, dtype=np.float64)
    trades_count = 0
    wins = 0
    
    # PnL Tracking for SQN and PF
    sum_pnl = 0.0
    sum_pnl_sq = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    
    # Constants
    SIDE_LONG = 1
    SIDE_SHORT = -1
    SIDE_NONE = 0
    

    
    # --- MAIN LOOP ---
    for i in range(count):
        open_p = opens[i]
        high_p = highs[i]
        low_p = lows[i]
        close_p = closes[i]
        
        # 0. EXECUTE PENDING ENTRY (NEXT_BAR_OPEN timing)
        if pending_entry_side != SIDE_NONE and position_size == 0.0:
            qty = pending_entry_qty
            entry_side = pending_entry_side
            
            if qty > 0:
                fill_price = open_p
                
                # Apply entry slippage
                if slippage > 0:
                    if slippage_type == 'percent':
                        skim = fill_price * slippage
                    else:
                        skim = slippage * tick_size
                    
                    if entry_side == SIDE_LONG:
                        fill_price += skim
                    else:
                        fill_price -= skim
                
                # Deduct margin + commission
                comm = fill_price * qty * commission
                notional = fill_price * qty
                margin_required = notional / leverage if use_leverage else notional
                cash -= (margin_required + comm)
                
                # Update position
                position_size = qty * entry_side
                position_avg_price = fill_price

                pending_stop_dist = pending_entry_stop_dist
                pending_target_dist = pending_entry_target_dist
            
            # Clear pending entry
            pending_entry_side = SIDE_NONE
            pending_entry_qty = 0.0
            pending_entry_stop_dist = 0.0
            pending_entry_target_dist = 0.0

        
        # 1. CHECK EXITS (if position exists)
        if position_size != 0.0:
            size_abs = abs(position_size)
            side = SIDE_LONG if position_size > 0 else SIDE_SHORT
            
            # Determine Stop/Limit Prices directly from stored distances
            if side == SIDE_LONG:
                stop_price = position_avg_price - pending_stop_dist
                target_price = position_avg_price + pending_target_dist
            else:
                stop_price = position_avg_price + pending_stop_dist
                target_price = position_avg_price - pending_target_dist
            
            # Check Fills
            did_close = False
            fill_price = 0.0
            pnl_raw = 0.0
            
            # --- LONG EXIT ---
            if side == SIDE_LONG:
                # 1. Stop Loss check (Open/Low)
                if open_p <= stop_price:
                    fill_price = stop_price
                    did_close = True
                elif low_p <= stop_price:
                    fill_price = stop_price
                    did_close = True
                
                if did_close:
                    # Apply Slippage to Stop (Market Sell)
                    if slippage > 0:
                        if slippage_type == 'percent':
                            fill_price *= (1 - slippage)
                        else:
                            fill_price -= (slippage * tick_size)
                
                else:
                    # 2. Take Profit check (Open/High)
                    if open_p >= target_price:
                        fill_price = target_price
                        did_close = True
                    elif high_p >= target_price:
                        fill_price = target_price
                        did_close = True

                if did_close:
                    pnl_raw = (fill_price - position_avg_price) * size_abs
            
            # --- SHORT EXIT ---
            else:
                # 1. Stop Loss check (Open/High)
                if open_p >= stop_price:
                    fill_price = stop_price
                    did_close = True
                elif high_p >= stop_price:
                    fill_price = stop_price
                    did_close = True
                
                if did_close:
                    # Apply Slippage to Stop (Market Buy)
                    if slippage > 0:
                        if slippage_type == 'percent':
                            fill_price *= (1 + slippage)
                        else:
                            fill_price += (slippage * tick_size)
                else:
                    # 2. Take Profit check (Open/Low)
                    if open_p <= target_price:
                        fill_price = target_price
                        did_close = True
                    elif low_p <= target_price:
                        fill_price = target_price
                        did_close = True
                
                if did_close:
                    pnl_raw = (position_avg_price - fill_price) * size_abs


            # Process Execution
            if did_close:
                comm = fill_price * size_abs * commission
                pnl_net = pnl_raw - comm
                
                # Returns PnL to cash
                notional = position_avg_price * size_abs
                margin_released = notional / leverage if use_leverage else notional
                cash += (margin_released + pnl_net)
                
                # Reset Position
                position_size = 0.0
                position_avg_price = 0.0
                trades_count += 1
                if pnl_net > 0:
                    wins += 1
                
                # Track PnL Stats
                sum_pnl += pnl_net
                sum_pnl_sq += (pnl_net * pnl_net)
                if pnl_net > 0:
                    gross_win += pnl_net
                else:
                    gross_loss += abs(pnl_net)


        # 2. CHECK ENTRIES (if flat)
        if position_size == 0.0:
            is_long = long_entries[i]
            is_short = short_entries[i]
            
            if is_long and is_short:
                pass
            
            entry_side = SIDE_NONE
            if is_long:
                entry_side = SIDE_LONG
            elif is_short:
                entry_side = SIDE_SHORT
            
            if entry_side != SIDE_NONE:
                # Calculate Quantity
                atr_val = float(atrs[i])  # Ensure scalar
                stop_dist = atr_val * atr_sl_mult
                
                if stop_dist > 0 and open_p > 0:
                    # Calc Equity (Cash is equity since flat)
                    equity_now = cash
                    
                    # Estimate max quantity based on current price (approximate for Open, exact for Close)
                    price_for_sizing = close_p if config.process_orders_on_close else open_p
                    
                    risk_val = equity_now * (risk_pct / 100.0)
                    qty = risk_val / stop_dist
                    max_qty = (equity_now * max_leverage) / price_for_sizing
                    qty = min(qty, max_qty)
                    
                    # Rounding
                    if step_size > 0:
                        qty = round(qty / step_size) * step_size
                    
                    if qty > 0:
                        # CHECK EXECUTION TIMING
                        if config.process_orders_on_close:
                            # EXECUTE IMMEDIATELY ON CLOSE
                            fill_price = close_p
                            
                            # Apply entry slippage
                            if slippage > 0:
                                if slippage_type == 'percent':
                                    skim = fill_price * slippage
                                else:
                                    skim = slippage * tick_size
                                
                                if entry_side == SIDE_LONG:
                                    fill_price += skim
                                else:
                                    fill_price -= skim
                            
                            # Deduct margin + commission
                            comm = fill_price * qty * commission
                            notional = fill_price * qty
                            margin_required = notional / leverage if use_leverage else notional
                            cash -= (margin_required + comm)
                            
                            # Update position
                            position_size = qty * entry_side
                            position_avg_price = fill_price

                            pending_stop_dist = stop_dist
                            pending_target_dist = atr_val * atr_tp_mult
                            
                            # Note: No pending entry needed
                        
                        else:
                            # SET PENDING ENTRY (will execute next bar at open)
                            pending_entry_side = entry_side
                            pending_entry_qty = qty
                            pending_entry_stop_dist = stop_dist
                            pending_entry_target_dist = atr_val * atr_tp_mult

        
        # 3. UPDATE EQUITY CURVE
        equity_val = cash
        if position_size != 0.0:
            size_abs = abs(position_size)
            if position_size > 0:
                unrealized = (close_p - position_avg_price) * size_abs
            else:
                unrealized = (position_avg_price - close_p) * size_abs
            
            # Add back margin that's tied up in the position
            notional = position_avg_price * size_abs
            margin_in_use = notional / leverage if use_leverage else notional
            equity_val += (margin_in_use + unrealized)
        
        equity_curve[i] = equity_val

    # Finalize
    total_return = (equity_curve[-1] - initial_cap) / initial_cap if initial_cap != 0 else 0.0
    win_rate = (wins / trades_count) if trades_count > 0 else 0.0
    
    # Calculate Expectancy, SQN, and Profit Factor
    expectancy = 0.0
    sqn = 0.0
    profit_factor = 0.0
    
    if trades_count > 0:
        mean_pnl = sum_pnl / trades_count
        expectancy = mean_pnl
        
        if trades_count > 1:
            variance = (sum_pnl_sq / trades_count) - (mean_pnl * mean_pnl)
            # Avoid negative variance due to precision
            if variance < 0: variance = 0.0
            std_dev = np.sqrt(variance)
            
            if std_dev > 0:
                sqn = np.sqrt(trades_count) * (mean_pnl / std_dev)
    
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = float('inf')
        
    return FastBacktestResult(
        final_equity=equity_curve[-1],
        trades_count=trades_count,
        win_rate=win_rate,
        total_return=total_return,
        equity_curve=equity_curve,
        expectancy=expectancy,
        sqn=sqn,
        profit_factor=profit_factor
    )
