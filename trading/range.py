import math
import time

import db.positions as positionsdb
import state
from utils import print_with_date, debug
from exchange import btse as exchange

from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN

def place_range_positions(symbol, sides=("LONG", "SHORT"), lookback=50,
                          entry_offset_pct=0.5, take_profit_pct=0.7, stop_loss_pct=0.5):
    """
    Place range-trading orders: enter near support/resistance with tight SL/TP.
    """

    df = exchange.fetch_5m_ohlcv(symbol)
    if df is None or df.empty or len(df) < lookback:
        print_with_date(f"[RANGE STRATEGY] Insufficient data for {symbol}")
        return

    recent = df.tail(lookback)
    high = recent['high'].max()
    low = recent['low'].min()
    range_mid = (high + low) / 2

    contracts = state.CONTRACTS_MAP.get(symbol, 1)

    price = exchange.get_current_price(symbol)
    if price is None:
        print_with_date(f"[RANGE STRATEGY] Failed to fetch price for {symbol}")
        return

    for i, side in enumerate(sides):
        if side == "LONG":
            entry_price = range_mid * (1 - entry_offset_pct / 100)
            take_profit = entry_price * (1 + take_profit_pct / 100)
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            order_side = "BUY"
        elif side == "SHORT":
            entry_price = range_mid * (1 + entry_offset_pct / 100)
            take_profit = entry_price * (1 - take_profit_pct / 100)
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            order_side = "SELL"
        else:
            continue

        print_with_date(f"[DEBUG-place_range_positions] (PRE-round) side: {side}")
        print_with_date(f"[DEBUG-place_range_positions] (PRE-round) entry_price: {entry_price}")
        print_with_date(f"[DEBUG-place_range_positions] (PRE-round) take_profit: {take_profit}")
        print_with_date(f"[DEBUG-place_range_positions] (PRE-round) stop_loss: {stop_loss}")

        min_price_increment = state.MIN_PRICE_INCREMENTS.get(symbol)
        if not min_price_increment:
            raise ValueError(f"No min price increment found for {symbol}")

        # Ensure price decimal scale is the right one
        entry_price = round(entry_price, int(-math.log10(state.MIN_PRICE_INCREMENTS[symbol])))
        take_profit = round(take_profit, int(-math.log10(state.MIN_PRICE_INCREMENTS[symbol])))
        stop_loss = round(stop_loss, int(-math.log10(state.MIN_PRICE_INCREMENTS[symbol])))


        # Long: Find a lower price to enter
        # so that we have a better entry price
        if (side == "SHORT"):
            ENTRY_PRICE_ROUND_SIDE=ROUND_HALF_UP
        else:
            ENTRY_PRICE_ROUND_SIDE=ROUND_HALF_DOWN

        # Long: Find a lower price to exit
        # to exit even with less profit
        if (side == "SHORT"):
            TAKE_PROFIT_ROUND_SIDE=ROUND_HALF_UP
        else:
            TAKE_PROFIT_ROUND_SIDE=ROUND_HALF_DOWN

        # Long: Find a higher price to exit
        # to exit with less loss
        if (side == "SHORT"):
            STOP_LOSS_ROUND_SIDE=ROUND_HALF_DOWN
        else:
            STOP_LOSS_ROUND_SIDE=ROUND_HALF_UP

        entry_price = Decimal(str(entry_price)).quantize(min_price_increment, rounding=ENTRY_PRICE_ROUND_SIDE)
        take_profit = Decimal(str(take_profit)).quantize(min_price_increment, rounding=TAKE_PROFIT_ROUND_SIDE)
        stop_loss  = Decimal(str(stop_loss)).quantize(min_price_increment, rounding=STOP_LOSS_ROUND_SIDE)

        print_with_date(f"[DEBUG-place_range_positions] (POST-round) side: {side}")
        print_with_date(f"[DEBUG-place_range_positions] (POST-round) entry_price: {entry_price}")
        print_with_date(f"[DEBUG-place_range_positions] (POST-round) take_profit: {take_profit}")
        print_with_date(f"[DEBUG-place_range_positions] (POST-round) stop_loss: {stop_loss}")

        cl_order_id = f"{symbol}-range-{side.lower()}-{i}-{int(time.time())}"

        print_with_date(f"[NEW] [RANGE] {symbol} {side} | Entry: {entry_price:.4f}, TP: {take_profit:.4f}, SL: {stop_loss:.4f}, Qty: {contracts}")

        # You may need to customize this to your real API structure:
        result = exchange.place_range_order(symbol=symbol,
                          position_side=order_side,
                          contracts=contracts,
                          entry_price=entry_price,
                          take_profit=take_profit,
                          stop_loss=stop_loss,
                          cl_order_id=cl_order_id)

        if result is None or result[0] is None:
            print_with_date(f"[ERROR] Failed to place range order for {symbol} {side}")
            continue

        # Optionally track position:
        pid = f"range-{side.lower()}-{i}"
        state.positions[symbol][pid] = {
            "position_id": cl_order_id,
            "opening_order_id": cl_order_id,
            "closing_order_id": None,
            "side": side,
            "callback": None,
            "active": True,
            "opening_price": entry_price,
            "trail_value": None,
            "opened_at": time.time()
        }
        positionsdb.update_position(pid, state.positions[symbol][pid], symbol)
