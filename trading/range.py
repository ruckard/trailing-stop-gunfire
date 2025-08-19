import db.positions as positionsdb
import state
from utils import print_with_date, debug
from exchange import btse as exchange

def place_range_positions(symbol, sides=("LONG", "SHORT"), lookback=50,
                          entry_offset_pct=0.5, take_profit_pct=0.7, stop_loss_pct=0.5):
    """
    Place range-trading orders: enter near support/resistance with tight SL/TP.
    """

    df = fetch_4h_ohlcv(symbol)
    if df is None or df.empty or len(df) < lookback:
        print_with_date(f"[RANGE STRATEGY] Insufficient data for {symbol}")
        return

    recent = df.tail(lookback)
    high = recent['high'].max()
    low = recent['low'].min()
    range_mid = (high + low) / 2

    contracts = CONTRACTS_MAP.get(symbol, 1)

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

        # Ensure price decimal scale is the right one
        entry_price = round(entry_price, int(-math.log10(MIN_PRICE_INCREMENTS[symbol])))
        take_profit = round(take_profit, int(-math.log10(MIN_PRICE_INCREMENTS[symbol])))
        stop_loss = round(stop_loss, int(-math.log10(MIN_PRICE_INCREMENTS[symbol])))

        cl_order_id = f"{symbol}-range-{side.lower()}-{i}-{int(time.time())}"

        print_with_date(
            f"[RANGE STRATEGY] {symbol} {side} | Entry: {entry_price:.4f}, "
            f"TP: {take_profit:.4f}, SL: {stop_loss:.4f}, Qty: {contracts}"
        )

        # You may need to customize this to your real API structure:
        result = place_range_order(symbol=symbol,
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
