from . import trend
from decimal import Decimal

import state

def build_trailing_stops_map():
    result = {}
    for symbol, cfg in state.SYMBOL_CONFIGS.items():
        trailing_start = trend.calculate_trailing_start_from_atr(symbol)
        if trailing_start is None:
            continue  # or raise/log error

        trailing_start_decimal = Decimal(str(trailing_start))
        step_multiplier = Decimal(str(cfg.get("TRAILING_STEP_MULTIPLIER", state.DEFAULT_TRAILING_STEP_MULTIPLIER)))
        trailing_step = trailing_start_decimal * step_multiplier
        trailing_count = cfg.get("TRAILING_COUNT", state.DEFAULT_TRAILING_COUNT)

        result[symbol] = [
            float(round(trailing_start_decimal + i * trailing_step, 8))
            for i in range(trailing_count)
        ]
    return result

def update_trailing_stops_for_symbol(symbol):
    cfg = state.SYMBOL_CONFIGS.get(symbol, {})

    trailing_start = trend.calculate_trailing_start_from_atr(symbol)
    if trailing_start is None:
        print_with_date(f"[ERROR] Could not calculate trailing start for {symbol}")
        return

    trailing_start = Decimal(str(trailing_start))  # Ensure Decimal type
    step_multiplier = Decimal(str(cfg.get("TRAILING_STEP_MULTIPLIER", TRAILING_STEP_MULTIPLIER_DEFAULT)))

    trailing_step = trailing_start * step_multiplier
    trailing_count = cfg.get("TRAILING_COUNT", TRAILING_COUNT_DEFAULT)

    TRAILING_STOPS_MAP[symbol] = [
        round(trailing_start + i * trailing_step, 2)
        for i in range(trailing_count)
    ]

    print_with_date(f"[UPDATED TRAILING STOPS] {symbol}: {TRAILING_STOPS_MAP[symbol]}")

# === Place Trailing Stop Order on BTSE ===
def place_trailing_stop(symbol, position_side, callback_rate, contracts):
    try:
        current_price = get_current_price(symbol)
        if not current_price:
            print_with_date("[ERROR] Failed to get current price.")
            return None, None, None, None, None
        callback_rate_float = float(callback_rate)

        min_price_increment = state.MIN_PRICE_INCREMENTS.get(symbol)
        if not min_price_increment:
            raise ValueError(f"No min price increment found for {symbol}")

        precision = abs(Decimal(str(min_price_increment)).as_tuple().exponent)
        trail_value = round(current_price * (callback_rate_float / 100), precision)

        side = "BUY" if position_side == "SHORT" else "SELL"  # Closing side
        market_side = "SELL" if position_side == "SHORT" else "BUY"  # Entry side

        url_path = '/api/v2.2/order'
        full_url = BASE_URL + url_path

        # === Market Order ===
        debug(f"[DEBUG] Placing MARKET order: {market_side} {contracts} contracts")
        nonce = str(int(time.time() * 1000))
        market_order = {
            "postOnly": False,
            "price": 0.0,
            "reduceOnly": False,
            "side": market_side,
            "size": contracts,
            "symbol": symbol,
            "time_in_force": "GTC",
            "type": "MARKET",
            "txType": "LIMIT",
            "positionMode": "ISOLATED"
        }
        market_body_str = json.dumps(market_order, separators=(',', ':'))
        market_sig = generate_signature(API_SECRET, url_path, nonce, market_body_str)
        market_headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': market_sig,
            'Content-Type': 'application/json'
        }

        debug(f"MARKET order payload: {market_body_str}")
        market_response = throttled_request('POST', full_url, headers=market_headers, data=market_body_str)
        debug(f"MARKET order response status: {market_response.status_code}")
        debug(f"MARKET order response body: {market_response.text}")
        market_response.raise_for_status()
        market_data = market_response.json()
        if not isinstance(market_data, list) or not market_data:
            print_with_date("[ERROR] Unexpected market order response.")
            return None, None, None, None, None

        position_id = market_data[0].get('positionId')
        if not position_id:
            print_with_date("[ERROR] Missing position ID.")
            return None, None, None, None, None

        opening_order_id = market_data[0].get('orderID')
        opening_price = market_data[0].get('price')

        debug(f"Placing TRAILING STOP order: {side} with trail {trail_value}")
        nonce = str(int(time.time() * 1000))
        trail_order = {
            "postOnly": False,
            "price": 0.0,
            "reduceOnly": True,
            "side": side,
            "size": contracts,
            "symbol": symbol,
            "time_in_force": "GTC",
            "trailValue": -trail_value if side == "SELL" else trail_value,
            "type": "MARKET",
            "txType": "STOP",
            "positionMode": "ISOLATED",
            "positionId": position_id
        }
        trail_body_str = json.dumps(trail_order, separators=(',', ':'))
        trail_sig = generate_signature(API_SECRET, url_path, nonce, trail_body_str)
        trail_headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': trail_sig,
            'Content-Type': 'application/json'
        }

        debug(f"TRAILING STOP order payload: {trail_body_str}")
        trail_response = throttled_request('POST', full_url, headers=trail_headers, data=trail_body_str)
        debug(f"TRAILING STOP response status: {trail_response.status_code}")
        debug(f"TRAILING STOP response body: {trail_response.text}")
        trail_response.raise_for_status()
        trail_data = trail_response.json()
        closing_order_id = trail_data[0].get("orderID") if trail_data else None

        if not closing_order_id:
            print_with_date("[ERROR] Missing trailing stop order ID.")
            return None, None, None, None, None

        print_with_date(f"[NEW] {symbol} | {position_side} | Callback: {callback_rate}%")
        return position_id, opening_order_id, closing_order_id, opening_price, trail_value
    except Exception as e:
        print_with_date(f"[ERROR] Failed to place order: {e}")
        return None, None, None, None, None

# === Place All Positions ===
def place_all_positions(symbol, sides=("LONG", "SHORT")):
    global CONTRACTS_MAP
    print_with_date(f"[STARTING NEW {symbol} CYCLE]")
    positions[symbol].clear()
    clear_positions(symbol)
    trend_type = classify_trend_or_range(symbol)

    if trend_type == "trend":
        place_trend_positions(symbol, sides)
    elif trend_type == "range":
        place_range_positions(symbol, sides, entry_offset_pct=RANGE_ENTRY_OFFSET_PCT, take_profit_pct=RANGE_TAKE_PROFIT_PCT, stop_loss_pct=RANGE_STOP_LOSS_PCT)
    else:
        print_with_date(f"[SKIP] Could not classify trend/range for {symbol}")

def place_trend_positions(symbol, sides):
    for i, callback in enumerate(TRAILING_STOPS_MAP[symbol]):
        #for side in ["LONG"]:
        for side in sides:
            pid = f"{side.lower()}-{i}"
            contracts = CONTRACTS_MAP.get(symbol, 1)
            result = place_trailing_stop(symbol, side, callback, contracts)
            # Check if the result is valid (i.e., position_id and opening_order_id and closing_order_id are returned)
            if result is None or result[0] is None or result[1] is None or result[2] is None:
                print_with_date(f"[ERROR] Failed to place trailing stop for {symbol} {side} at {callback}%")
                continue
            pos_id, opening_order_id, closing_order_id, opening_price, trail_value = result
            positions[symbol][pid] = {
                "position_id": pos_id,
                "opening_order_id": opening_order_id,
                "closing_order_id": closing_order_id,
                "side": side,
                "callback": callback,
                "active": True,
                "opening_price" : opening_price,
                "trail_value" : trail_value,
                "opened_at": time.time()
            }
            position_info = positions[symbol][pid]
            update_position(pid, position_info, symbol)

# === Check and Manage Positions ===
def check_positions(symbol):
    global CONTRACTS_MAP
    all_closed = True
    for pid, info in positions[symbol].items():
        debug(f"[check_positions] Checking... {symbol} {pid}")
        if not info["active"]:
            continue

        opened_at = info.get("opened_at")
        if opened_at:
            elapsed_minutes = (time.time() - opened_at) / 60
            if elapsed_minutes >= TRADE_MAX_CANDLES * CANDLE_INTERVAL_MINUTES:
                print_with_date(f"[TIMEOUT] Closing {symbol} {pid} after {elapsed_minutes:.1f} minutes.")
                # Code to close the position immediately:
                close_position(symbol, info)  # You'll need to implement or call your existing close logic
                info["active"] = False
                update_position(pid, info, symbol)
                continue

        position_data = get_position_status(info["position_id"])

        if position_data == "_network_error_":
            print_with_date(f"[SKIPPING] {symbol} {pid} due to network timeout. Will retry later.")
            all_closed = False
            return all_closed

        if position_data == "_unexpected_error_":
            print_with_date(f"[SKIPPING] {symbol} {pid} due to unexpected error. Will retry later.")
            all_closed = False
            return all_closed

        if not position_data:
            print_with_date(f"[CLOSED?] {symbol} {pid} position_id not found. Checking trade history...")

            # Closing trade check
            time.sleep(1)
            trade = get_trade_by_closing_order_id(symbol, info["closing_order_id"])

            if not trade:
                print_with_date(f"[ERROR] No closing trade found for order_id {info['closing_order_id']}")
                info["active"] = False
                update_position(pid, info, symbol)
                continue
            pnl1 = Decimal(str(trade.get("total")))

            # Opening trade check
            time.sleep(1)
            trade = get_trade_by_opening_order_id(symbol, info["opening_order_id"])
            if not trade:
                print_with_date(f"[ERROR] No opening trade found for order_id {info['opening_order_id']}")
                continue
            pnl2 = Decimal(str(trade.get("total")))
            closing_price = Decimal(str(trade.get("price")))

            pnl = pnl1 + pnl2

            debug(f"[CLOSED/TRADE] {symbol} {pid} | Realized PnL1: {pnl1:.8f}")
            debug(f"[CLOSED/TRADE] {symbol} {pid} | Realized PnL2: {pnl2:.8f}")
            print_with_date(f"[CLOSED/TRADE] {symbol} {pid} | Realized PnL: {pnl:.8f}")
            info["active"] = False
            update_position(pid, info, symbol)

            if is_win_from_trade(pnl):
                if REOPEN_ON_WIN:
                    print_with_date(f"[WIN] Reopening {symbol} {pid}")
                    global CONTRACTS_MAP
                    contracts = CONTRACTS_MAP.get(symbol, 1)
                    new_pos_id, new_opening_order_id, new_closing_order_id, opening_price, trail_value = place_trailing_stop(symbol, info["side"], info["callback"], contracts)
                    if new_pos_id and new_closing_order_id:
                        positions[symbol][pid] = {
                            "position_id": new_pos_id,
                            "opening_order_id": new_opening_order_id,
                            "closing_order_id": new_closing_order_id,
                            "side": info["side"],
                            "callback": info["callback"],
                            "active": True,
                            "opening_price" : opening_price,
                            "trail_value" : trail_value
                        }
                        position_info = positions[symbol][pid]
                        update_position(pid, position_info, symbol)
                        all_closed = False
                        continue
                else:
                    print_with_date(f"[WIN] Not reopening {symbol} {pid} (REOPEN_ON_WIN=False)")
                    continue
            elif pnl is not None and is_breakeven_from_trade(symbol, info, closing_price):
                if REOPEN_ON_BREAKEVEN:
                    print_with_date(f"[BREAKEVEN] Reopening {symbol} {pid}")
                    contracts = CONTRACTS_MAP.get(symbol, 1)
                    new_pos_id, new_opening_order_id, new_closing_order_id, opening_price, trail_value = place_trailing_stop(symbol, info["side"], info["callback"], contracts)
                    if new_pos_id and new_closing_order_id:
                        positions[symbol][pid] = {
                            "position_id": new_pos_id,
                            "opening_order_id": new_opening_order_id,
                            "closing_order_id": new_closing_order_id,
                            "side": info["side"],
                            "callback": info["callback"],
                            "active": True,
                            "opening_price" : opening_price,
                            "trail_value" : trail_value
                        }
                        position_info = positions[symbol][pid]
                        update_position(pid, position_info, symbol)
                        all_closed = False
                        continue
                    else:
                        print_with_date(f"[BREAKEVEN] Not reopening {symbol} {pid} (REOPEN_ON_BREAKEVEN=False)")
                        continue
            else:
                print_with_date(f"[LOSS] Not reopening {symbol} {pid}")
            continue
        size = float(position_data.get("size", 0))
        debug(f"{pid} | size={size}")
        if size > 0:
            all_closed = False
            continue
        else:
            print_with_date(f"[CLOSED] {symbol} {pid} position is now closed.")
            info["active"] = False
            update_position(pid, info, symbol)
            trade = get_trade_by_closing_order_id(symbol, info["closing_order_id"])
            pnl = trade.get("total") if trade else None
            if pnl is not None and is_win_from_trade(pnl):
                if REOPEN_ON_WIN:
                    print_with_date(f"[WIN] Reopening {symbol} {pid}")
                    contracts = CONTRACTS_MAP.get(symbol, 1)
                    new_pos_id, new_opening_order_id, new_closing_order_id, opening_price, trail_value = place_trailing_stop(symbol, info["side"], info["callback"], contracts)
                    if new_pos_id and new_closing_order_id:
                        positions[symbol][pid] = {
                            "position_id": new_pos_id,
                            "opening_order_id": new_opening_order_id,
                            "closing_order_id": new_closing_order_id,
                            "side": info["side"],
                            "callback": info["callback"],
                            "active": True,
                            "opening_price" : opening_price,
                            "trail_value" : trail_value
                        }
                        position_info = positions[symbol][pid]
                        update_position(pid, position_info, symbol)
                        all_closed = False
                        continue
                else:
                    print_with_date(f"[WIN] Not reopening {symbol} {pid} (REOPEN_ON_WIN=False)")
                    continue
            elif pnl is not None and is_breakeven_from_trade(symbol, info, closing_price):
                if REOPEN_ON_BREAKEVEN:
                    print_with_date(f"[BREAKEVEN] Reopening {symbol} {pid}")
                    contracts = CONTRACTS_MAP.get(symbol, 1)
                    new_pos_id, new_opening_order_id, new_closing_order_id, opening_price, trail_value = place_trailing_stop(symbol, info["side"], info["callback"], contracts)
                    if new_pos_id and new_closing_order_id:
                        positions[symbol][pid] = {
                            "position_id": new_pos_id,
                            "opening_order_id": new_opening_order_id,
                            "closing_order_id": new_closing_order_id,
                            "side": info["side"],
                            "callback": info["callback"],
                            "active": True,
                            "opening_price" : opening_price,
                            "trail_value" : trail_value
                        }
                        position_info = positions[symbol][pid]
                        update_position(pid, position_info, symbol)
                        all_closed = False
                        continue
                else:
                    print_with_date(f"[BREAKEVEN] Not reopening {symbol} {pid} (REOPEN_ON_BREAKEVEN=False)")
                    continue
            else:
                print_with_date(f"[LOSS] Not reopening {symbol} {pid}")
    return all_closed
