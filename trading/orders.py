from . import trend
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN
from utils import print_with_date, debug

from exchange import btse as exchange
from config import API_KEY, API_SECRET, BASE_URL
import time, json

import state
import db.positions as positionsdb
from trading.positions import get_position_status, get_trade_by_closing_order_id, get_trade_by_opening_order_id
from trading.trend import place_trend_positions
from trading.range import place_range_positions
from trading.common import get_dynamic_trade_max_candles

from trading.analysis import (
    is_win_from_trade,
    is_breakeven_from_trade,
)

import time

def build_trailing_stops_map():
    result = {}
    for symbol, cfg in state.SYMBOL_CONFIGS.items():
        trailing_start = trend.calculate_trailing_start_from_atr(symbol, ma_period=state.ATR_MA_PERIOD)
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

    trailing_start = trend.calculate_trailing_start_from_atr(symbol, ma_period=state.ATR_MA_PERIOD)
    if trailing_start is None:
        print_with_date(f"[ERROR] Could not calculate trailing start for {symbol}")
        return

    trailing_start = Decimal(str(trailing_start))  # Ensure Decimal type
    step_multiplier = Decimal(str(cfg.get("TRAILING_STEP_MULTIPLIER", state.DEFAULT_TRAILING_STEP_MULTIPLIER)))

    trailing_step = trailing_start * step_multiplier
    trailing_count = cfg.get("TRAILING_COUNT", state.DEFAULT_TRAILING_COUNT)

    state.TRAILING_STOPS_MAP[symbol] = [
        round(trailing_start + i * trailing_step, 2)
        for i in range(trailing_count)
    ]

    debug(f"[UPDATED TRAILING STOPS] {symbol}: {state.TRAILING_STOPS_MAP[symbol]}")

# === Place Trailing Stop Order on BTSE ===
def place_trailing_stop(symbol, position_side, callback_rate, contracts):
    try:
        current_price = exchange.get_current_price(symbol)
        if not current_price:
            print_with_date("[ERROR] Failed to get current price.")
            return None, None, None, None, None, None
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

        custom_sl = state.TREND_STOP_LOSSES.get(symbol)
        if custom_sl is not None:
            # Do not make the stop loss bigger when rounding
            if (position_side == "SHORT"):
                ROUND_SIDE=ROUND_HALF_DOWN
            else:
                ROUND_SIDE=ROUND_HALF_UP
            custom_sl = Decimal(str(custom_sl)).quantize(Decimal(str(min_price_increment)), rounding=ROUND_SIDE)
            market_order["stopLossPrice"] = float(custom_sl)
            market_order["stopLossTrigger"] = "lastPrice"

            stop_loss_price = float(custom_sl)

        score = state.TREND_SCORES_TMP.get(symbol)

        market_body_str = json.dumps(market_order, separators=(',', ':'))
        market_sig = exchange.generate_signature(API_SECRET, url_path, nonce, market_body_str)
        market_headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': market_sig,
            'Content-Type': 'application/json'
        }

        debug(f"MARKET order payload: {market_body_str}")
        market_response = exchange.throttled_request('POST', full_url, headers=market_headers, data=market_body_str)
        debug(f"MARKET order response status: {market_response.status_code}")
        debug(f"MARKET order response body: {market_response.text}")
        market_response.raise_for_status()
        market_data = market_response.json()
        if not isinstance(market_data, list) or not market_data:
            print_with_date("[ERROR] Unexpected market order response.")
            return None, None, None, None, None, None

        position_id = market_data[0].get('positionId')
        if not position_id:
            print_with_date("[ERROR] Missing position ID.")
            return None, None, None, None, None, None

        opening_order_id = market_data[0].get('orderID')
        opening_price = market_data[0].get('price')

        # Wait for the market order to be executed
        # before binding the TP/SL order
        time.sleep(1)
        #debug(f"Placing Bind TP/SL order for {side} | TP: {take_profit_price} | SL: {stop_loss_price}")
        debug(f"Placing Bind TP/SL order for {side} | SL: {stop_loss_price}")

        nonce = str(int(time.time() * 1000))
        url_path = '/api/v2.2/order/bind/tpsl'
        full_url = BASE_URL + url_path

        tpsl_order = {
            "symbol": symbol,
            "side": side,
            #"takeProfitPrice": take_profit_price,
            #"takeProfitTrigger": "markPrice",
            "stopLossPrice": float(custom_sl),
            "stopLossTrigger": "lastPrice",
            "positionMode": "ISOLATED",
            "positionId": position_id
        }

        tpsl_body_str = json.dumps(tpsl_order, separators=(',', ':'))
        tpsl_sig = exchange.generate_signature(API_SECRET, url_path, nonce, tpsl_body_str)
        tpsl_headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': tpsl_sig,
            'Content-Type': 'application/json'
        }

        debug(f"Bind TP/SL payload: {tpsl_body_str}")
        tpsl_response = exchange.throttled_request('POST', full_url, headers=tpsl_headers, data=tpsl_body_str)
        debug(f"Bind TP/SL response status: {tpsl_response.status_code}")
        debug(f"Bind TP/SL response body: {tpsl_response.text}")
        tpsl_response.raise_for_status()
        tpsl_data = tpsl_response.json()
        closing_order_id = tpsl_data[0].get("orderID") if tpsl_data else None

        if not closing_order_id:
            print_with_date("[ERROR] Missing TP/SL bind order ID.")
            return None, None, None, None, None, None

        #print_with_date(f"[NEW] [BIND TP/SL] {symbol} | {position_side} | TP: {take_profit_price} | SL: {stop_loss_price}")
        print_with_date(f"[NEW] [BIND TP/SL] {symbol} | {position_side} | SL: {stop_loss_price}")
        return position_id, opening_order_id, closing_order_id, opening_price, trail_value, score
    except Exception as e:
        print_with_date(f"[ERROR] Failed to place TRAILING STOP order: {e}")

        # Extra debug info if variables exist
        if 'market_body_str' in locals():
            print_with_date(f"[ERROR-Debug] MARKET order payload: {market_body_str}")
        if 'market_response' in locals() and market_response is not None:
            if hasattr(market_response, 'status_code'):
                print_with_date(f"[ERROR-Debug] MARKET order response status: {market_response.status_code}")
            if hasattr(market_response, 'text'):
                print_with_date(f"[ERROR-Debug] MARKET order response body: {market_response.text}")

        # Extra debug info if variables exist
        if 'trail_body_str' in locals():
            print_with_date(f"[ERROR-Debug] TRAILING STOP order payload: {trail_body_str}")
        if 'trail_response' in locals() and trail_response is not None:
            if hasattr(trail_response, 'status_code'):
                print_with_date(f"[ERROR-Debug] TRAILING STOP response status: {trail_response.status_code}")
            if hasattr(trail_response, 'text'):
                print_with_date(f"[ERROR-Debug] TRAILING STOP response body: {trail_response.text}")
        return None, None, None, None, None, None

# === Place All Positions ===
def place_all_positions(symbol, sides=("LONG", "SHORT")):
    print_with_date(f"[STARTING NEW {symbol} CYCLE]")
    state.positions[symbol].clear()
    positionsdb.clear_positions(symbol)
    trend_type = trend.classify_trend_or_range(symbol)

    if trend_type == "trend":
        place_trend_positions(symbol, sides)
    elif trend_type == "range":
        print_with_date(f"[SKIP] Range detected for: {symbol}. RANGE DISABLED on purpose.")
        #place_range_positions(symbol, sides, entry_offset_pct=state.RANGE_ENTRY_OFFSET_PCT, take_profit_pct=state.RANGE_TAKE_PROFIT_PCT, stop_loss_pct=state.RANGE_STOP_LOSS_PCT)
    else:
        print_with_date(f"[SKIP] Could not classify trend/range for {symbol}")

# === Check and Manage Positions ===
def check_positions(symbol):
    all_closed = True

    # Return immediately if symbol is not in positions
    if (not (symbol in state.positions)):
        debug(f"[check_positions] No positions for symbol {symbol}")
        return all_closed  # All "closed" by default

    for pid, info in state.positions[symbol].items():
        debug(f"[check_positions] Checking... {symbol} {pid}")
        if not info["active"]:
            continue

        opened_at = info.get("opened_at")
        if opened_at:
            # Position max_candles
            position_max_candles = info.get("max_candles")
            if position_max_candles:
                if (position_max_candles > 36):
                    position_max_candles = 36
            else:
                position_max_candles = 36

            elapsed_minutes = (time.time() - opened_at) / 60
            if elapsed_minutes >= position_max_candles * state.CANDLE_INTERVAL_MINUTES:
                print_with_date(f"[TIMEOUT] Closing {symbol} {pid} after {elapsed_minutes:.1f} minutes.")
                # Code to close the position immediately:
                exchange.close_position(symbol, info)  # You'll need to implement or call your existing close logic
                info["active"] = False
                positionsdb.update_position(pid, info, symbol)
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
                positionsdb.update_position(pid, info, symbol)
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
            positionsdb.update_position(pid, info, symbol)

            if is_win_from_trade(pnl):
                if state.REOPEN_ON_WIN:
                    print_with_date(f"[WIN] Reopening {symbol} {pid}")
                    contracts = state.CONTRACTS_MAP.get(symbol, 1)
                    new_pos_id, new_opening_order_id, new_closing_order_id, opening_price, trail_value, score = place_trailing_stop(symbol, info["side"], info["callback"], contracts)
                    if new_pos_id and new_closing_order_id:
                        max_candles = get_dynamic_trade_max_candles(symbol, score)
                        state.positions[symbol][pid] = {
                            "position_id": new_pos_id,
                            "opening_order_id": new_opening_order_id,
                            "closing_order_id": new_closing_order_id,
                            "side": info["side"],
                            "callback": info["callback"],
                            "active": True,
                            "opening_price" : opening_price,
                            "trail_value" : trail_value,
                            "opened_at": time.time(),
                            "max_candles": max_candles
                        }
                        position_info = state.positions[symbol][pid]
                        positionsdb.update_position(pid, position_info, symbol)
                        all_closed = False
                        continue
                else:
                    print_with_date(f"[WIN] Not reopening {symbol} {pid} (REOPEN_ON_WIN=False)")
                    continue
            elif pnl is not None and is_breakeven_from_trade(symbol, info, closing_price):
                if state.REOPEN_ON_BREAKEVEN:
                    print_with_date(f"[BREAKEVEN] Reopening {symbol} {pid}")
                    contracts = state.CONTRACTS_MAP.get(symbol, 1)
                    new_pos_id, new_opening_order_id, new_closing_order_id, opening_price, trail_value, score = place_trailing_stop(symbol, info["side"], info["callback"], contracts)
                    if new_pos_id and new_closing_order_id:
                        max_candles = get_dynamic_trade_max_candles(symbol, score)
                        state.positions[symbol][pid] = {
                            "position_id": new_pos_id,
                            "opening_order_id": new_opening_order_id,
                            "closing_order_id": new_closing_order_id,
                            "side": info["side"],
                            "callback": info["callback"],
                            "active": True,
                            "opening_price" : opening_price,
                            "trail_value" : trail_value,
                            "opened_at": time.time(),
                            "max_candles": max_candles
                        }
                        position_info = state.positions[symbol][pid]
                        positionsdb.update_position(pid, position_info, symbol)
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
            positionsdb.update_position(pid, info, symbol)
            trade = get_trade_by_closing_order_id(symbol, info["closing_order_id"])
            pnl = trade.get("total") if trade else None
            if pnl is not None and is_win_from_trade(pnl):
                if state.REOPEN_ON_WIN:
                    print_with_date(f"[WIN] Reopening {symbol} {pid}")
                    contracts = state.CONTRACTS_MAP.get(symbol, 1)
                    new_pos_id, new_opening_order_id, new_closing_order_id, opening_price, trail_value, score = place_trailing_stop(symbol, info["side"], info["callback"], contracts)
                    if new_pos_id and new_closing_order_id:
                        max_candles = get_dynamic_trade_max_candles(symbol, score)
                        state.positions[symbol][pid] = {
                            "position_id": new_pos_id,
                            "opening_order_id": new_opening_order_id,
                            "closing_order_id": new_closing_order_id,
                            "side": info["side"],
                            "callback": info["callback"],
                            "active": True,
                            "opening_price" : opening_price,
                            "trail_value" : trail_value,
                            "opened_at": time.time(),
                            "max_candles": max_candles
                        }
                        position_info = state.positions[symbol][pid]
                        positionsdb.update_position(pid, position_info, symbol)
                        all_closed = False
                        continue
                else:
                    print_with_date(f"[WIN] Not reopening {symbol} {pid} (REOPEN_ON_WIN=False)")
                    continue
            elif pnl is not None and is_breakeven_from_trade(symbol, info, closing_price):
                if state.REOPEN_ON_BREAKEVEN:
                    print_with_date(f"[BREAKEVEN] Reopening {symbol} {pid}")
                    contracts = state.CONTRACTS_MAP.get(symbol, 1)
                    new_pos_id, new_opening_order_id, new_closing_order_id, opening_price, trail_value, score = place_trailing_stop(symbol, info["side"], info["callback"], contracts)
                    if new_pos_id and new_closing_order_id:
                        max_candles = get_dynamic_trade_max_candles(symbol, score)
                        state.positions[symbol][pid] = {
                            "position_id": new_pos_id,
                            "opening_order_id": new_opening_order_id,
                            "closing_order_id": new_closing_order_id,
                            "side": info["side"],
                            "callback": info["callback"],
                            "active": True,
                            "opening_price" : opening_price,
                            "trail_value" : trail_value,
                            "opened_at": time.time(),
                            "max_candles": max_candles
                        }
                        position_info = state.positions[symbol][pid]
                        positionsdb.update_position(pid, position_info, symbol)
                        all_closed = False
                        continue
                else:
                    print_with_date(f"[BREAKEVEN] Not reopening {symbol} {pid} (REOPEN_ON_BREAKEVEN=False)")
                    continue
            else:
                print_with_date(f"[LOSS] Not reopening {symbol} {pid}")
    return all_closed
