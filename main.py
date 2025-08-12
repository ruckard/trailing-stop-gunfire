import time
import hmac
import hashlib
import requests
import json
import os
from datetime import datetime, timedelta, timezone
import sqlite3
from decimal import Decimal, getcontext, ROUND_FLOOR
import traceback
import pandas as pd
import numpy as np
import math
import importlib

from contextlib import contextmanager
from api_lock_client import api_lock_acquire_lock, api_lock_release_lock

import state
import db.positions as positionsdb
import db.knownsymbols as knownsymbolsdb

from exchange.btse import (
    get_market_summary,
    fetch_top_symbols_by_volume,
    fetch_contract_sizes,
    fetch_min_price_increments,
    get_current_price,
    place_range_order,
    close_position,
    update_leverage,
    update_position_mode,
    update_leverage_again,
    update_symbol_settings
)

from trading.indicators import get_atr

from trading.common import (
    bool_to_int,
    int_to_bool,
    debug_latest_trades,
    compute_contracts_from_prices,
)

from trading.trend import (
    classify_trend_or_range_real,
    classify_trend_or_range,
    calculate_easy_trend6_with_rsi,
    calculate_easy_trend5_with_rsi,
    calculate_easy_trend4_with_rsi,
    calculate_easy_trend3_with_rsi,
    calculate_easy_trend2_with_rsi,
    calculate_easy_trend_with_rsi,
    calculate_trendest_with_rsi,
    calculate_trend_with_rsi,
    calculate_ema_trend_score,
    calculate_atr,
    calculate_trailing_start_from_atr,
)

from trading.range import (
    place_range_positions,
)

from trading.positions import (
    show_positions,
    get_positions_status,
    get_position_status,
    get_trade_by_closing_order_id,
    get_trade_by_opening_order_id,
)

from trading.orders import (
    build_trailing_stops_map,
    update_trailing_stops_for_symbol,
    place_trailing_stop,
    place_all_positions,
    place_trend_positions,
    check_positions,
)

from trading.symbols import (
    update_symbol_registry,
    set_symbol_as_ready,
    setup_symbol_modes,
    filter_symbols_by_age_and_volume,
    filter_symbols_by_rank,
)

from trading.analysis import (
    is_win_from_trade,
    is_breakeven_from_trade,
)

from utils import print_with_date, debug, safe_override_import_or_default

class PriceFetchError(Exception):
    """Raised when the current price could not be fetched from the API."""
    pass

getcontext().prec = 16

# === DEBUG MODE ===
DEBUG_MODE = False  # Set to False to disable debug logs

def get_active_symbols_from_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT symbol, side FROM positions WHERE active = 1")
    rows = c.fetchall()
    conn.close()

    active_symbols = {}
    for symbol, side in rows:
        if symbol not in active_symbols:
            active_symbols[symbol] = set()
        active_symbols[symbol].add(side)
    return active_symbols  # e.g. {'BTC-PERP': {'LONG'}, 'ETH-PERP': {'SHORT'}}

def debug(msg):
    if DEBUG_MODE:
        print_with_date(f"[DEBUG] {msg}")

# === Custom Print Function ===
def print_with_date(msg, end='\n'):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {msg}", end=end)
    check_sleep_start=True

# === Import Configuration ===
from config import API_KEY, API_SECRET, BASE_URL, DB_PATH

# === Default Values ===
DEFAULT_SYMBOL_CONFIGS = {
    "BTC-PERP": {
        "TRAILING_STEP_MULTIPLIER": 0.375, # Step is going to be 37.5% of calculated trailing start
        "TRAILING_COUNT": 2,
    }
}

state.DEFAULT_TRAILING_STEP_MULTIPLIER = 0.375  # default global value
state.DEFAULT_TRAILING_COUNT = 1
DEFAULT_TOP_SYMBOLS_BY_VOLUME = 1000
DEFAULT_TRADE_MAX_CANDLES = 50
DEFAULT_CANDLE_INTERVAL_MINUTES = 5

DEFAULT_ADDITIONAL_SYMBOLS = []
DEFAULT_EXCLUDED_SYMBOLS = []

DEFAULT_API_DELAY_MS = 500  # Default 500ms between API requests
DEFAULT_ATR_MA_PERIOD = 48

DEFAULT_VOL_BOTTOM_PERCENTILE = 10
DEFAULT_VOL_TOP_PERCENTILE = 90

DEFAULT_REOPEN_ON_WIN = False
DEFAULT_REOPEN_ON_BREAKEVEN = False

DEFAULT_RANGE_ENTRY_OFFSET_PCT = 0.5
DEFAULT_RANGE_TAKE_PROFIT_PCT = 0.7
DEFAULT_RANGE_STOP_LOSS_PCT = 0.5

DEFAULT_MAXIMUM_LONG_TRADES_NUMBER = 6
DEFAULT_MAXIMUM_SHORT_TRADES_NUMBER = 6

state.KNOWN_SYMBOLS_DB_PATH = "known_symbols.db"
state.DB_PATH = DB_PATH
state.MIN_CONTRACT_AGE_DAYS = 15

# Default client name is the directory name where script is running
DEFAULT_CLIENT_NAME = os.path.basename(os.getcwd())

state.SYMBOL_CONFIGS = safe_override_import_or_default("override_config", "SYMBOL_CONFIGS", DEFAULT_SYMBOL_CONFIGS)
API_DELAY_MS = safe_override_import_or_default("override_config", "API_DELAY_MS", DEFAULT_API_DELAY_MS)
ATR_MA_PERIOD = safe_override_import_or_default("override_config", "ATR_MA_PERIOD", DEFAULT_ATR_MA_PERIOD)
state.TOP_SYMBOLS_BY_VOLUME = safe_override_import_or_default("override_config", "TOP_SYMBOLS_BY_VOLUME", DEFAULT_TOP_SYMBOLS_BY_VOLUME)
TRADE_MAX_CANDLES = safe_override_import_or_default("override_config", "TRADE_MAX_CANDLES", DEFAULT_TRADE_MAX_CANDLES)
CANDLE_INTERVAL_MINUTES = safe_override_import_or_default("override_config", "CANDLE_INTERVAL_MINUTES", DEFAULT_CANDLE_INTERVAL_MINUTES)
VOL_BOTTOM_PERCENTILE = safe_override_import_or_default("override_config", "VOL_BOTTOM_PERCENTILE", DEFAULT_VOL_BOTTOM_PERCENTILE)
VOL_TOP_PERCENTILE = safe_override_import_or_default("override_config", "VOL_TOP_PERCENTILE", DEFAULT_VOL_TOP_PERCENTILE)
REOPEN_ON_WIN = safe_override_import_or_default("override_config", "REOPEN_ON_WIN", DEFAULT_REOPEN_ON_WIN)
REOPEN_ON_BREAKEVEN = safe_override_import_or_default("override_config", "REOPEN_ON_BREAKEVEN", DEFAULT_REOPEN_ON_BREAKEVEN)
RANGE_ENTRY_OFFSET_PCT = safe_override_import_or_default("override_config", "RANGE_ENTRY_OFFSET_PCT", DEFAULT_RANGE_ENTRY_OFFSET_PCT)
RANGE_TAKE_PROFIT_PCT = safe_override_import_or_default("override_config", "RANGE_TAKE_PROFIT_PCT", DEFAULT_RANGE_TAKE_PROFIT_PCT)
RANGE_STOP_LOSS_PCT = safe_override_import_or_default("override_config", "RANGE_STOP_LOSS_PCT", DEFAULT_RANGE_STOP_LOSS_PCT)
MAXIMUM_LONG_TRADES_NUMBER = safe_override_import_or_default("override_config", "MAXIMUM_LONG_TRADES_NUMBER", DEFAULT_MAXIMUM_LONG_TRADES_NUMBER)
MAXIMUM_SHORT_TRADES_NUMBER = safe_override_import_or_default("override_config", "MAXIMUM_SHORT_TRADES_NUMBER", DEFAULT_MAXIMUM_SHORT_TRADES_NUMBER)
CLIENT_NAME = safe_override_import_or_default("override_config", "CLIENT_NAME", DEFAULT_CLIENT_NAME)
state.ADDITIONAL_SYMBOLS = safe_override_import_or_default("override_config", "ADDITIONAL_SYMBOLS", DEFAULT_ADDITIONAL_SYMBOLS)
state.EXCLUDED_SYMBOLS = safe_override_import_or_default("override_config", "EXCLUDED_SYMBOLS", DEFAULT_EXCLUDED_SYMBOLS)

CONTRACTS_MAP = {}
CONTRACT_SIZES = {}
state.MIN_PRICE_INCREMENTS = {}

LAST_AVAILABLE_BALANCE = None

TRENDRANGE_CACHE = {}  # symbol → (timestamp, result)
TRENDRANGE_CACHE_TIMEOUT = 5 * 60  # 5 minutes

MARKET_SUMMARY_CACHE = {
    "data": None,
    "timestamp": None,
}
MARKET_SUMMARY_CACHE_TIMEOUT = timedelta(hours=1)

def get_available_balance(currency="USDT"):
    """
    Query the CROSS wallet and return the available balance for the given currency.
    Prints balance change with color.
    """
    global LAST_AVAILABLE_BALANCE

    try:
        url_path = "/api/v2.2/user/wallet"
        url = BASE_URL + url_path

        nonce = str(int(time.time() * 1000))
        sig = generate_signature(API_SECRET, url_path, nonce, "")

        headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': sig
        }

        response = throttled_request("GET", url, headers=headers)
        response.raise_for_status()
        data = response.json()

        cross_wallet = next((w for w in data if w.get("wallet") == "CROSS@"), None)
        if not cross_wallet:
            print_with_date("[BALANCE] No CROSS@ wallet found.")
            return Decimal("0")

        available_balance = Decimal(str(cross_wallet.get("availableBalance", 0)))

        # Compute change
        balance_change = None
        if LAST_AVAILABLE_BALANCE is not None:
            balance_change = available_balance - LAST_AVAILABLE_BALANCE

        # Save for next call
        LAST_AVAILABLE_BALANCE = available_balance

        # Format change with color
        change_str = ""
        if balance_change is not None:
            if balance_change > 0:
                change_str = f"\033[92mChange: +{balance_change:.2f} {currency}\033[0m"
            elif balance_change < 0:
                change_str = f"\033[91mChange: {balance_change:.2f} {currency}\033[0m"
            else:
                change_str = f"Change: 0.00 {currency}"

        if change_str:
            print_with_date(f"[BALANCE] Available {currency}: {available_balance} | {change_str}")
        else:
            print_with_date(f"[BALANCE] Available {currency}: {available_balance}")

        return available_balance

    except Exception as e:
        print_with_date(f"[BALANCE ERROR] {e}")
        return Decimal("0")

def prune_market_summary_cache():
    if MARKET_SUMMARY_CACHE["timestamp"] is None:
        return
    if datetime.utcnow() - MARKET_SUMMARY_CACHE["timestamp"] >= MARKET_SUMMARY_CACHE_TIMEOUT:
        MARKET_SUMMARY_CACHE["data"] = None
        MARKET_SUMMARY_CACHE["timestamp"] = None

def get_final_symbol_list():
    top_symbols = fetch_top_symbols_by_volume(limit=TOP_SYMBOLS_BY_VOLUME)

    # Include additional symbols
    combined = top_symbols + ADDITIONAL_SYMBOLS

    # Remove excluded symbols and deduplicate while preserving order
    seen = set()
    final = []
    for s in combined:
        if s not in EXCLUDED_SYMBOLS and s not in seen:
            final.append(s)
            seen.add(s)
    return final

TRAILING_STOPS_MAP = build_trailing_stops_map()

# === Constants ===
CONTRACT_SIZE = 0.00001  # fixed for BTC-PERP on BTSE

# === Signature Generator ===
def generate_signature(api_secret, url_path, nonce, body_str):
    signature_payload = url_path + nonce + body_str
    signature = hmac.new(
        api_secret.encode('utf-8'),
        signature_payload.encode('utf-8'),
        hashlib.sha384
    ).hexdigest()
    return signature

# === Store Positions ===
positions = {}

def start_new_cycle(resume=False):
    global positions
    if resume:
        active_symbols = get_active_symbols_from_db()
        symbols = list(active_symbols.keys())
        long_symbols = [s for s, sides in active_symbols.items() if "LONG" in sides]
        short_symbols = [s for s, sides in active_symbols.items() if "SHORT" in sides]
        print_with_date(f"[RESUME] Resuming cycle with LONG symbols: {long_symbols}")
        print_with_date(f"[RESUME] Resuming cycle with SHORT symbols: {short_symbols}")
    else:
        # 1️⃣ Init DB
        knownsymbolsdb.init()

        # 2️⃣ Fetch market summary from BTSE
        market_summary = get_market_summary()
        if market_summary is None:
            return None, None, None

        # 3️⃣ Filter symbols by age and volume using the new helper
        filtered_symbols = filter_symbols_by_age_and_volume(market_summary)

        # 4️⃣ Update DB registry with filtered symbols
        update_symbol_registry(filtered_symbols)

        # 5️⃣ Setup modes for new symbols (mockup)
        setup_symbol_modes()

        # 6️⃣ Get only 'ready' symbols for trading
        base_symbols = knownsymbolsdb.get_ready_symbols()
        # Forget about old trades if we are starting a new cycle
        positions = {}
        for symbol in base_symbols:
            positionsdb.clear_positions(symbol)
        symbols, long_symbols, short_symbols = filter_symbols_by_rank(
            base_symbols,
            long_top_number=MAXIMUM_LONG_TRADES_NUMBER,
            short_top_number=MAXIMUM_SHORT_TRADES_NUMBER,
            rank_type='EASY6',
            vol_bottom_percentile = VOL_BOTTOM_PERCENTILE,
            vol_top_percentile = VOL_TOP_PERCENTILE
        )

        # Handle case where no symbols are selected
        if symbols is None or long_symbols is None or short_symbols is None:
            print_with_date("[CYCLE] No valid symbols found. Skipping cycle.")
            return None, None, None

    global CONTRACT_SIZES, CONTRACTS_MAP
    CONTRACT_SIZES = fetch_contract_sizes(symbols)
    state.MIN_PRICE_INCREMENTS = fetch_min_price_increments(symbols)
    CONTRACTS_MAP, MAX_EXPECTED_LOSS = compute_contracts_from_prices(symbols, CONTRACT_SIZES)

    print_with_date(f"[CONTRACT_SIZES] {CONTRACT_SIZES}")
    print_with_date(f"[CONTRACTS_MAP] {CONTRACTS_MAP}")

    if (not resume):
        print_with_date(f"[NEW] New cycle with LONG symbols: {long_symbols}")
        print_with_date(f"[NEW] New cycle with SHORT symbols: {short_symbols}")
        print_with_date(f"[NEW] MaxExpectedLoss: {MAX_EXPECTED_LOSS:.2f} USDT")
        for symbol in symbols:
            trend_type = classify_trend_or_range(symbol)
            print_with_date(f"[CLASSIFY] {symbol} : {trend_type.upper()}")

    for symbol in symbols:
        if symbol not in positions:
            positions[symbol] = {}
        positionsdb.load_positions(symbol)
        if not positions[symbol] and not resume:
            update_trailing_stops_for_symbol(symbol)
            if symbol in long_symbols:
                place_all_positions(symbol, sides=("LONG",))
            elif symbol in short_symbols:
                place_all_positions(symbol, sides=("SHORT",))
        else:
            show_positions(symbol)

    return symbols, long_symbols, short_symbols

# === Main Loop ===
def run_main_loop():
    positionsdb.init()

    active_symbols = get_active_symbols_from_db()
    resume_cycle = bool(active_symbols)

    # Ensure we have valid symbols before entering the main loop
    symbols = None
    while symbols is None:
        symbols, long_symbols, short_symbols = start_new_cycle(resume=resume_cycle)
        if symbols is None:
            print_with_date("[MAIN LOOP] No active symbols. Waiting 10 minutes before retry.")
            time.sleep(600)
            resume_cycle = False  # ensure it's not treated as resume on next try

    global check_sleep_start
    check_sleep_start = True

    while True:
        try:
            if check_sleep_start:
                print_with_date("", end='')
            print("C", end='', flush=True)
            check_sleep_start = False
            time.sleep(1)

            batch_all_closed = True

            if symbols is not None:
                for symbol in symbols:
                    if not check_positions(symbol):
                        batch_all_closed = False

            if batch_all_closed:
                print_with_date("[CYCLE] All symbols closed. Starting new cycle.")
                symbols, long_symbols, short_symbols = start_new_cycle()

            # If no symbols, just sleep and retry next iteration
            if symbols is None:
                print_with_date("[MAIN LOOP] No symbols in new cycle. Waiting 10 minutes.")
                time.sleep(600)
                continue

        except requests.exceptions.RequestException as e:
            print_with_date(f"[NETWORK ERROR] {e}. Retrying in 5 minutes.")
        except Exception as e:
            print_with_date(f"[UNHANDLED EXCEPTION] {e}. Traceback: {traceback.format_exc()} Retrying in 5 minutes.")
        print("S", end='', flush=True)
        check_sleep_start = False
        time.sleep(5 * 60)

if __name__ == "__main__":
    run_main_loop()
