import time
import hmac
import hashlib
import requests
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

import state
import db.positions as positionsdb
import db.knownsymbols as knownsymbolsdb

from exchange import btse as exchange

from trading.indicators import get_atr

from trading.common import compute_contracts_from_prices

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

from trading.positions import show_positions

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

from client.cache import api_cache_fetch

class PriceFetchError(Exception):
    """Raised when the current price could not be fetched from the API."""
    pass

getcontext().prec = 16

# === DEBUG MODE ===
state.DEBUG_MODE = False  # Set to False to disable debug logs

# === Import Configuration ===
from config import DB_PATH

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
DEFAULT_TRADE_MAX_CANDLES = 12
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

# Default client name is the directory name where script is running
DEFAULT_CLIENT_NAME = os.path.basename(os.getcwd())

DEFAULT_FIB_LEVEL = 0.618

state.SYMBOL_CONFIGS = safe_override_import_or_default("override_config", "SYMBOL_CONFIGS", DEFAULT_SYMBOL_CONFIGS)
state.API_DELAY_MS = safe_override_import_or_default("override_config", "API_DELAY_MS", DEFAULT_API_DELAY_MS)
state.ATR_MA_PERIOD = safe_override_import_or_default("override_config", "ATR_MA_PERIOD", DEFAULT_ATR_MA_PERIOD)
state.TOP_SYMBOLS_BY_VOLUME = safe_override_import_or_default("override_config", "TOP_SYMBOLS_BY_VOLUME", DEFAULT_TOP_SYMBOLS_BY_VOLUME)
state.TRADE_MAX_CANDLES = safe_override_import_or_default("override_config", "TRADE_MAX_CANDLES", DEFAULT_TRADE_MAX_CANDLES)
state.CANDLE_INTERVAL_MINUTES = safe_override_import_or_default("override_config", "CANDLE_INTERVAL_MINUTES", DEFAULT_CANDLE_INTERVAL_MINUTES)
state.VOL_BOTTOM_PERCENTILE = safe_override_import_or_default("override_config", "VOL_BOTTOM_PERCENTILE", DEFAULT_VOL_BOTTOM_PERCENTILE)
state.VOL_TOP_PERCENTILE = safe_override_import_or_default("override_config", "VOL_TOP_PERCENTILE", DEFAULT_VOL_TOP_PERCENTILE)
state.REOPEN_ON_WIN = safe_override_import_or_default("override_config", "REOPEN_ON_WIN", DEFAULT_REOPEN_ON_WIN)
state.REOPEN_ON_BREAKEVEN = safe_override_import_or_default("override_config", "REOPEN_ON_BREAKEVEN", DEFAULT_REOPEN_ON_BREAKEVEN)
state.RANGE_ENTRY_OFFSET_PCT = safe_override_import_or_default("override_config", "RANGE_ENTRY_OFFSET_PCT", DEFAULT_RANGE_ENTRY_OFFSET_PCT)
state.RANGE_TAKE_PROFIT_PCT = safe_override_import_or_default("override_config", "RANGE_TAKE_PROFIT_PCT", DEFAULT_RANGE_TAKE_PROFIT_PCT)
state.RANGE_STOP_LOSS_PCT = safe_override_import_or_default("override_config", "RANGE_STOP_LOSS_PCT", DEFAULT_RANGE_STOP_LOSS_PCT)
state.CLIENT_NAME = safe_override_import_or_default("override_config", "CLIENT_NAME", DEFAULT_CLIENT_NAME)
state.ADDITIONAL_SYMBOLS = safe_override_import_or_default("override_config", "ADDITIONAL_SYMBOLS", DEFAULT_ADDITIONAL_SYMBOLS)
state.EXCLUDED_SYMBOLS = safe_override_import_or_default("override_config", "EXCLUDED_SYMBOLS", DEFAULT_EXCLUDED_SYMBOLS)
state.FIB_LEVEL = safe_override_import_or_default("override_config", "FIB_LEVEL", DEFAULT_FIB_LEVEL)

state.KNOWN_SYMBOLS_DB_PATH = "known_symbols.db"
state.DB_PATH = DB_PATH
state.MIN_CONTRACT_AGE_DAYS = 15

state.CONTRACTS_MAP = {}
state.CONTRACT_SIZES = {}
state.MIN_PRICE_INCREMENTS = {}
state.TREND_STOP_LOSSES = {}

state.LAST_AVAILABLE_BALANCE = None

state.TRENDRANGE_CACHE_TIMEOUT = 5 * 60  # 5 minutes

MAXIMUM_LONG_TRADES_NUMBER = safe_override_import_or_default("override_config", "MAXIMUM_LONG_TRADES_NUMBER", DEFAULT_MAXIMUM_LONG_TRADES_NUMBER)
MAXIMUM_SHORT_TRADES_NUMBER = safe_override_import_or_default("override_config", "MAXIMUM_SHORT_TRADES_NUMBER", DEFAULT_MAXIMUM_SHORT_TRADES_NUMBER)

MARKET_SUMMARY_CACHE = {
    "data": None,
    "timestamp": None,
}
MARKET_SUMMARY_CACHE_TIMEOUT = timedelta(hours=1)

def prune_market_summary_cache():
    if MARKET_SUMMARY_CACHE["timestamp"] is None:
        return
    if datetime.utcnow() - MARKET_SUMMARY_CACHE["timestamp"] >= MARKET_SUMMARY_CACHE_TIMEOUT:
        MARKET_SUMMARY_CACHE["data"] = None
        MARKET_SUMMARY_CACHE["timestamp"] = None

def get_final_symbol_list():
    top_symbols = exchange.fetch_top_symbols_by_volume(limit=TOP_SYMBOLS_BY_VOLUME)

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

state.TRAILING_STOPS_MAP = build_trailing_stops_map()

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
state.positions = {}

def start_new_cycle(resume=False):
    if resume:
        active_symbols = positionsdb.get_active_symbols()
        symbols = list(active_symbols.keys())
        long_symbols = [s for s, sides in active_symbols.items() if "LONG" in sides]
        short_symbols = [s for s, sides in active_symbols.items() if "SHORT" in sides]
        print_with_date(f"[RESUME] Resuming cycle with LONG symbols: {long_symbols}")
        print_with_date(f"[RESUME] Resuming cycle with SHORT symbols: {short_symbols}")
    else:
        # 1️⃣ Init DB
        knownsymbolsdb.init()

        # 2️⃣ Fetch market summary from BTSE
        print_with_date("DEBUG3.1 - get_market_summary - BEFORE");
        market_summary = exchange.get_market_summary()
        print_with_date("DEBUG3.2 - get_market_summary - AFTER");
        if market_summary is None:
            return None, None, None

        print_with_date("DEBUG4.1");
        # 3️⃣ Filter symbols by age and volume using the new helper
        filtered_symbols = filter_symbols_by_age_and_volume(market_summary)

        print_with_date("DEBUG5.1");
        # 4️⃣ Update DB registry with filtered symbols
        update_symbol_registry(filtered_symbols)

        print_with_date("DEBUG6.1");
        # 5️⃣ Setup modes for new symbols (mockup)
        setup_symbol_modes()

        print_with_date("DEBUG7.1");
        # 6️⃣ Get only 'ready' symbols for trading
        base_symbols = knownsymbolsdb.get_ready_symbols()
        # Forget about old trades if we are starting a new cycle
        state.positions = {}
        print_with_date("DEBUG8.1");
        for symbol in base_symbols:
            positionsdb.clear_positions(symbol)
        symbols, long_symbols, short_symbols = filter_symbols_by_rank(
            base_symbols,
            long_top_number=MAXIMUM_LONG_TRADES_NUMBER,
            short_top_number=MAXIMUM_SHORT_TRADES_NUMBER,
            rank_type='EASY8',
            vol_bottom_percentile = state.VOL_BOTTOM_PERCENTILE,
            vol_top_percentile = state.VOL_TOP_PERCENTILE
        )

        print_with_date("DEBUG9.1");
        # Handle case where no symbols are selected
        if symbols is None or long_symbols is None or short_symbols is None:
            print_with_date("[CYCLE] No valid symbols found. Skipping cycle.")
            return None, None, None

    print_with_date("DEBUG10.1");
    state.CONTRACT_SIZES = api_cache_fetch.fetch_contract_sizes(symbols)
    print_with_date("DEBUG11.1");
    state.MIN_PRICE_INCREMENTS = api_cache_fetch.fetch_min_price_increments(symbols)
    state.CONTRACTS_MAP, MAX_EXPECTED_LOSS = compute_contracts_from_prices(symbols, state.CONTRACT_SIZES)

    print_with_date(f"[CONTRACT_SIZES] {state.CONTRACT_SIZES}")
    print_with_date(f"[CONTRACTS_MAP] {state.CONTRACTS_MAP}")

    if (not resume):
        print_with_date(f"[NEW] New cycle with LONG symbols: {long_symbols}")
        print_with_date(f"[NEW] New cycle with SHORT symbols: {short_symbols}")
        print_with_date(f"[NEW] MaxExpectedLoss: {MAX_EXPECTED_LOSS:.2f} USDT")
        for symbol in symbols:
            trend_type = classify_trend_or_range(symbol)
            print_with_date(f"[CLASSIFY] {symbol} : {trend_type.upper()}")

    for symbol in symbols:
        if symbol not in state.positions:
            state.positions[symbol] = {}
        positionsdb.load_positions(symbol)
        if not state.positions[symbol] and not resume:
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

    active_symbols = positionsdb.get_active_symbols()
    resume_cycle = bool(active_symbols)

    # Ensure we have valid symbols before entering the main loop
    symbols = None
    while symbols is None:
        print_with_date("DEBUG1.1 - while symbols is None");
        symbols, long_symbols, short_symbols = start_new_cycle(resume=resume_cycle)
        print_with_date("DEBUG1.2 - After start_new_cycle");
        if symbols is None:
            print_with_date("[MAIN LOOP] No active symbols. Waiting 10 minutes before retry.")
            time.sleep(600)
            resume_cycle = False  # ensure it's not treated as resume on next try

    state.check_sleep_start = True

    while True:
        try:
            if state.check_sleep_start:
                print_with_date("", end='')
            print("C", end='', flush=True)
            state.check_sleep_start = False
            time.sleep(1)

            batch_all_closed = True

            if symbols is not None:
                for symbol in symbols:
                    if not check_positions(symbol):
                        batch_all_closed = False

            if batch_all_closed:
                print_with_date("[CYCLE] All symbols closed. Starting new cycle.")
                print_with_date("DEBUG2.1 - New cycle main loop - BEFORE");
                symbols, long_symbols, short_symbols = start_new_cycle()
                print_with_date("DEBUG2.2 - New cycle main loop - AFTER");

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
        state.check_sleep_start = False
        time.sleep(5 * 60)

if __name__ == "__main__":
    run_main_loop()
