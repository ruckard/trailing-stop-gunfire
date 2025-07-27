import time
import hmac
import hashlib
import requests
import json
import os
from datetime import datetime
import sqlite3
from decimal import Decimal, getcontext, ROUND_FLOOR
import traceback
import pandas as pd

from contextlib import contextmanager
from api_lock_client import api_lock_acquire_lock, api_lock_release_lock

class PriceFetchError(Exception):
    """Raised when the current price could not be fetched from the API."""
    pass

getcontext().prec = 16

@contextmanager
def lock_guard(client_id):
    api_lock_acquire_lock(client_id)
    try:
        yield
    finally:
        api_lock_release_lock(client_id)

def throttled_request(method, url, **kwargs):
    with lock_guard("ClientA"):
        return requests.request(method, url, timeout=30, **kwargs)

def fetch_4h_ohlcv(symbol, limit=100):
    url = f"{BASE_URL}/api/v2.2/ohlcv"
    end_time = int(time.time() * 1000)  # current timestamp in ms
    params = {
        'symbol': symbol,
        'resolution': '5',  # 5m candles
        'end': end_time,
    }
    response = throttled_request("GET", url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if not data or len(data) < 20:
        print_with_date(f"[ERROR] Not enough candle data to calculate ATR for {symbol}")
        return None

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df

def calculate_ema_trend_score(symbol, lookback=50):
    df = fetch_4h_ohlcv(symbol)  # currently returns 5m candles
    if df is None or len(df) < lookback:
        return 0

    df['ema'] = df['close'].ewm(span=lookback, adjust=False).mean()
    # Slope = difference between last EMA and EMA N bars ago
    slope = df['ema'].iloc[-1] - df['ema'].iloc[-lookback]
    return slope

def calculate_atr(df, period=14, ma='SMA', ma_period=48):
    """
    Calculate the Average True Range (ATR) using specified moving average method.

    Args:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        period (int): The period for True Range calculation (typically 14).
        ma (str): Type of moving average - 'SMA', 'EMA', 'RMA', or 'Highest'.
        ma_period (int): The period for the moving average (default is same as `period`).

    Returns:
        float: The latest ATR value.
    """
    if ma_period is None:
        ma_period = period

    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    ma = ma.upper()
    if ma == 'SMA':
        df['ATR'] = df['TR'].rolling(window=ma_period).mean()
    elif ma == 'EMA':
        df['ATR'] = df['TR'].ewm(span=ma_period, adjust=False).mean()
    elif ma == 'RMA':
        df['ATR'] = df['TR'].ewm(alpha=1 / ma_period, adjust=False).mean()
    elif ma == 'HIGHEST':
        df['ATR'] = df['TR'].rolling(window=ma_period).max()
    else:
        raise ValueError("Invalid ma type. Use 'SMA', 'EMA', 'RMA', or 'Highest'.")

    return df['ATR'].iloc[-1]

def calculate_trailing_start_from_atr(symbol, multiplier=2.125, ma='HIGHEST', ma_period=48):
    df = fetch_4h_ohlcv(symbol)
    if df is None:
        return None
    atr = calculate_atr(df, ma_period=ATR_MA_PERIOD, ma=ma)
    last_close = df['close'].iloc[-1]
    atr_percent = (atr / last_close) * 100
    trailing_start = round(atr_percent * multiplier, 2)
    print_with_date(f"[ATR] {symbol} {ma}(ATR(ma_period)) = {atr:.2f}, % = {atr_percent:.2f}, TRAILING_START = {trailing_start}%")
    return trailing_start

# === DEBUG MODE ===
DEBUG_MODE = False  # Set to False to disable debug logs

def bool_to_int(value: bool) -> int:
    return 1 if value else 0

def int_to_bool(value: int) -> bool:
    return bool(value)

def debug_latest_trades(symbol, limit=10):
    try:
        url_path = '/api/v2.2/user/trade_history'
        url = BASE_URL + url_path
        nonce = str(int(time.time() * 1000))
        sig = generate_signature(API_SECRET, url_path, nonce, "")
        headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': sig,
            'Content-Type': 'application/json'
        }
        params = {
            'symbol': symbol,
            'includeOld': 'true',
            'count': limit  # Get latest N trades
        }
        response = throttled_request("GET", url, headers=headers, params=params)
        response.raise_for_status()
        trades = response.json()

        print_with_date(f"[DEBUG] Showing latest {limit} trades:")
        for i, trade in enumerate(trades, 1):
            print_with_date(
                f"{i}. Trade={trade}"
            )
        return trades

    except Exception as e:
        print_with_date(f"[ERROR] Failed to fetch trade history: {e}")
        return []

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            pid TEXT,
            position_id TEXT,
            opening_order_id TEXT,
            closing_order_id TEXT,
            side TEXT,
            callback REAL,
            active INTEGER,
            opening_price TEXT,
            trail_value REAL,
            symbol TEXT,
            PRIMARY KEY (pid, symbol)
        )
    ''')
    conn.commit()
    conn.close()

def show_positions(symbol):
    print_with_date("[POSITIONS LOADED FROM DB]")
    if not positions[symbol]:
        print_with_date(f"No {symbol} positions stored.")
        return
    for pid, info in positions[symbol].items():
        print_with_date(
            f"[STORED] {symbol} | {info['side']} | Callback: {info['callback']}%"
        )

def load_positions(symbol):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"SELECT pid, position_id, opening_order_id, closing_order_id, side, callback, active, opening_price, trail_value FROM positions WHERE symbol = \"{symbol}\"")
    rows = c.fetchall()
    conn.close()
    for pid, position_id, opening_order_id, closing_order_id, side, callback, active, opening_price, trail_value in rows:
        positions[symbol][pid] = {
            "position_id": position_id,
            "opening_order_id": opening_order_id,
            "closing_order_id": closing_order_id,
            "side": side,
            "callback": callback,
            "active": int_to_bool(active),
            "trail_value": trail_value,
        }

def update_position(pid, info, symbol):

    # Undefined opening_price workaround
    if "opening_price" not in info:
        opening_price = 0.0
        info["opening_price"] = opening_price
    else:
        opening_price = info["opening_price"]

    # Undefined trail_value workaround
    if "trail_value" not in info:
        trail_value = 0.0
    else:
        trail_value = info["trail_value"]

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO positions (pid, position_id, opening_order_id, closing_order_id, side, callback, active, opening_price, trail_value, symbol)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pid, symbol) DO UPDATE SET
            position_id=excluded.position_id,
            opening_order_id=excluded.opening_order_id,
            closing_order_id=excluded.closing_order_id,
            side=excluded.side,
            callback=excluded.callback,
            active=excluded.active,
            opening_price=excluded.opening_price,
            trail_value=excluded.trail_value,
            symbol=excluded.symbol
    ''', (pid, info['position_id'], info['opening_order_id'], info['closing_order_id'], info['side'], float(info['callback']), bool_to_int(info['active']), opening_price, trail_value, symbol))
    conn.commit()
    conn.close()

def clear_positions(symbol):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"DELETE FROM positions WHERE symbol = \"{symbol}\"")
    conn.commit()
    conn.close()

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

TRAILING_STEP_MULTIPLIER_DEFAULT = 0.375  # default global value
TRAILING_COUNT_DEFAULT = 2
TOP_SYMBOLS_BY_VOLUME_DEFAULT = 5

DEFAULT_ADDITIONAL_SYMBOLS = []
DEFAULT_EXCLUDED_SYMBOLS = []

DEFAULT_API_DELAY_MS = 500  # Default 500ms between API requests
DEFAULT_ATR_MA_PERIOD = 48

# === Check if override_config.py exists and load values if present ===
if os.path.exists('override_config.py'):
    from override_config import (
        SYMBOL_CONFIGS as OV_SYMBOL_CONFIGS,
        API_DELAY_MS as OV_API_DELAY_MS,
        ATR_MA_PERIOD as OV_ATR_MA_PERIOD
    )
else:
    OV_SYMBOL_CONFIGS = OV_API_DELAY_MS = OV_ATR_MA_PERIOD = None

try:
    from override_config import TOP_SYMBOLS_BY_VOLUME as OV_TOP_SYMBOLS_BY_VOLUME
except ImportError:
    OV_TOP_SYMBOLS_BY_VOLUME = None

try:
    from override_config import ADDITIONAL_SYMBOLS as OV_ADDITIONAL_SYMBOLS
except ImportError:
    OV_ADDITIONAL_SYMBOLS = []

try:
    from override_config import EXCLUDED_SYMBOLS as OV_EXCLUDED_SYMBOLS
except ImportError:
    OV_EXCLUDED_SYMBOLS = []

# === Final Config Values (Override if provided) ===
SYMBOL_CONFIGS = OV_SYMBOL_CONFIGS if OV_SYMBOL_CONFIGS is not None else DEFAULT_SYMBOL_CONFIGS
API_DELAY_MS = OV_API_DELAY_MS if OV_API_DELAY_MS is not None else DEFAULT_API_DELAY_MS
ATR_MA_PERIOD = OV_ATR_MA_PERIOD if OV_ATR_MA_PERIOD is not None else DEFAULT_ATR_MA_PERIOD
TOP_SYMBOLS_BY_VOLUME = OV_TOP_SYMBOLS_BY_VOLUME if OV_TOP_SYMBOLS_BY_VOLUME is not None else TOP_SYMBOLS_BY_VOLUME_DEFAULT
ADDITIONAL_SYMBOLS = OV_ADDITIONAL_SYMBOLS if OV_ADDITIONAL_SYMBOLS is not None else DEFAULT_ADDITIONAL_SYMBOLS
EXCLUDED_SYMBOLS = OV_EXCLUDED_SYMBOLS if OV_EXCLUDED_SYMBOLS is not None else DEFAULT_EXCLUDED_SYMBOLS

CONTRACTS_MAP = {}
CONTRACT_SIZES = {}
MIN_PRICE_INCREMENTS = {}

def fetch_top_symbols_by_volume(limit=5):
    try:
        url = f"{BASE_URL}/api/v2.2/market_summary"
        params = {"listFullAttributes": "true"}
        response = throttled_request("GET", url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            print_with_date("[ERROR] No data received from market_summary.")
            return []

        # Sort by 24h volume descending
        sorted_data = sorted(
            data,
            key=lambda m: m.get("volume", 0),
            reverse=True
        )

        # Extract top symbols
        top_symbols = [m["symbol"] for m in sorted_data if m.get("symbol") and m.get("volume") > 0]

        return top_symbols[:limit]

    except Exception as e:
        print_with_date(f"[ERROR] Failed to fetch top volume symbols: {e}")
        return []

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

def filter_symbols_by_rank(symbols, long_top_number=3, short_top_number=3, rank_type='EMA'):
    """
    Rank and filter symbols based on a trend score.

    Args:
        long_top_number (int): Number of top long symbols to select.
        short_top_number (int): Number of top short symbols to select.
        rank_type (str): Type of ranking metric to use ('EMA' supported for now).

    Returns:
        tuple: (symbols, long_symbols, short_symbols)
    """

    trend_scores = {}
    for symbol in symbols:
        if rank_type == 'EMA':
            score = calculate_ema_trend_score(symbol)
        else:
            raise ValueError(f"Unsupported rank_type: {rank_type}")
        trend_scores[symbol] = score

    # Sort symbols by score for long/short
    sorted_symbols = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)

    long_symbols = [s for s, score in sorted_symbols if score > 0][:long_top_number]
    short_symbols = [s for s, score in sorted(trend_scores.items(), key=lambda x: x[1]) if score < 0][:short_top_number]

    final_symbols = long_symbols + short_symbols

    return final_symbols, long_symbols, short_symbols

def fetch_contract_sizes(symbols):
    contract_sizes = {}
    for symbol in symbols:
        try:
            url = f"{BASE_URL}/api/v2.2/market_summary"
            params = {
                "symbol": symbol,
                "listFullAttributes": "true"
            }
            response = throttled_request("GET", url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                print_with_date(f"[WARN] No market data returned for {symbol}")
                continue

            market = data[0] if isinstance(data, list) else data
            contract_size = market.get("contractSize")
            if contract_size and Decimal(str(contract_size)) > 0:
                contract_sizes[symbol] = Decimal(str(contract_size))
            else:
                print_with_date(f"[WARN] No valid contractSize for {symbol}")

        except Exception as e:
            print_with_date(f"[ERROR] Failed to fetch contract size for {symbol}: {e}")

    if not contract_sizes:
        print_with_date("[ERROR] No contract sizes could be determined.")
    return contract_sizes

def fetch_min_price_increments(symbols):
    min_price_increments = {}
    for symbol in symbols:
        try:
            url = f"{BASE_URL}/api/v2.2/market_summary"
            params = {
                "symbol": symbol,
                "listFullAttributes": "true"
            }
            response = throttled_request("GET", url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                print_with_date(f"[WARN] No market data returned for {symbol}")
                continue

            market = data[0] if isinstance(data, list) else data
            min_price_increment = market.get("minPriceIncrement")
            if min_price_increment and Decimal(str(min_price_increment)) > 0:
                min_price_increments[symbol] = Decimal(str(min_price_increment))
            else:
                print_with_date(f"[WARN] No valid minPriceIncrement for {symbol}")

        except Exception as e:
            print_with_date(f"[ERROR] Failed to fetch contract size for {symbol}: {e}")

    if not min_price_increments:
        print_with_date("[ERROR] No contract sizes could be determined.")
    return min_price_increments

def compute_contracts_from_prices(symbols, contract_sizes):
    prices = {}
    contract_values = {}

    for symbol in symbols:
        price = get_current_price(symbol)
        if price is None or symbol not in contract_sizes:
            print_with_date(f"[ERROR] Skipping {symbol}, missing price or contract size.")
            continue
        price = Decimal(str(price))
        size = contract_sizes[symbol]
        value = price * size
        prices[symbol] = price
        contract_values[symbol] = value

    if not contract_values:
        print_with_date("[ERROR] No data to compute contracts.")
        return {}

    # Find the most expensive contract (e.g. BTC)
    max_value = max(contract_values.values())

    # Calculate how many contracts of each symbol stay <= max_value
    contracts_map = {}
    for symbol, value in contract_values.items():
        base_contracts = (max_value / value).to_integral_value(rounding=ROUND_FLOOR)
        base_contracts = max(base_contracts, 1)

        # Apply SYMBOL_MULTIPLIER from config (optional)
        config = SYMBOL_CONFIGS.get(symbol, {})
        multiplier = Decimal(str(config.get("SYMBOL_MULTIPLIER", 1.0)))
        adjusted_contracts = int((Decimal(base_contracts) * multiplier).to_integral_value(rounding=ROUND_FLOOR))

        contracts_map[symbol] = max(adjusted_contracts, 1) # at least 1

    return contracts_map

def build_trailing_stops_map():
    result = {}
    for symbol, cfg in SYMBOL_CONFIGS.items():
        trailing_start = calculate_trailing_start_from_atr(symbol)
        if trailing_start is None:
            continue  # or raise/log error

        trailing_start_decimal = Decimal(str(trailing_start))
        step_multiplier = Decimal(str(cfg.get("TRAILING_STEP_MULTIPLIER", TRAILING_STEP_MULTIPLIER_DEFAULT)))
        trailing_step = trailing_start_decimal * step_multiplier
        trailing_count = cfg.get("TRAILING_COUNT", TRAILING_COUNT_DEFAULT)

        result[symbol] = [
            float(round(trailing_start_decimal + i * trailing_step, 8))
            for i in range(trailing_count)
        ]
    return result

TRAILING_STOPS_MAP = build_trailing_stops_map()

def update_trailing_stops_for_symbol(symbol):
    cfg = SYMBOL_CONFIGS.get(symbol, {})

    trailing_start = calculate_trailing_start_from_atr(symbol)
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

# === Get Current Price ===
def get_current_price(symbol):
    try:
        url = f"{BASE_URL}/api/v2.2/price?symbol={symbol}"
        response = throttled_request('GET', url)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return float(data[0].get("lastPrice"))
    except Exception as e:
        print_with_date(f"[ERROR] Fetching price failed: {e}")
    return None

# === Place Trailing Stop Order on BTSE ===
def place_trailing_stop(symbol, position_side, callback_rate, contracts):
    global MIN_PRICE_INCREMENTS
    try:
        current_price = get_current_price(symbol)
        if not current_price:
            print_with_date("[ERROR] Failed to get current price.")
            return None, None, None, None, None
        callback_rate_float = float(callback_rate)

        min_price_increment = MIN_PRICE_INCREMENTS.get(symbol)
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

# === Get All Positions Status (BTSE) ===
def get_positions_status(symbol=None):
    try:
        # Using the correct endpoint to query position status
        endpoint_path = '/api/v2.2/user/positions'
        url = BASE_URL + endpoint_path

        # Query parameters: Optionally filter by symbol to get positions for a specific market
        if (symbol == None):
            params = {}
        else:
            params = {'symbol': symbol}

        # Signature generation (use the correct method for GET requests)
        nonce = str(int(time.time() * 1000))  # Generate nonce
        body_str = ""
        signature = generate_signature(API_SECRET, endpoint_path, nonce, body_str)

        # Headers for authentication
        headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': signature,
            'Content-Type': 'application/json'
        }

        if (symbol == None):
            debug(f"Sending request to get all positions status for every symbol")
        else:
            debug(f"Sending request to get all positions status for symbol: {symbol}")

        debug(f"Request parameters: {params}")
        debug(f"Request headers: {headers}")

        # Send GET request to the BTSE API to get positions
        response = throttled_request('GET', url, headers=headers, params=params)

        debug(f"Response status code: {response.status_code}")
        debug(f"Response body: {response.text}")
        
        # Check response status
        response.raise_for_status()  # Will raise an exception for 4xx or 5xx status codes

        # Parse the response JSON
        data = response.json()

        # Ensure the response is a list of positions (or empty if none)
        if not isinstance(data, list) or not data:
            print_with_date(f"[ERROR] Unexpected response format or empty data: {data}")
            return []

        # Return the list of all positions
        return data

    except requests.exceptions.ReadTimeout as e:
        print_with_date(f"[NETWORK TIMEOUT] Error while checking all position status: {e}")
        raise
    except requests.exceptions.RequestException as e:
        print_with_date(f"[ERROR] Network error while checking all position status: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print_with_date(f"[ERROR] Response status code: {e.response.status_code}")
            print_with_date(f"[ERROR] Response body: {e.response.text}")
        raise
    except Exception as e:
        print_with_date(f"[ERROR] Fetching positions failed: {e}")
        raise

# === Get Position Status by ID ===
def get_position_status(position_id):
    try:
        # Get all positions first
        positions = get_positions_status()
        debug(f"Checking positions for position_id: {position_id}")
        debug(f"All positions: {positions}")

        # Find the position with the matching position_id
        for position in positions:
            if position.get('positionId') == position_id:
                debug(f"Found position with position_id: {position_id}")
                return position

        print_with_date(f"[ERROR] Position with position_id: {position_id} not found.")
        return None

    except requests.exceptions.ReadTimeout as e:
        print_with_date(f"[NETWORK TIMEOUT] While checking position_id {position_id}: {e}")
        return "_network_error_"
    except Exception as e:
        print_with_date(f"[ERROR] Unexpected error while checking position status for position_id {position_id}: {e}")
        return "_unexpected_error_"

# === Get Trade by closing Order ID ===
def get_trade_by_closing_order_id(symbol, order_id):
    try:
        url_path = '/api/v2.2/user/trade_history'
        url = BASE_URL + url_path
        nonce = str(int(time.time() * 1000))
        sig = generate_signature(API_SECRET, url_path, nonce, "")
        headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': sig,
            'Content-Type': 'application/json'
        }
        params = {
            'symbol': symbol,
            'clOrderID': order_id,
            'includeOld': 'true'
        }
        response = throttled_request("GET", url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        debug(f"closing orderID: {order_id}")
        debug(f"Data from trade_history: {data}")
        return data[0] if data else None
    except Exception as e:
        print_with_date(f"[ERROR] Trade lookup failed for closing order_id {order_id}: {e}")
        return None

# === Get Trade by opening Order ID ===
def get_trade_by_opening_order_id(symbol, order_id):
    try:
        url_path = '/api/v2.2/user/trade_history'
        url = BASE_URL + url_path
        nonce = str(int(time.time() * 1000))
        sig = generate_signature(API_SECRET, url_path, nonce, "")
        headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': sig,
            'Content-Type': 'application/json'
        }
        params = {
            'symbol': symbol,
            'orderID': order_id,
            'includeOld': 'true'
        }
        response = throttled_request("GET", url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        debug(f"opening orderID: {order_id}")
        debug(f"Data from trade_history: {data}")
        return data[0] if data else None
    except Exception as e:
        print_with_date(f"[ERROR] Trade lookup failed for opening order_id {order_id}: {e}")
        return None

# === Win Check ===
def is_win_from_trade(realized_pnl):
    try:
        return float(realized_pnl) > 0
    except:
        return False

def is_breakeven_from_trade(symbol, info, closing_price):
    opening_price = info["opening_price"]
    trail_value = info["trail_value"]
    breakeven_percentage = 10

    current_price = get_current_price(symbol)
    if not current_price:
        print_with_date("[ERROR] Failed to get current price.")
        raise PriceFetchError("[ERROR] Failed to get current price for breakeven check.")

    try:
        trail_value = Decimal(str(trail_value))
        opening_price = Decimal(str(opening_price))
        closing_price = Decimal(str(closing_price))
        breakeven_percentage = Decimal(str(breakeven_percentage))
        one_percent = Decimal(str(0.01))
        
        return abs(closing_price-opening_price) < (breakeven_percentage * one_percent * trail_value)
    except Exception as e:
        print_with_date(f"[ERROR] Exception during breakeven check: {e}")
        return False

# === Store Positions ===
positions = {}

# === Place All Positions ===
def place_all_positions(symbol, sides=("LONG", "SHORT")):
    global CONTRACTS_MAP
    print_with_date(f"[STARTING NEW {symbol} CYCLE]")
    positions[symbol].clear()
    clear_positions(symbol)
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
                "trail_value" : trail_value
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
            elif pnl is not None and is_breakeven_from_trade(symbol, info, closing_price):
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
            elif pnl is not None and is_breakeven_from_trade(symbol, info, closing_price):
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
                print_with_date(f"[LOSS] Not reopening {symbol} {pid}")
    return all_closed

def start_new_cycle(resume=False):
    if resume:
        active_symbols = get_active_symbols_from_db()
        symbols = list(active_symbols.keys())
        long_symbols = [s for s, sides in active_symbols.items() if "LONG" in sides]
        short_symbols = [s for s, sides in active_symbols.items() if "SHORT" in sides]
        print_with_date(f"[RESUME] Resuming cycle with symbols: {symbols}")
    else:
        base_symbols = get_final_symbol_list()
        symbols, long_symbols, short_symbols = filter_symbols_by_rank(
            base_symbols,
            long_top_number=3,
            short_top_number=3,
            rank_type='EMA'
        )

    global CONTRACT_SIZES, MIN_PRICE_INCREMENTS, CONTRACTS_MAP
    CONTRACT_SIZES = fetch_contract_sizes(symbols)
    MIN_PRICE_INCREMENTS = fetch_min_price_increments(symbols)
    CONTRACTS_MAP = compute_contracts_from_prices(symbols, CONTRACT_SIZES)

    print_with_date(f"[CONTRACT_SIZES] {CONTRACT_SIZES}")
    print_with_date(f"[CONTRACTS_MAP] {CONTRACTS_MAP}")

    for symbol in symbols:
        if symbol not in positions:
            positions[symbol] = {}
        load_positions(symbol)
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
    init_db()

    active_symbols = get_active_symbols_from_db()
    resume_cycle = bool(active_symbols)
    symbols, long_symbols, short_symbols = start_new_cycle(resume=resume_cycle)

    global check_sleep_start
    check_sleep_start = True

    while True:
        try:
            if check_sleep_start:
                print_with_date("", end='')
            print("C", end='', flush=True)
            check_sleep_start = False
            time.sleep(1)

            batch_all_closed = all(check_positions(symbol) for symbol in symbols)

            if batch_all_closed:
                print_with_date("[CYCLE] All symbols closed. Starting new cycle.")
                symbols, long_symbols, short_symbols = start_new_cycle()

        except requests.exceptions.RequestException as e:
            print_with_date(f"[NETWORK ERROR] {e}. Retrying in 5 minutes.")
        except Exception as e:
            print_with_date(f"[UNHANDLED EXCEPTION] {e}. Traceback: {traceback.format_exc()} Retrying in 5 minutes.")
        print("S", end='', flush=True)
        check_sleep_start = False
        time.sleep(5 * 60)

if __name__ == "__main__":
    run_main_loop()
