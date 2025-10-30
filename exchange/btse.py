import os
import time
import json
import math
import requests
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import state

# Import your config and utilities
from config import API_KEY, API_SECRET, BASE_URL
from utils import print_with_date, lock_guard, debug

from client.cache import api_cache_fetch

OHLCV_CACHE = {}
OHLCV_CACHE_TIMEOUT = timedelta(minutes=5)

# ===============================
# Others
# ===============================

def retry_until_valid(fetch_func, *args, max_retries=None, wait_seconds=10, **kwargs):
    attempt = 0
    while True:
        result = fetch_func(*args, **kwargs)
        if result is not None:
            return result
        attempt += 1
        print_with_date(f"[RETRY] {fetch_func.__name__} failed. Attempt {attempt}. Retrying in {wait_seconds}s...")
        time.sleep(wait_seconds)
        if max_retries is not None and attempt >= max_retries:
            print_with_date(f"[RETRY] Max retries reached for {fetch_func.__name__}. Returning None.")
            return None

def throttled_request(method, url, **kwargs):
    with lock_guard(state.CLIENT_NAME):
        return requests.request(method, url, timeout=30, **kwargs)

# ===============================
# Authentication & Request Helpers
# ===============================

def generate_signature(api_secret, path, nonce, data_str):
    import hmac
    import hashlib
    message = path + nonce + data_str
    signature = hmac.new(
        bytes(api_secret, "utf-8"),
        msg=bytes(message, "utf-8"),
        digestmod=hashlib.sha384
    ).hexdigest()
    return signature

# ===============================
# Market Summary (Cached)
# ===============================

MARKET_SUMMARY_CACHE = {
    "data": None,
    "timestamp": None,
}
MARKET_SUMMARY_CACHE_TIMEOUT = timedelta(minutes=5)

def prune_market_summary_cache():
    if MARKET_SUMMARY_CACHE["timestamp"] is None:
        return
    if datetime.utcnow() - MARKET_SUMMARY_CACHE["timestamp"] >= MARKET_SUMMARY_CACHE_TIMEOUT:
        MARKET_SUMMARY_CACHE["data"] = None
        MARKET_SUMMARY_CACHE["timestamp"] = None

def get_market_summary():
    prune_market_summary_cache()
    if MARKET_SUMMARY_CACHE["data"] is not None:
        return MARKET_SUMMARY_CACHE["data"]

    try:
        url = f"{BASE_URL}/api/v2.2/market_summary"
        params = {"listFullAttributes": "true"}
        response = throttled_request("GET", url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            print_with_date("[ERROR] No data received from market_summary.")
            return []
        # Cache fresh data with current timestamp
        MARKET_SUMMARY_CACHE["data"] = data
        MARKET_SUMMARY_CACHE["timestamp"] = datetime.utcnow()
        return data
    except Exception as e:
        print_with_date(f"[ERROR] Failed to fetch market summary: {e}")
        return []

def fetch_top_symbols_by_volume(limit=5):
    try:
        data = get_market_summary()
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

# ===============================
# OHLCV
# ===============================

def prune_ohlcv_cache():
    now = datetime.utcnow()
    expired = [sym for sym, (_, ts) in OHLCV_CACHE.items() if now - ts >= OHLCV_CACHE_TIMEOUT]
    for sym in expired:
        del OHLCV_CACHE[sym]

def fetch_5m_ohlcv_real(symbol, limit=100):
    url = f"{BASE_URL}/api/v2.2/ohlcv"
    end_time = int(time.time() * 1000)  # current timestamp in ms
    params = {
        'symbol': symbol,
        'resolution': '15',  # 15m candles
        'end': end_time,
    }
    response = throttled_request("GET", url, params=params)
    response.raise_for_status()
    data = response.json()

    if not data or len(data) < 20:
        print_with_date(f"[ERROR] Not enough candle data to calculate ATR for {symbol}")
        return None

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values('timestamp')
    return df

def fetch_5m_ohlcv(symbol, limit=100):
    """
    Cached wrapper around fetch_5m_ohlcv_real.
    Prunes expired entries and uses cache if available.
    """
    # Remove expired cache entries first
    prune_ohlcv_cache()

    # If symbol is cached after pruning, it's valid
    if symbol in OHLCV_CACHE:
        return OHLCV_CACHE[symbol][0]

    # Otherwise, fetch fresh data and cache it
    df = api_cache_fetch("fetch_5m_ohlcv_real", symbol, limit)
    OHLCV_CACHE[symbol] = (df, datetime.utcnow())
    return df

# ===============================
# Contract & Price Info
# ===============================

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

# ===============================
# Order Placement
# ===============================

def place_range_order(symbol, position_side, contracts, entry_price, take_profit, stop_loss, cl_order_id):
    """
    Place a limit order with both take profit and stop loss triggers.
    Uses BTSE's API v2.2 /order endpoint.
    """
    try:
        url = "https://api.btse.com/futures/api/v2.2/order"

        limit_side = "SELL" if position_side == "SHORT" else "BUY"  # Entry side

        url_path = '/api/v2.2/order'
        full_url = BASE_URL + url_path

        # === Limit Order ===
        print_with_date(f"[DEBUG] Placing LIMIT order: {limit_side} {contracts} contracts")
        nonce = str(int(time.time() * 1000))
        limit_order = {
            "postOnly": False,
            "price": float(entry_price),
            "reduceOnly": False,
            "side": limit_side,
            "size": contracts,
            "symbol": symbol,
            "takeProfitPrice": float(take_profit),
            "takeProfitTrigger": "markPrice",
            "stopLossPrice": float(stop_loss),
            "stopLossTrigger": "lastPrice",
            "time_in_force": "GTC",
            "type": "LIMIT",
            "txType": "LIMIT",
            "positionMode": "ISOLATED",
            "clOrderID": cl_order_id
        }
        limit_body_str = json.dumps(limit_order, separators=(',', ':'))
        limit_sig = generate_signature(API_SECRET, url_path, nonce, limit_body_str)
        limit_headers = {
            'request-api': API_KEY,
            'request-nonce': nonce,
            'request-sign': limit_sig,
            'Content-Type': 'application/json'
        }

        print_with_date(f"LIMIT order payload: {limit_body_str}")
        limit_response = throttled_request('POST', full_url, headers=limit_headers, data=limit_body_str)
        print_with_date(f"LIMIT order response status: {limit_response.status_code}")
        print_with_date(f"LIMIT order response body: {limit_response.text}")
        limit_response.raise_for_status()
        limit_data = limit_response.json()
        if not isinstance(limit_data, list) or not limit_data:
            print_with_date("[ERROR] Unexpected limit order response.")
            return None
        position_id = limit_data[0].get('positionId')
        if not position_id:
            print_with_date("[ERROR] Missing position ID.")
            return None
        return position_id
    except Exception as e:
        print_with_date(f"[ERROR] Failed to place LIMIT order: {e}")
        return None

def close_position(symbol, info):
    """
    Close an open position for the given symbol using BTSE's close_position endpoint.
    Automatically closes the entire position at market price.
    """
    try:
        position_id = info.get("position_id")

        url_path = "/api/v2.2/order/close_position"
        full_url = BASE_URL + url_path

        # Basic request body
        order = {
            "symbol": symbol,
            "type": "MARKET"
        }

        # For isolated/hedge mode, include positionId
        if position_id:
            order["positionId"] = position_id

        body_str = json.dumps(order, separators=(',', ':'))
        nonce = str(int(time.time() * 1000))
        sig = generate_signature(API_SECRET, url_path, nonce, body_str)
        headers = {
            "request-api": API_KEY,
            "request-nonce": nonce,
            "request-sign": sig,
            "Content-Type": "application/json"
        }

        debug(f"[CLOSE] Sending close_position request for {symbol}, positionId={position_id}")
        response = throttled_request("POST", full_url, headers=headers, data=body_str)
        response.raise_for_status()

        print_with_date(f"[CLOSE] Successfully sent close_position for {symbol}")
        return True

    except Exception as e:
        print_with_date(f"[ERROR] Failed to close {symbol}: {e}")
        return False

# ===============================
# Position & Mode Setup
# ===============================

def update_leverage(symbol):
    url_path = '/api/v2.2/leverage'
    full_url = BASE_URL + url_path

    params = {"symbol": symbol, "marginMode": "ISOLATED", "leverage": "1"}
    nonce = str(int(time.time() * 1000))
    body_str = json.dumps(params, separators=(',', ':'))
    sig = generate_signature(API_SECRET, url_path, nonce, body_str)
    headers = {
        'request-api': API_KEY,
        'request-nonce': nonce,
        'request-sign': sig,
        'Content-Type': 'application/json'
    }

    response = throttled_request("POST", full_url, headers=headers, data=body_str)
    print_with_date(f"[SETUP] {symbol}: Margin mode set to: isolated. Leverage set to 1x.")
    time.sleep(1)

def update_position_mode(symbol):
    url_path = '/api/v2.2/position_mode'
    full_url = BASE_URL + url_path

    params = {"symbol": symbol, "positionMode": "ISOLATED"}
    nonce = str(int(time.time() * 1000))
    body_str = json.dumps(params, separators=(',', ':'))
    sig = generate_signature(API_SECRET, url_path, nonce, body_str)
    headers = {
        'request-api': API_KEY,
        'request-nonce': nonce,
        'request-sign': sig,
        'Content-Type': 'application/json'
    }

    response = throttled_request("POST", full_url, headers=headers, data=body_str)
    print_with_date(f"[SETUP] {symbol}: Position mode set to ISOLATED")
    time.sleep(1)

def update_leverage_again(symbol):
    url_path = '/api/v2.2/leverage'
    full_url = BASE_URL + url_path

    params = {
        "symbol": symbol,
        "positionMode": "ISOLATED",
        "marginMode": "ISOLATED",
        "leverage": "1"
    }
    nonce = str(int(time.time() * 1000))
    body_str = json.dumps(params, separators=(',', ':'))
    sig = generate_signature(API_SECRET, url_path, nonce, body_str)
    headers = {
        'request-api': API_KEY,
        'request-nonce': nonce,
        'request-sign': sig,
        'Content-Type': 'application/json'
    }

    response = throttled_request("POST", full_url, headers=headers, data=body_str)
    print_with_date(f"[SETUP] {symbol}: (AGAIN) Margin mode set to: isolated. Leverage set to 1x.")

def update_symbol_settings(symbol):
    try:
        update_leverage(symbol)
        update_position_mode(symbol)
        update_leverage_again(symbol)
    except Exception as e:
        print_with_date(f"[ERROR] {symbol}: setup failed: {e}")

# ===============================
# More functions
# ===============================

def get_available_balance(currency="USDT"):
    """
    Query the CROSS wallet and return the available balance for the given currency.
    Prints balance change with color.
    """

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
        if state.LAST_AVAILABLE_BALANCE is not None:
            balance_change = available_balance - state.LAST_AVAILABLE_BALANCE

        # Save for next call
        state.LAST_AVAILABLE_BALANCE = available_balance

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
