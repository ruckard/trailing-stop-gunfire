import time
import hmac
import hashlib
import requests
import os
from datetime import datetime
from config import API_KEY, API_SECRET, BASE_URL

DEBUG_MODE = False

# === Utilities ===

def print_with_date(msg, end='\n'):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {msg}", end=end)

def debug(msg):
    if DEBUG_MODE:
        print_with_date(f"[DEBUG] {msg}")

def throttled_request(method, url, **kwargs):
    time.sleep(0.5)  # simple 500ms delay
    return requests.request(method, url, timeout=30, **kwargs)

def generate_signature(api_secret, url_path, nonce, body_str):
    signature_payload = url_path + nonce + body_str
    return hmac.new(api_secret.encode('utf-8'), signature_payload.encode('utf-8'), hashlib.sha384).hexdigest()

# === Fetch top 20 symbols by volume ===
def fetch_top_symbols_by_volume(limit=20):
    try:
        url = f"{BASE_URL}/api/v2.2/market_summary"
        params = {"listFullAttributes": "true"}
        response = throttled_request("GET", url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            print_with_date("[ERROR] No data received.")
            return []
        sorted_data = sorted(data, key=lambda m: m.get("volume", 0), reverse=True)
        return [m["symbol"] for m in sorted_data if m.get("symbol") and m.get("volume") > 0][:limit]
    except Exception as e:
        print_with_date(f"[ERROR] Failed to fetch top symbols: {e}")
        return []

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
    debug(f"[SETUP-DEBUG] {symbol}: Response to Margin mode set to: isolated {response.text}")
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
    debug(f"[SETUP-DEBUG] {symbol}: Response to Position mode set to ISOLATED {response.text}")
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
    debug(f"[SETUP-DEBUG] {symbol}: (AGAIN) Response to Margin mode set to: isolated {response.text}")
    print_with_date(f"[SETUP] {symbol}: (AGAIN) Margin mode set to: isolated. Leverage set to 1x.")

def update_symbol_settings(symbol):
    try:
        update_leverage(symbol)
        update_position_mode(symbol)
        update_leverage_again(symbol)
    except Exception as e:
        print_with_date(f"[ERROR] {symbol}: setup failed: {e}")

# === Main Execution ===
def main():
    symbols = fetch_top_symbols_by_volume(limit=20)
    print_with_date(f"[INFO] Fetched top 20 symbols: {symbols}")
    for symbol in symbols:
        update_symbol_settings(symbol)

if __name__ == "__main__":
    import json  # Only imported here since needed
    main()

