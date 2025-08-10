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
    c.execute(f"SELECT pid, position_id, opening_order_id, closing_order_id, side, callback, active, opening_price, trail_value, opened_at FROM positions WHERE symbol = \"{symbol}\"")
    rows = c.fetchall()
    conn.close()
    for pid, position_id, opening_order_id, closing_order_id, side, callback, active, opening_price, trail_value, opened_at in rows:
        positions[symbol][pid] = {
            "position_id": position_id,
            "opening_order_id": opening_order_id,
            "closing_order_id": closing_order_id,
            "side": side,
            "callback": callback,
            "active": int_to_bool(active),
            "trail_value": trail_value,
            "opened_at": opened_at,
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

    # Undefined opened_at workaround
    if "opened_at" not in info:
        opened_at = time.time()
    else:
        opened_at = info["opened_at"]

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO positions (pid, position_id, opening_order_id, closing_order_id, side, callback, active, opening_price, trail_value, symbol, opened_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pid, symbol) DO UPDATE SET
            position_id=excluded.position_id,
            opening_order_id=excluded.opening_order_id,
            closing_order_id=excluded.closing_order_id,
            side=excluded.side,
            callback=excluded.callback,
            active=excluded.active,
            opening_price=excluded.opening_price,
            trail_value=excluded.trail_value,
            symbol=excluded.symbol,
            opened_at=excluded.opened_at
    ''', (pid, info['position_id'], info['opening_order_id'], info['closing_order_id'], info['side'], float(info['callback']), bool_to_int(info['active']), opening_price, trail_value, symbol, opened_at))
    conn.commit()
    conn.close()

def clear_positions(symbol):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"DELETE FROM positions WHERE symbol = \"{symbol}\"")
    conn.commit()
    conn.close()

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
