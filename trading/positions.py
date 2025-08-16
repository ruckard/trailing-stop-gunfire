import state
from utils import print_with_date, debug
from config import API_KEY, API_SECRET, BASE_URL

def show_positions(symbol):
    print_with_date("[POSITIONS LOADED FROM DB]")
    if not state.positions[symbol]:
        print_with_date(f"No {symbol} positions stored.")
        return
    for pid, info in state.positions[symbol].items():
        print_with_date(
            f"[STORED] {symbol} | {info['side']} | Callback: {info['callback']}%"
        )

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
        state.positions = get_positions_status()
        debug(f"Checking positions for position_id: {position_id}")
        debug(f"All positions: {state.positions}")

        # Find the position with the matching position_id
        for position in state.positions:
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
