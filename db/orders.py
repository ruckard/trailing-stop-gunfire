import sqlite3
import state
import time

from trading.common import (
    bool_to_int,
    int_to_bool
)

def init():
    conn = sqlite3.connect(state.DB_PATH)
    c = conn.cursor()
    # Table for orders that are on the book but NOT yet positions
    c.execute('''
        CREATE TABLE IF NOT EXISTS pending_orders (
            order_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            callback REAL,
            contracts INTEGER,
            limit_price REAL,
            timestamp REAL,
            pid TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_pending_order(symbol, order_id, info):
    """Saves a limit order that has been placed but not yet filled."""
    conn = sqlite3.connect(state.DB_PATH)
    c = conn.cursor()
    
    # Extract values with fallbacks
    side = info.get('side')
    callback = float(info.get('callback', 0.0))
    contracts = int(info.get('contracts', 1))
    limit_price = float(info.get('limit_price', 0.0))
    timestamp = info.get('timestamp', time.time())
    pid = info.get('pid') # The internal tracking ID (e.g., long-0)

    c.execute('''
        INSERT INTO pending_orders (order_id, symbol, side, callback, contracts, limit_price, timestamp, pid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(order_id) DO UPDATE SET
            symbol=excluded.symbol,
            side=excluded.side,
            callback=excluded.callback,
            contracts=excluded.contracts,
            limit_price=excluded.limit_price,
            timestamp=excluded.timestamp,
            pid=excluded.pid
    ''', (order_id, symbol, side, callback, contracts, limit_price, timestamp, pid))
    
    conn.commit()
    conn.close()

def load_pending_orders():
    """Loads all pending orders from DB into the global state."""
    conn = sqlite3.connect(state.DB_PATH)
    c = conn.cursor()
    c.execute("SELECT order_id, symbol, side, callback, contracts, limit_price, timestamp, pid FROM pending_orders")
    rows = c.fetchall()
    conn.close()

    # Ensure the state dict is initialized
    if not hasattr(state, 'pending_orders'):
        state.pending_orders = {}

    for order_id, symbol, side, callback, contracts, limit_price, timestamp, pid in rows:
        if symbol not in state.pending_orders:
            state.pending_orders[symbol] = {}
        
        state.pending_orders[symbol][order_id] = {
            "side": side,
            "callback": callback,
            "contracts": contracts,
            "limit_price": limit_price,
            "timestamp": timestamp,
            "pid": pid
        }

def delete_pending_order(order_id):
    """Removes a pending order once it is filled or cancelled."""
    conn = sqlite3.connect(state.DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM pending_orders WHERE order_id = ?", (order_id,))
    conn.commit()
    conn.close()

def get_all_pending_by_symbol(symbol):
    """Returns a list of order IDs pending for a specific symbol."""
    conn = sqlite3.connect(state.DB_PATH)
    c = conn.cursor()
    c.execute("SELECT order_id FROM pending_orders WHERE symbol = ?", (symbol,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]
