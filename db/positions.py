import sqlite3
import state

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
            opened_at REAL,
            PRIMARY KEY (pid, symbol)
        )
    ''')
    conn.commit()
    conn.close()

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

    conn = sqlite3.connect(state.DB_PATH)
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
