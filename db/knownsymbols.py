import state

def init():
    conn = sqlite3.connect(state.KNOWN_SYMBOLS_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            symbol TEXT PRIMARY KEY,
            status TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_new_symbols():
    conn = sqlite3.connect(state.KNOWN_SYMBOLS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT symbol FROM symbols WHERE status='new'")
    result = [r[0] for r in c.fetchall()]
    conn.close()
    return result

def get_ready_symbols():
    conn = sqlite3.connect(state.KNOWN_SYMBOLS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT symbol FROM symbols WHERE status='ready'")
    result = [r[0] for r in c.fetchall()]
    conn.close()
    return result
