# exchange/btse_cache.py
from exchange import btse   # import your real BTSE API functions

def fetch_5m_ohlcv_real(symbol, limit=100):
    """Wrapper for BTSE 5m OHLCV fetch."""
    return btse.fetch_5m_ohlcv_real(symbol, limit=limit)

def fetch_contract_sizes(symbols):
    """Wrapper for BTSE contract sizes fetch."""
    return btse.fetch_contract_sizes(symbols)

def fetch_min_price_increments(symbols):
    """Wrapper for BTSE min price increments fetch."""
    return btse.fetch_min_price_increments(symbols)
