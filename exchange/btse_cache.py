# exchange/btse_cache.py
from exchange import btse   # import your real BTSE API functions

def fetch_4h_ohlcv_real(symbol: str):
    """Wrapper for BTSE 4h OHLCV fetch."""
    return btse.fetch_4h_ohlcv_real(symbol)

def fetch_contract_sizes():
    """Wrapper for BTSE contract sizes fetch."""
    return btse.fetch_contract_sizes()

def fetch_min_price_increments():
    """Wrapper for BTSE min price increments fetch."""
    return btse.fetch_min_price_increments()
