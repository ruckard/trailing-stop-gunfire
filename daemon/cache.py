import time
import pickle
import threading
import hashlib
import json

CACHE_TTL = 300
REQUEST_DELAY = 0.5  # 500 ms between requests to exchange

def _make_cache_key(func_name, args):
    """
    Create a short, deterministic hash key from the function name and arguments.
    Arguments are serialized to JSON (with sorted keys) to ensure stable order.
    """
    try:
        args_serialized = json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        # fallback if not serializable (e.g. NumPy arrays, custom classes)
        args_serialized = repr(args)

    raw_key = f"{func_name}:{args_serialized}"
    return hashlib.sha256(raw_key.encode()).hexdigest()[:16]  # short hash

class CacheManager:
    _last_request_time = 0.0
    _request_lock = threading.Lock()

    def __init__(self, exchange_module):
        self.exchange = exchange_module
        self.cache = {}

    def _throttled_call(self, func, *args):
        """Ensure at least REQUEST_DELAY seconds between exchange calls."""
        with CacheManager._request_lock:
            now = time.time()
            elapsed = now - CacheManager._last_request_time
            if elapsed < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY - elapsed)
            CacheManager._last_request_time = time.time()

        # After enforced delay, make the actual API call
        return func(*args)

    def get(self, func_name, args):
        now = time.time()

        # Generate key from function name and arguments
        key = _make_cache_key(func_name, args)

        # Check if cached
        if key in self.cache:
            ts, data = self.cache[key]
            if now - ts < CACHE_TTL:
                return data

        # Call the real function from exchange module, throttled
        func = getattr(self.exchange, func_name, None)
        if not func:
            raise ValueError(f"Unknown function {func_name}")

        data = self._throttled_call(func, *args)
        self.cache[key] = (now, data)
        return data
