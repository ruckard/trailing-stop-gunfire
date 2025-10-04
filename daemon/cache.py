import time
import pickle
import threading

CACHE_TTL = 300
REQUEST_DELAY = 0.5  # 500 ms between requests to exchange


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
        if func_name in self.cache:
            ts, data = self.cache[func_name]
            if now - ts < CACHE_TTL:
                return data

        # Call the real function from exchange module, throttled
        func = getattr(self.exchange, func_name, None)
        if not func:
            raise ValueError(f"Unknown function {func_name}")

        data = self._throttled_call(func, *args)
        self.cache[func_name] = (now, data)
        return data
