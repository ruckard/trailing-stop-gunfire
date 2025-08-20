import time
import pickle

CACHE_TTL = 300

class CacheManager:
    def __init__(self, exchange_module):
        self.exchange = exchange_module
        self.cache = {}

    def get(self, func_name, args):
        now = time.time()
        if func_name in self.cache:
            ts, data = self.cache[func_name]
            if now - ts < CACHE_TTL:
                return data

        # Call the real function from exchange module
        func = getattr(self.exchange, func_name, None)
        if not func:
            raise ValueError(f"Unknown function {func_name}")

        data = func(*args)
        self.cache[func_name] = (now, data)
        return data
