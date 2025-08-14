import threading
from datetime import datetime
import sys
import importlib
import state

# ===============================
# Printing with timestamp
# ===============================

def print_with_date(msg):
    """
    Prints a message prefixed with a UTC timestamp in [YYYY-MM-DD HH:MM:SS] format.
    """
    now_str = datetime.utcnow().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now_str} {msg}")
    sys.stdout.flush()

# ===============================
# Lock Guard for throttling
# ===============================

_locks = {}

def lock_guard(name):
    """
    Context manager for a named lock.
    Used to synchronize access across threads for throttled API calls.
    """
    if name not in _locks:
        _locks[name] = threading.Lock()
    return _locks[name]

# ===============================
# Debug Logging
# ===============================

def debug(msg):
    if state.DEBUG_MODE:
        print_with_date(f"[DEBUG] {msg}")

# ===============================
# Others
# ===============================

def safe_override_import_or_default(
    module_name,
    symbol_name,
    default_value=None
):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)
    except (ImportError, AttributeError):
        return default_value
