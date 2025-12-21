import threading
from datetime import datetime
import sys
import importlib
from contextlib import contextmanager

from client.api_lock import api_lock_acquire_lock, api_lock_release_lock
import state
import json

try:
    import numpy as np
except ImportError:
    np = None

# ===============================
# Printing with timestamp
# ===============================

def print_with_date(msg, end='\n'):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {msg}", end=end)
    state.check_sleep_start=True

# ===============================
# Lock Guard for throttling
# ===============================

@contextmanager
def lock_guard(client_id):
    api_lock_acquire_lock(client_id)
    try:
        yield
    finally:
        api_lock_release_lock(client_id)

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

_debug_lock = threading.Lock()  # Ensure thread-safe writes

def _json_safe(value):
    """
    Recursively convert values to JSON-serializable primitives.
    Preserves semantic meaning (especially booleans).
    """
    # Fast path for native JSON types
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # NumPy scalars
    if np is not None:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)

    # Exceptions → structured info
    if isinstance(value, Exception):
        return {
            "exception_type": value.__class__.__name__,
            "message": str(value)
        }

    # Dict → recurse
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    # Iterable → recurse
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    # Fallback: string representation
    return str(value)

def ai_debug_log(event_type, data):
    """
    Logs structured debug data to ai_debug.log in JSON lines format.

    Parameters:
        event_type (str): Category of the event, e.g., "trend_score", "position", "symbol_filter".
        data (dict): Dictionary of key-value pairs to log.
    """
    try:
        safe_data = _json_safe(data)

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": str(event_type),
            "data": safe_data
        }

        log_line = json.dumps(log_entry, ensure_ascii=False)

        with _debug_lock:
            with open("ai_debug.log", "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

    except Exception as log_error:
        # Absolute last-resort fallback — logging must never break trading
        fallback = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "ai_debug_log_failure",
            "error": str(log_error),
            "original_event_type": str(event_type),
        }

        with _debug_lock:
            with open("ai_debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(fallback) + "\n")
