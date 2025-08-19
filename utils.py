import threading
from datetime import datetime
import sys
import importlib
from contextlib import contextmanager

from api_lock_client import api_lock_acquire_lock, api_lock_release_lock
import state


def debug_positions_structure(debug_hint=None):
    if (debug_hint is not None):
        print_with_date(debug_hint)
    print_with_date(f"[DEBUG] state.positions - BEGIN")
    print_with_date(state.positions)
    print_with_date(f"[DEBUG] state.positions - END")
    if isinstance(state.positions, dict):
        print_with_date("[DEBUG] state.positions is a dict keyed by symbol (ok)")
    elif isinstance(state.positions, list) and all(isinstance(p, dict) and "symbol" in p for p in state.positions):
        print_with_date("[DEBUG] state.positions is a list of dicts (wrong)")
    else:
        print_with_date(f"[DEBUG] state.positions has unexpected structure: {type(state.positions)}")

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
