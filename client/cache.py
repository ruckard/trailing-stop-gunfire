# client_cache.py
import socket, pickle
from utils import print_with_date, debug

HOST = "127.0.0.1"
PORT = 5005

def api_cache_fetch(func_name, *args):
    """Ask daemon to run a cached public API call."""
    print_with_date(f"DEBUG-api_cache_fetch: 1");
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print_with_date(f"DEBUG-api_cache_fetch: 2");
        s.connect((HOST, PORT))
        print_with_date(f"DEBUG-api_cache_fetch: 3");
        s.sendall(pickle.dumps((func_name, args)))
        print_with_date(f"DEBUG-api_cache_fetch: 4");
        data = pickle.loads(s.recv(4096))
        print_with_date(f"DEBUG-api_cache_fetch: 5");
    return data
