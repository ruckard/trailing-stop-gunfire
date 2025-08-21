# client_cache.py
import socket, pickle
from utils import print_with_date, debug

HOST = "127.0.0.1"
PORT = 5005

def api_cache_fetch(func_name, *args):
    """Ask daemon to run a cached public API call."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(pickle.dumps((func_name, args)))
        data = pickle.loads(s.recv(4096))
    return data
