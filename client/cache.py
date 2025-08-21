# client_cache.py
import socket
from utils import print_with_date, debug
from client.msg_utils import send_msg, recv_msg

HOST = "127.0.0.1"
PORT = 5005

def api_cache_fetch(func_name, *args):
    """Ask daemon to run a cached public API call."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        send_msg(s, (func_name, args))
        data = recv_msg(s)
    return data
