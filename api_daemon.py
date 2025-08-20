# api_daemon.py
import os
import socket
import threading
from daemon.lock import LockManager
from daemon.cache import CacheManager
import exchange.btse_cache as exchange_cache  # exchange-specific cache
from client.msg_utils import send_msg, recv_msg
from utils import safe_override_import_or_default
import state

HOST = "127.0.0.1"
PORT = 5005

DEFAULT_CLIENT_NAME = os.path.basename(os.getcwd())
state.CLIENT_NAME = "API_DAEMON-" + safe_override_import_or_default("override_config", "CLIENT_NAME", DEFAULT_CLIENT_NAME)

lock_manager = LockManager()
cache_manager = CacheManager(exchange_cache)

def handle_client(conn, addr):
    from utils import print_with_date, debug
    try:
        # receive a length-prefixed pickled object
        data = recv_msg(conn)
        if data is None:
            print_with_date(f"[APIDaemon] No data received from {addr}")
            return
        debug(f"[APIDaemon] Received data from {addr}: {data}")

        # --- LOCK commands ---
        if isinstance(data, tuple) and len(data) == 2 and data[0] in ("LOCK", "RELEASE"):
            command, client_id = data
            response = lock_manager.handle(command, client_id)
            # send back a length-prefixed pickled response
            send_msg(conn, response)

        # --- CACHE function calls ---
        elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], tuple):
            func_name, args = data
            result = cache_manager.get(func_name, args)
            send_msg(conn, result)

        # --- invalid format ---
        else:
            send_msg(conn, {"error": "Invalid request format"})
            print_with_date(f"[APIDaemon] Invalid request from {addr}: {data}")

    except Exception as e:
        send_msg(conn, {"error": str(e)})
        print_with_date(f"[APIDaemon] Exception handling {addr}: {e}")

    finally:
        conn.close()


def start_server():
    from utils import print_with_date
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print_with_date(f"[APIDaemon] Listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    start_server()
