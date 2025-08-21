# api_daemon.py
import socket
import threading
from daemon.lock import LockManager
from daemon.cache import CacheManager
import exchange.btse_cache as exchange_cache  # exchange-specific cache
from client.msg_utils import send_msg, recv_msg

HOST = "127.0.0.1"
PORT = 5005

lock_manager = LockManager()
cache_manager = CacheManager(exchange_cache)

def handle_client(conn, addr):
    from utils import print_with_date, debug
    try:
        data = recv_msg(conn)
        if data is None:
            print_with_date(f"[APIDaemon] No data received from {addr}")
            conn.close()
            return
        print_with_date(f"[APIDaemon] Received data from {addr}: {data}")

        # --- LOCK commands ---
        if isinstance(data, tuple) and len(data) == 2 and data[0] in ("LOCK", "RELEASE"):
            command, client_id = data
            response = lock_manager.handle(command, client_id)
            send_msg(conn, response)

        # --- CACHE function calls ---
        elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], tuple):
            func_name, args = data
            result = cache_manager.get(func_name, args)
            send_msg(conn, result)

        else:
            send_msg(conn, {"error": "Invalid request"})

    except Exception as e:
        send_msg(conn, {"error": str(e)})
    finally:
        conn.close()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print_with_date(f"[APIDaemon] Listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()
