import socket
import threading
import pickle

from daemon.lock import LockManager
from daemon.cache import CacheManager
import exchange.btse_cache as exchange_cache  # exchange-specific cache

HOST = "127.0.0.1"
PORT = 5005

lock_manager = LockManager()
cache_manager = CacheManager(exchange_cache)

def handle_client(conn, addr):
    try:
        request = conn.recv(4096)
        if not request:
            conn.close()
            return

        data = pickle.loads(request)

        # Distinguish lock vs cache
        if isinstance(data, str):
            # e.g. "LOCK client1" or "RELEASE client1"
            command, client_id = data.split()
            response = lock_manager.handle(command, client_id)
            conn.sendall(response.encode())

        elif isinstance(data, tuple):
            # e.g. ("fetch_4h_ohlcv_real", ("BTC-PERP",))
            func_name, args = data
            result = cache_manager.get(func_name, args)
            conn.sendall(pickle.dumps(result))

        else:
            conn.sendall(pickle.dumps({"error": "Invalid request"}))

    except Exception as e:
        conn.sendall(pickle.dumps({"error": str(e)}))
    finally:
        conn.close()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[APIDaemon] Listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()
