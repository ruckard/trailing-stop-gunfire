# wait_daemon.py
import socket
import threading
import time
from collections import deque

HOST = '127.0.0.1'
PORT = 5002
LOCK_TTL = 10  # seconds

lock = threading.Lock()
current_holder = None
lock_acquired_time = None
queue = deque()

def handle_client(conn, addr):
    global current_holder, lock_acquired_time
    try:
        data = conn.recv(1024).decode().strip()
        if not data:
            conn.close()
            return
        command, client_id = data.split()

        with lock:
            # Timeout old holder
            if current_holder and (time.time() - lock_acquired_time > LOCK_TTL):
                print(f"[Daemon] Lock by {current_holder} expired.")
                current_holder = None
                lock_acquired_time = None

            if command == "LOCK":
                if current_holder is None:
                    current_holder = client_id
                    lock_acquired_time = time.time()
                    print(f"[Daemon] Lock granted to {client_id}")
                    conn.sendall(b"GRANTED")
                elif client_id == current_holder:
                    # Already holds it
                    conn.sendall(b"GRANTED")
                else:
                    if client_id not in queue:
                        queue.append(client_id)
                    print(f"[Daemon] {client_id} queued.")
                    conn.sendall(b"WAIT")

            elif command == "RELEASE":
                if client_id == current_holder:
                    print(f"[Daemon] {client_id} released lock.")
                    current_holder = None
                    lock_acquired_time = None
                    # Pass lock to next in queue
                    if queue:
                        next_id = queue.popleft()
                        current_holder = next_id
                        lock_acquired_time = time.time()
                        print(f"[Daemon] Lock auto-transferred to {next_id}")
                conn.sendall(b"RELEASED")
    except Exception as e:
        print(f"[Daemon] Error: {e}")
    finally:
        conn.close()

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[Wait Daemon] Listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()
