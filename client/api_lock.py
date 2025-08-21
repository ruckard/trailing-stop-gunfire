# api_lock.py
import socket
import time
import pickle

WAIT_DAEMON_HOST = '127.0.0.1'
WAIT_DAEMON_PORT = 5005

def api_lock_send_command(command, client_id):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((WAIT_DAEMON_HOST, WAIT_DAEMON_PORT))
            # Send command as a pickled string, e.g. "LOCK client1"
            s.sendall(pickle.dumps(f"{command} {client_id}"))
            response = s.recv(4096)
            return pickle.loads(response)   # response is also pickled
    except Exception as e:
        print(f"[(LOCK) Client:{client_id}] Error: {e}")
        return "ERROR"

def api_lock_acquire_lock(client_id):
    while True:
        result = api_lock_send_command("LOCK", client_id)
        if result == "GRANTED":
            return
        elif result == "WAIT":
            time.sleep(0.5)
        else:
            print(f"[(LOCK) Client:{client_id}] Unexpected response: {result}")
            time.sleep(1)

def api_lock_release_lock(client_id):
    result = api_lock_send_command("RELEASE", client_id)
    # You can log result if you want
    return result
