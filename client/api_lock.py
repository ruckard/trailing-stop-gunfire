# api_lock.py
import socket
import time
from .msg_utils import send_msg, recv_msg   # assume helpers are in msg_utils.py

WAIT_DAEMON_HOST = '127.0.0.1'
WAIT_DAEMON_PORT = 5005

def api_lock_send_command(command, client_id):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((WAIT_DAEMON_HOST, WAIT_DAEMON_PORT))
            send_msg(s, f"{command} {client_id}")
            response = recv_msg(s)
            return response
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
    return api_lock_send_command("RELEASE", client_id)
