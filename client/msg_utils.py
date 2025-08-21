import pickle
import struct

def send_msg(sock, obj):
    """Send a pickled object with length prefix."""
    data = pickle.dumps(obj)
    length = struct.pack("!I", len(data))  # 4-byte unsigned int, network order
    sock.sendall(length + data)

def recv_msg(sock):
    """Receive a pickled object with length prefix."""
    # Read message length (4 bytes)
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack("!I", raw_len)[0]
    # Read the message data
    data = recvall(sock, msg_len)
    return pickle.loads(data)

def recvall(sock, n):
    """Helper to receive n bytes or return None if EOF."""
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
