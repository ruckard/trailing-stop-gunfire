import pickle
import struct

def send_msg(sock, obj):
    # TODO: Fix this circular import in a different way
    from utils import print_with_date, debug
    """Send a pickled object with length prefix."""
    data = pickle.dumps(obj)
    length = struct.pack("!I", len(data))  # 4-byte unsigned int, network order
    print_with_date(f"[DEBUG] Sending {len(data)} bytes")
    sock.sendall(length + data)

def recv_msg(sock):
    """Receive a pickled object with length prefix."""
    from utils import print_with_date
    # Read message length (4 bytes)
    raw_len = recvall(sock, 4)
    if raw_len is None or len(raw_len) != 4:
        print_with_date(f"[DEBUG] Invalid length prefix received: {raw_len}")
        return None
    msg_len = struct.unpack("!I", raw_len)[0]
    print_with_date(f"[DEBUG] Expecting {msg_len} bytes")
    data = recvall(sock, msg_len)
    if data is None:
        print_with_date(f"[DEBUG] Incomplete data received, expected {msg_len} bytes")
        return None
    print_with_date(f"[DEBUG] Received {len(data)} bytes")
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
