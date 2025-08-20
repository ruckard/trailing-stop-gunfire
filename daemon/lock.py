import time
from collections import deque

class LockManager:
    def __init__(self, ttl=10):
        self.ttl = ttl
        self.current_holder = None
        self.lock_acquired_time = None
        self.queue = deque()

    def handle(self, command, client_id):
        now = time.time()
        # Timeout expired holder
        if self.current_holder and (now - self.lock_acquired_time > self.ttl):
            self.current_holder = None
            self.lock_acquired_time = None

        if command == "LOCK":
            if self.current_holder is None:
                self.current_holder = client_id
                self.lock_acquired_time = now
                return "GRANTED"
            elif client_id == self.current_holder:
                return "GRANTED"
            else:
                if client_id not in self.queue:
                    self.queue.append(client_id)
                return "WAIT"

        elif command == "RELEASE":
            if client_id == self.current_holder:
                self.current_holder = None
                self.lock_acquired_time = None
                if self.queue:
                    next_id = self.queue.popleft()
                    self.current_holder = next_id
                    self.lock_acquired_time = now
            return "RELEASED"
