import random
import time
from dataclasses import dataclass, field

@dataclass
class Node:
    id: str
    gpu_load: float = 0.0
    kv_used: float = 0.0
    kv_capacity: float = 10_000  # MB
    request_queue: list = field(default_factory=list)

    def has_kv_for(self, session_id):
        return session_id in self.request_queue

    def enqueue(self, req):
        self.request_queue.append(req)

    def simulate_step(self):
        self.gpu_load = max(0.0, min(1.0, self.gpu_load + random.uniform(-0.05, 0.1)))

        self.kv_used += random.uniform(-50, 120)
        self.kv_used = min(self.kv_capacity, max(0, self.kv_used))

        if self.request_queue and random.random() > 0.5:
            self.request_queue.pop(0)
