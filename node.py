import random
import time
from dataclasses import dataclass, field
from models import Request

@dataclass
class Node:
    id: str
    gpu_load: float = 0.0
    kv_used: float = 0.0
    kv_capacity: float = 10_000  # MB
    request_queue: list = field(default_factory=list)
    active_sessions: set = field(default_factory=set)
    processing_request: Request = None
    
    def has_kv_for(self, session_id):
        return session_id in self.active_sessions

    def enqueue(self, req):
        self.request_queue.append(req)
        self.active_sessions.add(req.session_id)
        print(f"[NODE {self.id}] Enqueued request #{req.id} (session {req.session_id})")

    def process_requests(self):
        if self.processing_request:
            if self.processing_request.process(self):
                self.active_sessions.discard(self.processing_request.session_id)
                self.processing_request = None
                self.gpu_load = max(0.0, self.gpu_load - 0.1)
        elif self.request_queue:
            self.processing_request = self.request_queue.pop(0)
            self.gpu_load = min(1.0, self.gpu_load + 0.2)

    def simulate_step(self):
        self.process_requests()
        
        if not self.processing_request:
            self.gpu_load = max(0.0, self.gpu_load - 0.05)
        
        self.kv_used = len(self.active_sessions) * 100  # 100MB per session
        self.kv_used = min(self.kv_capacity, max(0, self.kv_used))
