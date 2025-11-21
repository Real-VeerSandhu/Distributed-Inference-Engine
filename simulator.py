from router import Router
from autoscaler import AutoScaler
from kv_cache import KVCacheManager
from models import Request
import random
from typing import List
from node import Node

class Simulator:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.router = Router(nodes)
        self.autoscaler = AutoScaler(nodes)
        self.cache = KVCacheManager(nodes)
        self.request_id = 0
        self.auto_generate = True
        self.requests_processed = 0

    def tick(self):
        if self.auto_generate and random.random() < 0.6:
            self.generate_request()

        for node in self.nodes:
            node.simulate_step()

        self.cache.enforce_policy()
        self.autoscaler.scale_if_needed()
        
        # Update metrics
        for node in self.nodes:
            if not node.request_queue and not node.processing_request:
                node.gpu_load = max(0, node.gpu_load - 0.02)

    def generate_request(self):
        req = Request(id=self.request_id, session_id=random.randint(1, 20))
        self.request_id += 1
        node = self.router.route(req)
        node.enqueue(req)
        print(f"[ROUTER] sent req#{req.id} (sess {req.session_id}) → {node.id}")
        return req

    def toggle_auto_generate(self):
        self.auto_generate = not self.auto_generate
        status = "ON" if self.auto_generate else "OFF"
        print(f"\n[SYSTEM] Auto-request generation: {status}")
        return status
