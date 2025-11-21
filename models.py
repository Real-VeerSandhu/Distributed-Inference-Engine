from dataclasses import dataclass, field
from time import time

@dataclass
class Request:
    id: int
    session_id: int
    created_at: float = field(default_factory=time)
    tokens_generated: int = 0
    
    def process(self, node):
        if not hasattr(self, '_start_time'):
            self._start_time = time()
            
        self.tokens_generated += 1
        
        if self.tokens_generated >= 20:
            latency = time() - self._start_time
            print(f"[NODE {node.id}] Completed request #{self.id} in {latency*1000:.1f}ms")
            return True
            
        return False