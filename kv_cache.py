class KVCacheManager:
    def __init__(self, nodes):
        self.nodes = nodes

    def enforce_policy(self):
        for n in self.nodes:
            if n.kv_used > n.kv_capacity * 0.85:
                evict_amt = n.kv_capacity * 0.2
                n.kv_used -= evict_amt
                print(f"[CACHE] Evicted {evict_amt/1000:.1f}GB from {n.id}")
