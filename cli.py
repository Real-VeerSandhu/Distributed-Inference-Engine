import time
import os

class Dashboard:
    def __init__(self, sim):
        self.sim = sim
        self.last_clear = time.time()
        
    def clear_screen(self):
        # Clear screen and move cursor to top left
        print('\033[2J\033[H', end='')
        
    def _format_node_row(self, node):
        gpu_usage = node.gpu_load
        kv_usage = node.kv_used / node.kv_capacity
        
        status = "Active" if node.processing_request else "Idle"
        if node.processing_request:
            status = f"\033[91m{status}\033[0m"  # Red for active
        
        gpu_str = f"{gpu_usage*100:3.0f}%"
        if gpu_usage > 0.7:
            gpu_str = f"\033[91m{gpu_str}\033[0m"  # Red for high usage
        elif gpu_usage > 0.4:
            gpu_str = f"\033[93m{gpu_str}\033[0m"  # Yellow for medium usage
            
        kv_str = f"{node.kv_used/1000:.1f}/{node.kv_capacity/1000:.1f}GB"
        if kv_usage > 0.7:
            kv_str = f"\033[91m{kv_str}\033[0m"  # Red for high usage
        elif kv_usage > 0.4:
            kv_str = f"\033[93m{kv_str}\033[0m"  # Yellow for medium usage
            
        return f"{node.id:5} | {gpu_str:5} | {kv_str:15} | {status:6} | {len(node.request_queue):2} waiting"

    def render(self):
        # Clear screen every 0.5 seconds to prevent flickering
        current_time = time.time()
        if current_time - self.last_clear > 0.5:
            self.clear_screen()
            self.last_clear = current_time
        
        # Header
        print(f"\033[1;34mInferMesh - Distributed LLM Inference Simulator\033[0m")
        print(f"Nodes: {len(self.sim.nodes)} | ", end='')
        print(f"Auto-gen: \033[93m{'ON' if self.sim.auto_generate else 'OFF'}\033[0m | ", end='')
        print(f"Requests: {self.sim.request_id} total\n")
        
        # Nodes table
        print(f"{'Node':5} | {'GPU':5} | {'KV Cache':15} | {'Status':6} | Queue")
        print("-" * 50)
        
        for node in sorted(self.sim.nodes, key=lambda n: int(n.id[1:])):
            print(self._format_node_row(node))
        
        # Controls
        print("\nControls:")
        print("  a: Add request")
        print("  s: Toggle auto-gen")
        print("  q: Quit\n")
        print("-" * 50)
        print("\033[?25l", end='')  # Hide cursor

