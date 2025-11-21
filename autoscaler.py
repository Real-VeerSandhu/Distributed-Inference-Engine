from node import Node

class AutoScaler:
    def __init__(self, nodes):
        self.nodes = nodes
        self.next_id = 3

    def scale_if_needed(self):
        avg = sum(n.gpu_load for n in self.nodes) / len(self.nodes)
        if avg > 0.75:
            nid = f"n{self.next_id}"
            print(f"[Autoscaler] Adding new node {nid}")
            self.nodes.append(Node(id=nid))
            self.next_id += 1
