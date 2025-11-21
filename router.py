class Router:
    def __init__(self, nodes):
        self.nodes = nodes

    def route(self, request):
        for n in self.nodes:
            if request.session_id in [r.session_id for r in n.request_queue]:
                return n

        # Otherwise load-based routing
        return min(self.nodes, key=lambda n: n.gpu_load)
