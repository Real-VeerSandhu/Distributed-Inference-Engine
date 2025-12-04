"""
Mid-Stage Demo - Complete Integration Test

This demo brings together all the components built so far:
- KV Store: Caching inference results
- Model Registry: Tracking models and shards
- Router: Routing requests to workers
- Worker: Processing inference requests

The demo shows a complete request flow:
1. Client sends request
2. Check KV cache (cache hit/miss)
3. Route to appropriate worker
4. Worker processes request
5. Cache result
6. Return to client

This helps identify any integration issues early.
"""

import asyncio
import json
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kvstore import KVCache
from src.model_registry import ModelRegistry
from src.router import Router
from src.worker import Worker
from src.config import ModelConfig


class InferenceClient:
    """
    Simple client that demonstrates the complete inference flow.
    
    This simulates what a coordinator would do:
    1. Check cache
    2. Route request
    3. Send to worker
    4. Cache result
    """
    
    def __init__(
        self,
        kvstore: KVCache,
        router: Router,
        cache_enabled: bool = True
    ):
        """Initialize the inference client."""
        self.kvstore = kvstore
        self.router = router
        self.cache_enabled = cache_enabled
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _generate_cache_key(self, model_name: str, version: str, inputs: Any) -> str:
        """Generate a cache key from request parameters."""
        # Create a hash of the inputs for consistent caching
        inputs_str = json.dumps(inputs, sort_keys=True)
        inputs_hash = hashlib.md5(inputs_str.encode()).hexdigest()
        return f"{model_name}:{version}:{inputs_hash}"
    
    async def send_request(
        self,
        model_name: str,
        version: str,
        inputs: Any,
        request_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send an inference request through the complete pipeline.
        
        Args:
            model_name: Name of the model
            version: Model version
            inputs: Input data for inference
            request_key: Key for routing (if None, uses hash of inputs)
            
        Returns:
            Response dictionary with results
        """
        self.request_count += 1
        start_time = time.time()
        
        # Generate request key if not provided
        if request_key is None:
            request_key = hashlib.md5(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        
        # Step 1: Check cache
        cache_key = self._generate_cache_key(model_name, version, inputs)
        if self.cache_enabled:
            cached_result = self.kvstore.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                return {
                    "success": True,
                    "cached": True,
                    "result": cached_result,
                    "latency_ms": (time.time() - start_time) * 1000
                }
        
        self.cache_misses += 1
        
        # Step 2: Route request
        shard = self.router.route_request(model_name, version, request_key)
        if shard is None:
            return {
                "success": False,
                "error": "Could not route request - no available worker"
            }
        
        # Step 3: Get worker address
        worker_address = self.router.get_worker_address(shard.worker_id)
        if worker_address is None:
            return {
                "success": False,
                "error": f"Worker {shard.worker_id} address not found"
            }
        
        # Step 4: Send request to worker
        try:
            result = await self._send_to_worker(worker_address, {
                "model": model_name,
                "inputs": inputs
            })
            
            # Step 5: Cache result if successful
            if result.get("success") and self.cache_enabled:
                self.kvstore.set(cache_key, result, ttl=300)  # 5 minute TTL
            
            # Step 6: Update router health tracking
            if result.get("success"):
                self.router.mark_worker_success(shard.worker_id)
            else:
                self.router.mark_worker_failure(shard.worker_id)
            
            result["latency_ms"] = (time.time() - start_time) * 1000
            result["shard"] = shard.shard_id
            result["worker"] = shard.worker_id
            return result
            
        except Exception as e:
            self.router.mark_worker_failure(shard.worker_id)
            return {
                "success": False,
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000
            }
    
    async def _send_to_worker(self, address: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to a worker via TCP."""
        host, port = address.split(':')
        port = int(port)
        
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0
            )
            
            # Send request
            request_json = json.dumps(request)
            writer.write(request_json.encode())
            await writer.drain()
            
            # Read response
            response_data = await asyncio.wait_for(reader.read(4096), timeout=10.0)
            writer.close()
            await writer.wait_closed()
            
            # Parse response
            response = json.loads(response_data.decode())
            return response
            
        except asyncio.TimeoutError:
            raise Exception(f"Timeout connecting to worker at {address}")
        except Exception as e:
            raise Exception(f"Error communicating with worker: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        total = self.request_count
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            "total_requests": total,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_enabled": self.cache_enabled
        }


async def setup_system():
    """Set up the complete system with all components."""
    print("=" * 70)
    print("MID-STAGE DEMO: Complete System Integration")
    print("=" * 70)
    print()
    
    print("STEP 1: Initializing components...")
    
    # KV Store for caching
    kvstore = KVCache(max_size=1000, eviction_policy="lru")
    print("  ✓ KV Store initialized")
    
    # Model Registry
    registry = ModelRegistry()
    print("  ✓ Model Registry initialized")
    
    # Router
    router = Router(
        registry=registry,
        health_check_interval=10.0,  # Longer interval for demo
        failover_enabled=True
    )
    await router.start()
    print("  ✓ Router started")
    
    print("\nSTEP 2: Registering model...")
    
    registry.register_model(
        model_name="text-classifier",
        version="1.0",
        model_path="/models/text-classifier-v1",
        input_schema={"text": "string"},
        output_schema={"label": "string", "confidence": "float"},
        batch_size=1,
        max_batch_size=32
    )
    print("  ✓ Model 'text-classifier' v1.0 registered")
    
    print("\nSTEP 3: Starting workers...")
    
    workers = []
    worker_configs = [
        ("worker-1", "127.0.0.1", 9001),
        ("worker-2", "127.0.0.1", 9002),
    ]
    
    for worker_id, host, port in worker_configs:
        worker = Worker(worker_id=worker_id, host=host, port=port)
        actual_port = await worker.start()
        
        # Load model on worker
        model_config = ModelConfig(
            model_name="text-classifier",
            model_path="/models/text-classifier-v1",
            batch_size=1,
            max_batch_size=32,
            input_schema={"text": "string"},
            output_schema={"label": "string", "confidence": "float"}
        )
        worker.load_model(model_config)
        
        workers.append((worker, f"{host}:{actual_port}"))
        print(f"  ✓ Worker {worker_id} started on {host}:{actual_port}")
    
    print("\nSTEP 4: Registering shards and workers...")
    
    for i, (worker, address) in enumerate(workers):
        worker_id = worker.worker_id
        
        # Add shard to registry
        registry.add_shard(
            model_name="text-classifier",
            version="1.0",
            shard_id=i,
            worker_id=worker_id
        )
        
        # Register worker with router
        router.register_worker(worker_id, address)
        router.mark_worker_success(worker_id)  # Mark as healthy for demo
        
        print(f"  ✓ Shard {i} on {worker_id} registered")

    print("\nSTEP 5: Creating inference client...")
    client = InferenceClient(kvstore, router, cache_enabled=True)
    print("  ✓ Client created")
    
    return {
        "kvstore": kvstore,
        "registry": registry,
        "router": router,
        "workers": workers,
        "client": client
    }


async def run_demo(system: Dict[str, Any]):
    """Run the actual demo with requests."""
    client = system["client"]
    kvstore = system["kvstore"]
    router = system["router"]
    
    print("\n" + "=" * 70)
    print("DEMO: Processing Inference Requests")
    print("=" * 70)
    print()
    
    # Test requests
    test_requests = [
        {"user_id": "user-123", "text": "This product is amazing!"},
        {"user_id": "user-456", "text": "Not very good quality"},
        {"user_id": "user-789", "text": "I love it!"},
        {"user_id": "user-123", "text": "This product is amazing!"},  # Cache hit
        {"user_id": "user-456", "text": "Not very good quality"},    # Cache hit
    ]
    
    print("Processing requests...")
    print()
    
    for i, req in enumerate(test_requests, 1):
        user_id = req["user_id"]
        text = req["text"]
        
        print(f"Request {i}: User {user_id}")
        print(f"  Input: '{text}'")
        
        # Send request
        response = await client.send_request(
            model_name="text-classifier",
            version="1.0",
            inputs=text,
            request_key=user_id
        )
        
        if response.get("success"):
            cached = response.get("cached", False)
            latency = response.get("latency_ms", 0)
            shard = response.get("shard")
            worker = response.get("worker")
            
            print(f"  Status: {'CACHE HIT' if cached else 'CACHE MISS'} → Worker {worker} (Shard {shard})")
            print(f"  Latency: {latency:.2f}ms")
            
            # Show output - handle both cached and non-cached responses
            if cached:
                output = response.get("result", {}).get("outputs", response.get("result", {}))
            else:
                output = response.get("outputs", response)
            
            print(f"  Output: {json.dumps(output, indent=4)}")
        else:
            print(f"  Status: ERROR - {response.get('error')}")
        
        print()
        
        # Small delay between requests
        await asyncio.sleep(0.1)
    
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print()
    
    # Client stats
    client_stats = client.get_stats()
    print("Client Statistics:")
    print(f"  Total Requests: {client_stats['total_requests']}")
    print(f"  Cache Hits: {client_stats['cache_hits']}")
    print(f"  Cache Misses: {client_stats['cache_misses']}")
    print(f"  Cache Hit Rate: {client_stats['cache_hit_rate']:.1%}")
    print()
    
    # KV Store stats
    kv_stats = kvstore.get_stats()
    print("KV Store Statistics:")
    print(f"  Size: {kv_stats['size']}/{kv_stats['max_size']}")
    print(f"  Hits: {kv_stats['hits']}")
    print(f"  Misses: {kv_stats['misses']}")
    print(f"  Hit Rate: {kv_stats['hit_rate']:.1%}")
    print()
    
    # Router stats
    router_stats = router.get_stats()
    print("Router Statistics:")
    print(f"  Total Workers: {router_stats['total_workers']}")
    print(f"  Healthy Workers: {router_stats['healthy_workers']}")
    print(f"  Unhealthy Workers: {router_stats['unhealthy_workers']}")
    print()


async def cleanup(system: Dict[str, Any]):
    """Clean up all resources."""
    print("Cleaning up...")
    
    router = system["router"]
    workers = system["workers"]
    
    # Stop router
    await router.stop()
    
    # Stop workers
    for worker, _ in workers:
        await worker.shutdown()
    
    print("✓ Cleanup complete")


async def main():
    """Main demo function."""
    system = None
    try:
        # Setup
        system = await setup_system()
        
        # Run demo
        await run_demo(system)
        
        # Keep system running for a bit to show health checks
        print("System running... (press Ctrl+C to stop)")
        print("(Health checks will run in background)")
        await asyncio.sleep(2)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if system:
            await cleanup(system)


if __name__ == "__main__":
    asyncio.run(main())

