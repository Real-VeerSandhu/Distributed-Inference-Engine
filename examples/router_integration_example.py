"""
Example showing how Router integrates with other components.

This demonstrates the typical flow:
1. Setup: Registry + Router + Workers
2. Request flow: Coordinator uses Router to route requests
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import ModelRegistry
from src.router import Router


async def demonstrate_router_usage():
    """Demonstrate how router fits into the system."""
    
    print("=" * 60)
    print("Router Integration Example")
    print("=" * 60)
    print()
    
    # ============================================================
    # STEP 1: Setup Components
    # ============================================================
    print("STEP 1: Setting up components...")
    
    # Create model registry
    registry = ModelRegistry()
    
    # Create router (requires registry)
    router = Router(
        registry=registry,
        health_check_interval=5.0,
        failover_enabled=True
    )
    
    # Start router (begins background health checks)
    await router.start()
    print("✓ Router started")
    
    # ============================================================
    # STEP 2: Register Model and Shards
    # ============================================================
    print("\nSTEP 2: Registering model with shards...")
    
    # Register a model
    registry.register_model(
        model_name="text-classifier",
        version="1.0",
        model_path="/models/text-classifier-v1",
        input_schema={"text": "string"},
        output_schema={"label": "string", "confidence": "float"},
        batch_size=8,
        max_batch_size=32
    )
    print("✓ Model 'text-classifier' v1.0 registered")
    
    # Add shards (each shard on a different worker)
    registry.add_shard("text-classifier", "1.0", shard_id=0, worker_id="worker-1")
    registry.add_shard("text-classifier", "1.0", shard_id=1, worker_id="worker-2")
    registry.add_shard("text-classifier", "1.0", shard_id=2, worker_id="worker-3")
    print("✓ Added 3 shards")
    
    # ============================================================
    # STEP 3: Register Workers
    # ============================================================
    print("\nSTEP 3: Registering workers...")
    
    router.register_worker("worker-1", "127.0.0.1:9001")
    router.register_worker("worker-2", "127.0.0.1:9002")
    router.register_worker("worker-3", "127.0.0.1:9003")
    print("✓ Registered 3 workers")
    
    # For demo purposes, mark workers as healthy (in production, health checks do this)
    # In real system, you'd wait for health checks or workers would register as healthy
    router.mark_worker_success("worker-1")
    router.mark_worker_success("worker-2")
    router.mark_worker_success("worker-3")
    print("✓ Marked workers as healthy (for demo)")
    
    # ============================================================
    # STEP 4: Simulate Coordinator Routing Requests
    # ============================================================
    print("\nSTEP 4: Simulating request routing (as Coordinator would do)...")
    print()
    
    # Simulate multiple requests
    requests = [
        {"user_id": "user-123", "text": "This is great!"},
        {"user_id": "user-456", "text": "Not so good"},
        {"user_id": "user-789", "text": "Amazing product"},
        {"user_id": "user-123", "text": "Another message"},  # Same user
    ]
    
    for req in requests:
        user_id = req["user_id"]
        text = req["text"]
        
        # This is what Coordinator would do:
        # 1. Route the request
        shard = router.route_request(
            model_name="text-classifier",
            version="1.0",
            request_key=user_id  # Use user_id for consistent routing
        )
        
        if shard:
            # 2. Get worker address
            address = router.get_worker_address(shard.worker_id)
            
            print(f"Request from {user_id}:")
            print(f"  → Routed to shard {shard.shard_id}")
            print(f"  → Worker: {shard.worker_id} at {address}")
            print(f"  → Text: '{text}'")
            print()
            
            # 3. In real coordinator, would send request to worker:
            #    await send_to_worker(address, {"model": "text-classifier", "inputs": text})
            
            # Simulate success (updates router health tracking)
            router.mark_worker_success(shard.worker_id)
        else:
            print(f"ERROR: Could not route request for {user_id}")
    
    # ============================================================
    # STEP 5: Demonstrate Consistent Routing
    # ============================================================
    print("\nSTEP 5: Demonstrating consistent routing...")
    print("(Same user_id always goes to same shard)")
    print()
    
    test_user = "user-123"
    for i in range(3):
        shard = router.route_request("text-classifier", "1.0", test_user)
        if shard:
            print(f"Request {i+1} for {test_user} → Shard {shard.shard_id} (Worker {shard.worker_id})")
        else:
            print(f"Request {i+1} for {test_user} → ERROR: Could not route")
    
    # ============================================================
    # STEP 6: Demonstrate Failover
    # ============================================================
    print("\nSTEP 6: Demonstrating failover...")
    print("(When worker-1 fails, requests automatically route to backup)")
    print()
    
    # Simulate worker-1 failing
    for _ in range(3):
        router.mark_worker_failure("worker-1")
    
    # Now route a request that would normally go to worker-1
    shard = router.route_request("text-classifier", "1.0", "user-123")
    print(f"Request for user-123 (normally worker-1):")
    if shard:
        print(f"  → Routed to shard {shard.shard_id} (Worker {shard.worker_id})")
        print(f"  → Failover active!")
    else:
        print(f"  → ERROR: Could not route (all workers down?)")
    
    # ============================================================
    # STEP 7: Show Statistics
    # ============================================================
    print("\nSTEP 7: Router Statistics...")
    stats = router.get_stats()
    print(f"Total Workers: {stats['total_workers']}")
    print(f"Healthy: {stats['healthy_workers']}")
    print(f"Unhealthy: {stats['unhealthy_workers']}")
    print(f"Failover Enabled: {stats['failover_enabled']}")
    
    # ============================================================
    # Cleanup
    # ============================================================
    print("\nCleaning up...")
    await router.stop()
    print("✓ Router stopped")


async def simulate_coordinator_usage():
    """
    Simulate how Coordinator would use Router in practice.
    
    This shows the typical pattern:
    1. Coordinator receives request
    2. Checks cache (kvstore)
    3. If miss, uses router to find worker
    4. Sends to batcher or directly to worker
    """
    
    print("\n" + "=" * 60)
    print("Simulated Coordinator Usage Pattern")
    print("=" * 60)
    print()
    
    # Setup
    registry = ModelRegistry()
    router = Router(registry)
    await router.start()
    
    # Register model and workers
    registry.register_model(
        model_name="my-model",
        version="1.0",
        model_path="/models/my-model",
        input_schema={"input": "string"},
        output_schema={"output": "string"}
    )
    registry.add_shard("my-model", "1.0", 0, "worker-1")
    router.register_worker("worker-1", "127.0.0.1:9001")
    
    # Simulate coordinator receiving request
    async def coordinator_handle_request(model_name, version, request_key, inputs):
        """This is what coordinator would do."""
        
        # Step 1: Check cache (would use kvstore here)
        # cache_key = f"{model_name}:{version}:{hash(inputs)}"
        # cached = kvstore.get(cache_key)
        # if cached:
        #     return cached
        
        # Step 2: Route request
        shard = router.route_request(model_name, version, request_key)
        if not shard:
            return {"error": "No available worker"}
        
        # Step 3: Get worker address
        address = router.get_worker_address(shard.worker_id)
        
        # Step 4: Send to worker (or batcher)
        # In real implementation:
        # response = await send_to_worker(address, {
        #     "model": model_name,
        #     "version": version,
        #     "inputs": inputs
        # })
        
        # Step 5: Update health based on result
        # if response.success:
        #     router.mark_worker_success(shard.worker_id)
        #     # kvstore.set(cache_key, response)
        # else:
        #     router.mark_worker_failure(shard.worker_id)
        
        return {
            "shard": shard.shard_id,
            "worker": shard.worker_id,
            "address": address
        }
    
    # Test it
    result = await coordinator_handle_request(
        "my-model", "1.0", "user-123", "test input"
    )
    print("Coordinator routing result:")
    print(f"  Shard: {result['shard']}")
    print(f"  Worker: {result['worker']}")
    print(f"  Address: {result['address']}")
    
    await router.stop()


if __name__ == "__main__":
    print("\n")
    asyncio.run(demonstrate_router_usage())
    asyncio.run(simulate_coordinator_usage())
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

