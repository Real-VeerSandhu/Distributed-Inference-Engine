# Router Component - How It Works

## Overview

The `router.py` component is the **traffic director** of the distributed inference engine. It decides which worker node should handle each inference request based on the model, request key, and worker health status.

## Architecture Position

```
┌─────────────┐
│   Client    │
│  Request    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│        COORDINATOR                   │
│  ┌──────────────────────────────┐   │
│  │  1. Check KV Cache           │   │
│  │  2. If miss, use ROUTER ────┼───┼──┐
│  │  3. Send to BATCHER          │   │  │
│  └──────────────────────────────┘   │  │
└─────────────────────────────────────┘  │
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │      ROUTER           │
                              │  ┌────────────────┐  │
                              │  │ 1. Query      │  │
                              │  │    Registry   │  │
                              │  │ 2. Hash-based │  │
                              │  │    Sharding   │  │
                              │  │ 3. Health     │  │
                              │  │    Check      │  │
                              │  │ 4. Failover   │  │
                              │  └────────┬───────┘  │
                              └───────────┼──────────┘
                                          │
                    ┌────────────────────┼────────────────────┐
                    │                    │                      │
                    ▼                    ▼                      ▼
            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
            │   Worker 1   │    │   Worker 2   │    │   Worker 3   │
            │  (Shard 0)   │    │  (Shard 1)   │    │  (Shard 2)   │
            └──────────────┘    └──────────────┘    └──────────────┘
```

## How Router Works

### 1. **Request Routing Flow**

When the coordinator needs to route a request:

```python
# Coordinator calls router
shard = router.route_request(
    model_name="my-model",
    version="1.0",
    request_key="user-123"  # Used for consistent hashing
)

# Router returns a ModelShard with worker_id
# Coordinator uses router.get_worker_address(shard.worker_id)
# to get "127.0.0.1:9001" and sends request there
```

### 2. **Hash-Based Sharding**

The router uses **consistent hashing** to ensure the same request key always goes to the same shard:

```python
# Inside router.route_request():
shard = registry.get_shard_for_key(model_name, version, request_key)
# This does: hash(request_key) % num_shards → shard_id
```

**Example:**
- Request key: `"user-123"` → Hash → `shard_id = 0` → `worker-1`
- Request key: `"user-456"` → Hash → `shard_id = 2` → `worker-3`
- Same key always maps to same shard (important for caching/stateful models)

### 3. **Worker Health Monitoring**

The router continuously monitors worker health:

```python
# Background task runs every 5 seconds (configurable)
async def _health_check_loop():
    while running:
        await asyncio.sleep(5.0)
        # Try TCP connection to each worker
        # Mark healthy/unhealthy based on results
```

**Health States:**
- **HEALTHY**: Worker responds to health checks
- **UNHEALTHY**: 3+ consecutive failures (configurable)
- **UNKNOWN**: Not yet checked or just registered

### 4. **Automatic Failover**

If the primary shard's worker is unhealthy, router automatically finds a backup:

```python
# Primary shard worker is down
if worker.health != WorkerHealth.HEALTHY:
    # Find alternative healthy shard for same model
    return _find_alternative_shard(model_name, version, request_key)
```

**Failover Strategy:**
- Finds all healthy shards for the model
- Uses round-robin based on request key hash
- Ensures same key → same backup shard (for consistency)

## Key Components

### Router Class

**Main Responsibilities:**
1. **Worker Management**: Register/unregister workers
2. **Request Routing**: Map requests to workers via shards
3. **Health Monitoring**: Background health checks
4. **Failover**: Automatic routing to healthy workers

**Key Methods:**
- `route_request()` - Main routing logic
- `register_worker()` - Add a worker
- `mark_worker_success/failure()` - Update health status
- `get_worker_address()` - Get network address for worker

### Integration Points

**1. Model Registry (Required)**
```python
router = Router(registry=model_registry)
# Router queries registry to find shards for models
```

**2. Coordinator (Uses Router)**
```python
# Coordinator will use router like this:
shard = router.route_request(model, version, key)
address = router.get_worker_address(shard.worker_id)
# Send request to address
```

**3. Workers (Register with Router)**
```python
# When worker starts, it registers:
router.register_worker("worker-1", "127.0.0.1:9001")
```

## Example Workflow

### Setup Phase:
```python
# 1. Create registry and router
registry = ModelRegistry()
router = Router(registry)

# 2. Register a model with shards
registry.register_model("my-model", "1.0", ...)
registry.add_shard("my-model", "1.0", shard_id=0, worker_id="worker-1")
registry.add_shard("my-model", "1.0", shard_id=1, worker_id="worker-2")

# 3. Register workers
router.register_worker("worker-1", "127.0.0.1:9001")
router.register_worker("worker-2", "127.0.0.1:9002")

# 4. Start router (begins health checks)
await router.start()
```

### Request Phase:
```python
# Coordinator receives request for "my-model" v1.0 with key "user-123"

# 1. Router finds shard
shard = router.route_request("my-model", "1.0", "user-123")
# Returns: ModelShard(shard_id=0, worker_id="worker-1")

# 2. Get worker address
address = router.get_worker_address("worker-1")
# Returns: "127.0.0.1:9001"

# 3. Coordinator sends request to worker-1
# 4. If worker-1 fails, router automatically routes to worker-2
```

## Benefits

1. **Consistency**: Same request key → same worker (important for stateful models)
2. **Reliability**: Automatic failover when workers go down
3. **Scalability**: Easy to add/remove workers and shards
4. **Monitoring**: Built-in health tracking and statistics
5. **Flexibility**: Configurable health check intervals, timeouts, failure thresholds

## Configuration Options

```python
router = Router(
    registry=registry,
    health_check_interval=5.0,      # Check every 5 seconds
    health_check_timeout=2.0,        # 2 second timeout
    max_consecutive_failures=3,      # Mark unhealthy after 3 failures
    failover_enabled=True            # Enable automatic failover
)
```

## Future Enhancements

- **Consistent Hashing**: Better distribution when workers are added/removed
- **Load-Based Routing**: Route to workers with lowest load
- **Geographic Routing**: Route to nearest worker
- **Circuit Breaker**: Temporarily stop routing to repeatedly failing workers

