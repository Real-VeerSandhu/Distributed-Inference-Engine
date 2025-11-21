# Distributed Inference Engine

InferMesh is a terminal-based distributed inference simulator written in Python. It emulates a multi-node inference cluster serving LLM requests using KV-cache aware routing, sharded model execution, and cache offloading policies.

It includes a mini KV-cache manager, a model executor, a router with load balancing, and a Kubernetes-style controller that scales nodes dynamically.

The CLI visualizes:

per-node GPU load

KV-cache usage

routing decisions

cache eviction/offloading events

simulated inference latency

request queueing & backpressure

you are to build this into a working mock prototype
Assume there is no real LLM, we can make a mock file to simualte one in python with a delay etc, 