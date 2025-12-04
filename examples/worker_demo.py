"""
Worker Demo for the Distributed Inference Engine.

This script demonstrates:
1. Starting a worker with model registry integration
2. Registering models with the registry
3. Interactive model inference
4. Monitoring and metrics
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.worker import Worker
from src.config import ModelConfig
from src.model_registry import ModelRegistry
from src.mock_models import FakeModel

class WorkerDemo:
    """Interactive worker demo with model registry integration."""
    
    def __init__(self, worker_id: str, host: str = "0.0.0.0", port: int = 0):
        """Initialize the worker demo."""
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.worker: Optional[Worker] = None
        self.registry = ModelRegistry()
        self.models: Dict[str, Dict] = {
            "test-model": {
                "version": "1.0",
                "batch_size": 8,
                "max_batch_size": 32,
                "input_schema": {"input": "string"},
                "output_schema": {"output": "string"}
            }
        }
    
    async def start(self) -> int:
        """Start the worker and register models."""
        # Create and start the worker
        self.worker = Worker(worker_id=self.worker_id, host=self.host, port=self.port)
        self.port = await self.worker.start()
        
        # Load and register models
        for model_name, config in self.models.items():
            # Load model into worker
            model_config = ModelConfig(
                model_name=model_name,
                model_path=f"models/{model_name}",
                batch_size=config["batch_size"],
                max_batch_size=config["max_batch_size"],
                input_schema=config["input_schema"],
                output_schema=config["output_schema"]
            )
            self.worker.load_model(model_config)
            
            # Register model with registry
            self.registry.register_model(
                model_name=model_name,
                version=config["version"],
                model_path=model_config.model_path,
                input_schema=model_config.input_schema,
                output_schema=model_config.output_schema,
                batch_size=model_config.batch_size,
                max_batch_size=model_config.max_batch_size
            )
            
            # Register shard
            self.registry.add_shard(
                model_name=model_name,
                version=config["version"],
                shard_id=0,  # Single shard per worker in this demo
                worker_id=self.worker_id
            )
        
        return self.port
    
    async def stop(self) -> None:
        """Stop the worker and clean up."""
        if self.worker:
            await self.worker.shutdown()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get worker and model metrics."""
        if not self.worker:
            return {"error": "Worker not started"}
        
        metrics = self.worker.get_metrics()
        
        # Add registry info
        metrics["registry"] = {
            "registered_models": self.registry.list_models(),
            "worker_models": self.registry.get_worker_models(self.worker_id)
        }
        
        return metrics
    
    async def predict(self, model_name: str, input_data: Any) -> Dict[str, Any]:
        """Make a prediction using the specified model."""
        if not self.worker:
            return {"error": "Worker not started"}
        
        # Check if model is registered
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found"}
        
        # Create request
        request = {
            "model": model_name,
            "version": self.models[model_name]["version"],
            "inputs": input_data
        }
        
        # Process request
        return await self.worker._process_request(request)

async def run_demo(worker_id: str, host: str, port: int):
    """Run the interactive worker demo."""
    demo = WorkerDemo(worker_id=worker_id, host=host, port=port)
    
    try:
        port = await demo.start()
        print(f"\n{'='*50}")
        print(f"Worker Demo - ID: {worker_id}")
        print(f"Running on: {host}:{port}")
        print("="*50)
        
        print("\nAvailable commands:")
        print("  predict <model> <input> - Get prediction from model")
        print("  metrics - Show worker and model metrics")
        print("  models - List registered models")
        print("  help - Show this help message")
        print("  exit - Shut down the worker")
        
        while True:
            try:
                # Get user input
                cmd = input("\n> ").strip().lower()
                
                if not cmd:
                    continue
                    
                if cmd in ("exit", "quit", "q"):
                    print("Shutting down...")
                    break
                    
                elif cmd == "metrics":
                    metrics = await demo.get_metrics()
                    print("\n=== Metrics ===")
                    print(f"Worker ID: {metrics['worker_id']}")
                    print(f"Uptime: {metrics['uptime']:.2f}s")
                    print(f"Memory: {metrics['memory_usage_mb']:.2f} MB")
                    print(f"CPU: {metrics['cpu_percent']}%")
                    
                    if metrics.get('model_metrics'):
                        print("\n=== Model Metrics ===")
                        for name, model_metrics in metrics['model_metrics'].items():
                            print(f"\nModel: {name}")
                            print(f"  Requests: {model_metrics['request_count']}")
                            print(f"  Errors: {model_metrics['error_count']}")
                            print(f"  Avg Latency: {model_metrics['avg_latency']:.2f}ms")
                
                elif cmd == "models":
                    models = demo.registry.list_models()
                    print("\n=== Registered Models ===")
                    for model in models:
                        versions = demo.registry.list_versions(model)
                        print(f"\nModel: {model}")
                        for version in versions:
                            print(f"  Version: {version}")
                            shards = demo.registry.get_model_version(model, version).shards
                            for shard in shards:
                                print(f"    Shard {shard.shard_id} on worker {shard.worker_id}")
                
                elif cmd == "help":
                    print("\nAvailable commands:")
                    print("  predict <model> <input> - Get prediction from model")
                    print("  metrics - Show worker and model metrics")
                    print("  models - List registered models")
                    print("  help - Show this help message")
                    print("  exit - Shut down the worker")
                    
                elif cmd.startswith("predict "):
                    parts = cmd[8:].strip().split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: predict <model> <input>")
                        continue
                        
                    model_name, input_text = parts
                    print(f"\nSending prediction request to {model_name}: {input_text}")
                    
                    try:
                        response = await demo.predict(model_name, input_text)
                        print("\n=== Prediction Result ===")
                        print(json.dumps(response, indent=2))
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\nType 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")
                
    finally:
        await demo.stop()

def main():
    """Parse command line arguments and start the demo."""
    parser = argparse.ArgumentParser(description="Distributed Inference Worker Demo")
    parser.add_argument("--worker-id", required=True, help="Unique worker ID")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=0, help="Port to listen on (0 = random)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_demo(
            worker_id=args.worker_id,
            host=args.host,
            port=args.port
        ))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())