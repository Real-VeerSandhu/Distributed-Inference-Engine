"""
Interactive worker example for the Distributed Inference Engine.

This script starts a worker and provides a simple command-line interface
to interact with it.
"""

import asyncio
import json
import argparse
from typing import Dict, Any

# Add the src directory to the path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.worker import Worker, ModelConfig

async def run_worker(worker_id: str, host: str, port: int):
    """Run the worker with an interactive command-line interface."""
    # Create and start the worker
    worker = Worker(worker_id=worker_id, host=host, port=port)
    port = await worker.start()
    
    # Load a test model
    config = ModelConfig(
        model_name="test-model",
        model_path="path/to/pretrained/model",
        batch_size=8,
        max_batch_size=32,
        input_schema={"input": "string"},
        output_schema={"output": "string"}
    )
    worker.load_model(config)
    
    print(f"Worker {worker_id} started on port {port}")
    print("Available commands:")
    print("  predict <input_text> - Get prediction for input text")
    print("  metrics - Show current metrics")
    print("  exit - Shut down the worker")
    print("  help - Show this help message")
    
    try:
        while True:
            try:
                # Get user input
                cmd = input("\n> ").strip()
                
                if not cmd:
                    continue
                    
                if cmd.lower() in ("exit", "quit", "q"):
                    print("Shutting down...")
                    break
                    
                elif cmd.lower() == "metrics":
                    metrics = worker.get_metrics()
                    print("\n=== Metrics ===")
                    print(f"Worker ID: {metrics['worker_id']}")
                    print(f"Uptime: {metrics['uptime_seconds']:.2f}s")
                    print(f"Total Requests: {metrics['total_requests']}")
                    print(f"Total Errors: {metrics['total_errors']}")
                    print(f"Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
                    print(f"CPU Usage: {metrics['cpu_percent']}%")
                    
                    if metrics['models']:
                        print("\nModel Metrics:")
                        for model_name, model_metrics in metrics['models'].items():
                            print(f"\nModel: {model_name}")
                            print(f"  Requests: {model_metrics['request_count']}")
                            print(f"  Errors: {model_metrics['error_count']}")
                            print(f"  Avg Latency: {model_metrics['avg_latency_ms']:.2f}ms")
                    
                elif cmd.lower() == "help":
                    print("\nAvailable commands:")
                    print("  predict <input_text> - Get prediction for input text")
                    print("  metrics - Show current metrics")
                    print("  exit - Shut down the worker")
                    print("  help - Show this help message")
                    
                elif cmd.lower().startswith("predict "):
                    input_text = cmd[8:].strip()
                    if not input_text:
                        print("Error: Please provide input text")
                        continue
                        
                    print(f"\nSending prediction request: {input_text}")
                    try:
                        # Create a request
                        request = {
                            "model": "test-model",
                            "inputs": input_text
                        }
                        
                        # Process the request
                        response = await worker._process_request(request)
                        
                        if response.get("success"):
                            print("\n=== Prediction ===")
                            print(f"Model: {response['model']}")
                            print(f"Output: {json.dumps(response['outputs'], indent=2)}")
                        else:
                            print(f"Error: {response.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        print(f"Error processing request: {e}")
                        
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for a list of commands")
                    
            except (KeyboardInterrupt, EOFError):
                print("\nUse 'exit' to shut down the worker")
            except Exception as e:
                print(f"Error: {e}")
                
    finally:
        # Clean up
        await worker.shutdown()

def main():
    """Parse command line arguments and start the worker."""
    parser = argparse.ArgumentParser(description="Run an interactive model worker")
    parser.add_argument("--worker-id", default="worker-1", help="Worker ID")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=0, help="Port to listen on (0 for random)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_worker(args.worker_id, args.host, args.port))
    except KeyboardInterrupt:
        print("\nWorker stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()