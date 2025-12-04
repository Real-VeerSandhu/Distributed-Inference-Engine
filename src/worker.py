"""
Worker implementation for the Distributed Inference Engine.

This module provides a worker that can load models, handle inference requests,
and communicate with the coordinator.
"""

import asyncio
import json
import logging
import time
import signal
import psutil
from typing import Any, Dict, List, Optional, Tuple

from src.config import ModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FakeModel:
    """A fake model that simulates inference with configurable latency."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the fake model."""
        self.config = config
        self.model_name = config.model_name
        self.batch_size = config.batch_size
        self.max_batch_size = config.max_batch_size
        self.input_schema = config.input_schema or {}
        self.output_schema = config.output_schema or {}
        
        # Track metrics
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0

    async def predict(self, inputs: Any) -> Dict[str, Any]:
        """Simulate model inference with configurable latency."""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Simulate processing time (50-150ms)
            await asyncio.sleep(0.05 + (time.time() % 0.1))
            
            # For now, just echo back the input with some metadata
            result = {
                "model": self.model_name,
                "output": inputs,
                "metadata": {
                    "batch_size": len(inputs) if isinstance(inputs, (list, tuple)) else 1,
                    "timestamp": time.time()
                }
            }
            
            return result
        except Exception as e:
            self.error_count += 1
            raise
        finally:
            self.total_latency += (time.time() - start_time)

class Worker:
    """Worker that loads models and handles inference requests."""
    
    def __init__(self, worker_id: str, host: str = "0.0.0.0", port: int = 0):
        """Initialize the worker."""
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.models: Dict[str, FakeModel] = {}
        self.server = None
        self._stop_event = asyncio.Event()
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0

    async def start(self) -> int:
        """Start the worker server and return the actual port number."""
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self.shutdown(s))
            )
        
        # Start the server
        self.server = await asyncio.start_server(
            self._handle_connection,
            host=self.host,
            port=self.port
        )
        
        # Get the actual port if port 0 was used (OS-assigned port)
        self.port = self.server.sockets[0].getsockname()[1]
        logger.info(f"Worker {self.worker_id} listening on {self.host}:{self.port}")
        return self.port

    async def shutdown(self, sig=None) -> None:
        """Gracefully shut down the worker."""
        if sig:
            logger.info(f"Received signal {sig.name}, shutting down...")
            
        # Stop accepting new connections
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Unload all models
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)
            
        logger.info(f"Worker {self.worker_id} shutdown complete")
        
        # Exit the application
        if sig:
            import sys
            sys.exit(0)

    async def _handle_connection(self, reader: asyncio.StreamReader, 
                               writer: asyncio.StreamWriter) -> None:
        """Handle incoming client connections."""
        self._request_count += 1
        start_time = time.time()
        response = {"success": False}
        
        try:
            # Read the request
            data = await asyncio.wait_for(reader.read(4096), timeout=30)
            if not data:
                return
                
            # Parse the request
            try:
                request = json.loads(data.decode())
            except json.JSONDecodeError:
                response["error"] = "Invalid JSON"
                raise

            # Process the request
            response = await self._process_request(request)
            if "error" in response:
                self._error_count += 1
                
        except asyncio.TimeoutError:
            response = {"error": "Request timeout", "success": False}
            self._error_count += 1
        except Exception as e:
            response = {"error": str(e), "success": False}
            self._error_count += 1
        finally:
            # Send the response
            try:
                writer.write(json.dumps(response).encode())
                await writer.drain()
            except Exception as e:
                logger.error(f"Error sending response: {e}")
            finally:
                writer.close()
                await writer.wait_closed()
                
            # Log request metrics
            duration = (time.time() - start_time) * 1000  # in ms
            logger.info(
                f"Request completed: status={'success' if response.get('success') else 'error'}, "
                f"duration={duration:.2f}ms, "
                f"total_requests={self._request_count}, "
                f"errors={self._error_count}"
            )

    async def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an inference request."""
        if not isinstance(request, dict):
            return {"error": "Request must be a JSON object", "success": False}

        model_name = request.get("model")
        inputs = request.get("inputs")
        
        if not model_name or inputs is None:
            return {"error": "Missing required fields: model and inputs are required", "success": False}
            
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found", "success": False}
            
        try:
            # Get predictions
            model = self.models[model_name]
            outputs = await model.predict(inputs)
            
            return {
                "model": model_name,
                "outputs": outputs,
                "worker_id": self.worker_id,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e), "success": False}

    def load_model(self, config: ModelConfig) -> bool:
        """Load a model into the worker."""
        if config.model_name in self.models:
            logger.warning(f"Model '{config.model_name}' is already loaded")
            return True
            
        try:
            self.models[config.model_name] = FakeModel(config)
            logger.info(f"Loaded model '{config.model_name}' on worker {self.worker_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading model '{config.model_name}': {e}")
            return False
            
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from the worker."""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Unloaded model '{model_name}' from worker {self.worker_id}")
            return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get current worker metrics."""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        model_metrics = {}
        for name, model in self.models.items():
            avg_latency = (model.total_latency / model.request_count) * 1000 if model.request_count > 0 else 0
            model_metrics[name] = {
                "request_count": model.request_count,
                "error_count": model.error_count,
                "avg_latency_ms": avg_latency
            }
        
        return {
            "worker_id": self.worker_id,
            "uptime_seconds": time.time() - self._start_time,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "models_loaded": list(self.models.keys()),
            "memory_usage_mb": mem_info.rss / (1024 * 1024),
            "cpu_percent": psutil.cpu_percent(),
            "models": model_metrics
        }

async def main():
    """Example usage of the Worker class."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Inference Worker")
    parser.add_argument("--worker-id", required=True, help="Unique worker ID")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=0, help="Port to listen on (0 for random)")
    args = parser.parse_args()
    
    # Create and start the worker
    worker = Worker(
        worker_id=args.worker_id,
        host=args.host,
        port=args.port
    )
    
    # Example: Load a test model
    config = ModelConfig(
        model_name="test-model",
        model_path="/path/to/model",
        batch_size=8,
        max_batch_size=32
    )
    worker.load_model(config)
    
    try:
        # Start the worker
        port = await worker.start()
        print(f"Worker started on port {port}. Press Ctrl+C to stop.")
        
        # Keep the worker running
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Worker error: {e}")
    finally:
        await worker.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
    