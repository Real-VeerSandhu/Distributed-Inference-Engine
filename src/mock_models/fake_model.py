"""
Fake model implementation for testing and demonstration.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from src.config import ModelConfig

class FakeModel:
    """A fake model that simulates inference with configurable latency."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the fake model.
        
        Args:
            config: Model configuration
        """
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
        self.last_inference_time = 0.0

    async def predict(self, inputs: Any) -> Dict[str, Any]:
        """Simulate model inference with configurable latency.
        
        Args:
            inputs: Input data for the model
            
        Returns:
            Dictionary containing the model output and metadata
        """
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
                    "timestamp": time.time(),
                    "request_id": f"req_{int(time.time() * 1000)}"
                }
            }
            
            return result
        except Exception as e:
            self.error_count += 1
            raise
        finally:
            latency = time.time() - start_time
            self.total_latency += latency
            self.last_inference_time = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model metrics.
        
        Returns:
            Dictionary containing model metrics
        """
        return {
            "model_name": self.model_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_latency": (self.total_latency / self.request_count) if self.request_count > 0 else 0,
            "last_inference_time": self.last_inference_time,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size
        }
