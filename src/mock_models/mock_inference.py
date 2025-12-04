"""
Mock inference functions for testing and demonstration.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

async def mock_batch_inference(
    model_name: str, 
    version: str, 
    inputs: List[Any],
    **kwargs
) -> List[Dict[str, Any]]:
    """Mock batch inference function that simulates processing a batch of inputs.
    
    Args:
        model_name: Name of the model to use for inference
        version: Model version
        inputs: List of inputs to process
        **kwargs: Additional keyword arguments
            - latency_ms: Simulated latency in milliseconds (default: 100)
            - error_rate: Probability of error (0-1, default: 0.0)
            
    Returns:
        List of processed results, one for each input
    """
    latency_ms = kwargs.get('latency_ms', 100)
    error_rate = kwargs.get('error_rate', 0.0)
    
    # Log the batch processing
    batch_size = len(inputs)
    logger.info(f"Processing batch of {batch_size} requests with {model_name}:{version}")
    
    # Simulate processing time
    start_time = time.time()
    await asyncio.sleep(latency_ms / 1000.0)  # Convert ms to seconds
    
    # Generate results
    results = []
    for i, input_data in enumerate(inputs):
        # Simulate errors based on error rate
        if error_rate > 0 and (i / batch_size) < error_rate:
            results.append({
                "success": False,
                "error": f"Simulated error processing input {i}",
                "model": model_name,
                "version": version,
                "input_id": i
            })
        else:
            results.append({
                "success": True,
                "result": f"Processed by {model_name}:{version} - {input_data}",
                "model": model_name,
                "version": version,
                "input_id": i,
                "metadata": {
                    "processing_time_ms": latency_ms,
                    "batch_size": batch_size,
                    "batch_position": i,
                    "timestamp": time.time()
                }
            })
    
    # Log completion
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"Processed batch of {batch_size} in {elapsed_ms:.2f}ms")
    
    return results
