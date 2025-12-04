"""
Configuration classes for the Distributed Inference Engine.

This module contains shared configuration dataclasses used across
different components of the system.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_name: str
    model_path: str
    batch_size: int = 1
    max_batch_size: int = 32
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

