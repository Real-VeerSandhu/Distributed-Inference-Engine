"""
Mock models for testing and demonstration purposes.

This module provides mock implementations of models that can be used for
testing and demonstration without requiring actual model files.
"""

from .fake_model import FakeModel
from .mock_inference import mock_batch_inference

__all__ = ['FakeModel', 'mock_batch_inference']
