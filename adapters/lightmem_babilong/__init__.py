"""
LightMemory adapter for BABILong benchmark.

This package provides a complete adapter for running LightMemory on the BABILong dataset.

Key components:
- DataConverter: Converts BABILong format to LightMemory format
- LightMemWrapper: Wraps LightMemory for BABILong
- ResultProcessor: Processes and evaluates results
- run_experiment.py: Main experiment runner

Constraints:
- Does NOT modify LightMemory's core reading/processing/retrieval mechanisms
- Does NOT modify BABILong's evaluation protocol
- All code is isolated in this adapters/ directory
"""

from .data_converter import DataConverter
from .model_wrapper import LightMemWrapper
from .result_processor import ResultProcessor

__version__ = "0.1.0"
__all__ = [
    "DataConverter",
    "LightMemWrapper",
    "ResultProcessor",
]
