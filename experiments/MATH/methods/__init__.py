"""
Memory method plugins for MATH-500 streaming evaluation.

To add a new method:
  1. Create methods/your_method.py implementing MemoryModule
  2. Register it in METHODS dict below
  3. Run: python run_math500.py --method your_method
"""

from .base import MemoryModule
from .no_memory import NoMemory
from .amadeus import AmadeusMemory

METHODS = {
    "none": NoMemory,
    "amadeus": AmadeusMemory,
}


def load_method(name: str, **kwargs) -> MemoryModule:
    if name not in METHODS:
        available = ", ".join(sorted(METHODS.keys()))
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    return METHODS[name](**kwargs)
