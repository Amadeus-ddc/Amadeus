"""
Memory method plugins for ScienceWorld streaming evaluation.

To add a new method:
  1. Create a file in this directory (e.g., methods/your_method.py)
  2. Define a class that inherits from MemoryModule
  3. Implement search() and evolve()
  4. Register it in METHODS dict below

Example:
    class YourMemory(MemoryModule):
        def __init__(self, **kwargs): ...
        def search(self, query: str) -> str: ...
        def evolve(self, task_name, goal, trajectory, success, progress, idx): ...
"""

from .base import MemoryModule
from .no_memory import NoMemory
from .amadeus import AmadeusMemory
from .history import HistoryMemory
from .exprag import ExpRAGMemory

# Registry: method_name -> class
METHODS = {
    "none": NoMemory,
    "amadeus": AmadeusMemory,
    "history": HistoryMemory,
    "exprag": ExpRAGMemory,
}


def load_method(name: str, **kwargs) -> MemoryModule:
    """Load a memory method by name."""
    if name not in METHODS:
        available = ", ".join(sorted(METHODS.keys()))
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    return METHODS[name](**kwargs)
