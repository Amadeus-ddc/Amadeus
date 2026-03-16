"""
Memory method plugins for ALFWorld evaluation (both streaming and batch).

To add a new method:
  1. Create methods/your_method.py implementing MemoryModule
  2. Register it in METHODS dict below
  3. Run: python run_alfworld_streaming.py --method your_method
"""

from .base import MemoryModule
from .no_memory import NoMemory
from .amadeus import AmadeusMemory
from .history import HistoryMemory
from .exprag import ExpRAGMemory
from .memp import MempMemory
from .mcma import MCMAMemory

# Registry: method_name -> class
METHODS = {
    "none": NoMemory,
    "amadeus": AmadeusMemory,
    "history": HistoryMemory,
    "exprag": ExpRAGMemory,
    "memp": MempMemory,
    "mcma": MCMAMemory,
}


def load_method(name: str, **kwargs) -> MemoryModule:
    """Load a memory method by name."""
    if name not in METHODS:
        available = ", ".join(sorted(METHODS.keys()))
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    return METHODS[name](**kwargs)
