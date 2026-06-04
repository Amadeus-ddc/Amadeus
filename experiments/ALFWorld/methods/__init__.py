"""Amadeus memory plugin for ALFWorld evaluation."""

from .base import MemoryModule
from .amadeus import AmadeusMemory

METHODS = {
    "amadeus": AmadeusMemory,
}


def load_method(name: str, **kwargs) -> MemoryModule:
    """Load the Amadeus memory method."""
    if name not in METHODS:
        available = ", ".join(sorted(METHODS.keys()))
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    return METHODS[name](**kwargs)
