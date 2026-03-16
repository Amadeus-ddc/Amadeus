"""No-memory baseline — agent has no experience retrieval."""

from typing import List, Tuple
from .base import MemoryModule


class NoMemory(MemoryModule):
    """Baseline: no memory, no experience retrieval."""

    def __init__(self, **kwargs):
        pass

    def search(self, query: str) -> str:
        return ""

    def evolve(
        self,
        task_type: str,
        task_description: str,
        trajectory: List[Tuple[str, str]],
        success: bool,
        episode_idx: int,
    ) -> None:
        pass
