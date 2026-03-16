"""No-memory baseline — agent has no experience retrieval."""

from typing import List
from .base import MemoryModule


class NoMemory(MemoryModule):
    """Baseline: no memory, no experience retrieval."""

    def __init__(self, **kwargs):
        pass

    def search(self, query: str) -> str:
        return ""

    def evolve(
        self,
        task_name: str,
        goal: str,
        trajectory: List[dict],
        success: bool,
        progress: float,
        episode_idx: int,
    ) -> None:
        pass
