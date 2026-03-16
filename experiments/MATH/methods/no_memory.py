"""No-memory baseline."""

from .base import MemoryModule


class NoMemory(MemoryModule):
    def __init__(self, **kwargs):
        pass

    def search(self, query: str) -> str:
        return ""

    def evolve(self, problem, reasoning, answer, is_correct, problem_idx):
        pass
