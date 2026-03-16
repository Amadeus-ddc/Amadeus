"""Abstract base class for memory methods in ALFWorld evaluation."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class MemoryModule(ABC):
    """
    Interface for memory methods in ALFWorld streaming evaluation.

    The evaluation harness calls:
      1. search(query)  — before each episode, retrieve relevant past experience
      2. evolve(...)    — after each episode, store the new experience

    For batch (non-streaming) mode, search() always returns "" and evolve() is a no-op,
    regardless of method — the harness controls this, not the method.

    To implement a new method, subclass MemoryModule and implement both.
    """

    @abstractmethod
    def search(self, query: str) -> str:
        """
        Retrieve relevant past experiences for the current task.

        Args:
            query: A string describing the current task goal.

        Returns:
            Formatted string of relevant experience to inject into prompt.
            Return "" if no relevant experience found.
        """
        ...

    @abstractmethod
    def evolve(
        self,
        task_type: str,
        task_description: str,
        trajectory: List[Tuple[str, str]],
        success: bool,
        episode_idx: int,
    ) -> None:
        """
        Store the completed episode as new experience.

        Args:
            task_type: ALFWorld task type (e.g. "pick_and_place")
            task_description: Natural language goal
            trajectory: List of (observation, action) pairs
            success: Whether the task was completed
            episode_idx: Sequential index of this episode in the streaming run
        """
        ...

    @classmethod
    def add_args(cls, parser) -> None:
        """Optionally add method-specific CLI arguments."""
        pass
