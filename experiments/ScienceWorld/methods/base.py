"""Abstract base class for memory methods."""

from abc import ABC, abstractmethod
from typing import List


class MemoryModule(ABC):
    """
    Interface for memory methods in ScienceWorld streaming evaluation.

    The evaluation harness calls:
      1. search(query)  — before each episode, to retrieve relevant experience
      2. evolve(...)    — after each episode, to store the new experience

    To implement a new method, subclass MemoryModule and implement both methods.
    """

    @abstractmethod
    def search(self, query: str) -> str:
        """
        Retrieve relevant past experiences for the current task.

        Args:
            query: A string like "task_name: goal_description"

        Returns:
            A formatted string of relevant experiences to inject into the prompt.
            Return "" if no relevant experiences are found.
        """
        ...

    @abstractmethod
    def evolve(
        self,
        task_name: str,
        goal: str,
        trajectory: List[dict],
        success: bool,
        progress: float,
        episode_idx: int,
    ) -> None:
        """
        Store the completed episode as new experience.

        Args:
            task_name: ScienceWorld task identifier
            goal: Natural language goal description
            trajectory: List of dicts with keys like "Action", "Observation", "Goal", "Progress Rate"
            success: Whether all subgoals were completed
            progress: Fraction of subgoals completed (0.0 to 1.0)
            episode_idx: Sequential index of this episode in the streaming run
        """
        ...

    @classmethod
    def add_args(cls, parser) -> None:
        """
        Optionally add method-specific CLI arguments.

        Override this to add --method_xxx style arguments.
        """
        pass
