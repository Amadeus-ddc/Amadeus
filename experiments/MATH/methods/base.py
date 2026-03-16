"""Abstract base class for memory methods in MATH-500 streaming evaluation."""

from abc import ABC, abstractmethod
from typing import List


class MemoryModule(ABC):
    """
    Interface for memory methods in MATH-500 streaming evaluation.

    The evaluation harness calls:
      1. search(query)  — before each problem, retrieve relevant prior solutions
      2. evolve(...)    — after each problem, store the result

    To implement a new method, subclass MemoryModule and implement both.
    """

    @abstractmethod
    def search(self, query: str) -> str:
        """
        Retrieve relevant prior solutions for the current problem.

        Args:
            query: The math problem text.

        Returns:
            Formatted string of relevant prior solutions to inject into prompt.
            Return "" if nothing relevant.
        """
        ...

    @abstractmethod
    def evolve(
        self,
        problem: str,
        reasoning: str,
        answer: str,
        is_correct: bool,
        problem_idx: int,
    ) -> None:
        """
        Store the solved problem as new experience.

        Args:
            problem: The problem statement
            reasoning: The model's reasoning text
            answer: The extracted answer
            is_correct: Whether the answer matched gold
            problem_idx: Sequential index
        """
        ...

    @classmethod
    def add_args(cls, parser) -> None:
        """Optionally add method-specific CLI arguments."""
        pass
