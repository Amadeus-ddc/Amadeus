"""
Abstract interface for memory-augmented models in Evo-Memory streaming evaluation.

Any model with a memory module can be evaluated on GPQA by implementing MemoryAgent.
The evaluator calls methods in this order for each question in the stream:

    answer_text = agent.answer(question, choices)
    is_correct  = check(answer_text, ground_truth)
    agent.receive_feedback(question, choices, answer_text, is_correct)

This mirrors the Evo-Memory search -> synthesis -> evolve loop while leaving
all memory internals (storage, retrieval, update) entirely to the model.
"""

from abc import ABC, abstractmethod
from typing import List


class MemoryAgent(ABC):
    """Base class that any memory-augmented model must implement."""

    @abstractmethod
    def answer(self, question: str, choices: List[str]) -> str:
        """
        Answer a 4-choice question, optionally leveraging stored memories.

        Args:
            question: The question text.
            choices:  Four answer texts, indexed as [A, B, C, D].

        Returns:
            One of "A", "B", "C", "D".
        """

    @abstractmethod
    def receive_feedback(
        self,
        question: str,
        choices: List[str],
        model_answer: str,
        is_correct: bool,
    ) -> None:
        """
        Receive a correctness signal after answering.
        The model should use this to evolve its memory.

        NOTE: Only told correct / wrong.  The correct answer is NOT revealed.

        Args:
            question:     The question text.
            choices:      Four answer texts.
            model_answer: The letter the model chose ("A"/"B"/"C"/"D").
            is_correct:   True if the answer was correct, False otherwise.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory state for a fresh evaluation run."""

    @property
    def name(self) -> str:
        """Return the display name of this agent (used in logs / results)."""
        return self.__class__.__name__

    def finalize(self) -> None:
        """Called after evaluation loop ends. Override to flush remaining state."""
        pass
