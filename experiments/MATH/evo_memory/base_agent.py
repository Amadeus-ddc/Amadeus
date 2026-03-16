"""
Abstract interface for memory-augmented models in Evo-Memory streaming evaluation.

Any model with a memory module can be evaluated on MATH-500 by implementing MemoryAgent.
The evaluator calls methods in this order for each question in the stream:

    answer_text = agent.answer(question)
    is_correct  = check(answer_text, ground_truth)
    agent.receive_feedback(question, answer_text, is_correct)

This mirrors the Evo-Memory search -> synthesis -> evolve loop while leaving
all memory internals (storage, retrieval, update) entirely to the model.

NOTE: MATH is open-ended (no multiple-choice), so the interface omits `choices`.
"""

from abc import ABC, abstractmethod


class MemoryAgent(ABC):
    """Base class that any memory-augmented model must implement."""

    @abstractmethod
    def answer(self, question: str) -> str:
        """
        Answer an open-ended math question, optionally leveraging stored memories.

        Args:
            question: The math problem text.

        Returns:
            The predicted answer string (e.g. "42", "\\frac{1}{2}").
        """

    @abstractmethod
    def receive_feedback(
        self,
        question: str,
        model_answer: str,
        is_correct: bool,
    ) -> None:
        """
        Receive a correctness signal after answering.
        The model should use this to evolve its memory.

        NOTE: Only told correct / wrong.  The correct answer is NOT revealed.

        Args:
            question:     The math problem text.
            model_answer: The answer the model produced.
            is_correct:   True if the answer was correct, False otherwise.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory state for a fresh evaluation run."""

    @property
    def name(self) -> str:
        """Return the display name of this agent (used in logs / results)."""
        return self.__class__.__name__
