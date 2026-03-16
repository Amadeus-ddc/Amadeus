"""
No-memory baseline agent for GPQA streaming evaluation.

Answers questions using direct LLM inference without any memory.
"""

import logging
import os
import re
from typing import List

from openai import OpenAI
from .base_agent import MemoryAgent

logger = logging.getLogger("EvoMemory.NoMemoryAgent")


class NoMemoryAgent(MemoryAgent):
    """Baseline: no memory, direct LLM inference."""

    def __init__(
        self,
        model_name: str = "qwen2.5-7b-instruct",
        api_base: str = "http://localhost:8006/v1",
        api_key: str = "EMPTY",
        **kwargs,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name

    def answer(self, question: str, choices: List[str]) -> str:
        choice_block = "\n".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )
        prompt = (
            f"What is the correct answer to this question: {question}\n\n"
            f"Choices:\n{choice_block}\n\n"
            f'Format your response as follows: "The correct answer is (insert answer here)"'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a very intelligent assistant, who follows instructions directly."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raw = ""

        return self._parse_letter(raw)

    def receive_feedback(
        self,
        question: str,
        choices: List[str],
        model_answer: str,
        is_correct: bool,
    ) -> None:
        pass  # No memory to update

    def reset(self) -> None:
        pass  # Nothing to reset

    @property
    def name(self) -> str:
        return f"NoMemory({self.model_name})"

    @staticmethod
    def _parse_letter(text: str) -> str:
        if not text:
            return "A"
        patterns = [
            r"answer is \(([A-D])\)",
            r"Answer: \(([A-D])\)",
            r"answer: \(([A-D])\)",
            r"answer \(([A-D])\)",
            r"\(([A-D])\)\s*$",
            r"\(([A-D])\)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
        m = re.search(r"\b([A-D])\b", text)
        if m:
            return m.group(1)
        return "A"
