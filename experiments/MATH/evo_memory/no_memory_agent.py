"""
No-memory baseline agent for MATH-500 streaming evaluation.

Answers questions using direct LLM inference without any memory.
"""

import logging
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

    def answer(self, question: str) -> str:
        prompt = (
            f"What is the correct answer to this question: {question}\n\n"
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

        self._last_raw_response = raw
        return self._parse_answer(raw)

    def receive_feedback(
        self,
        question: str,
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
    def _parse_answer(text: str) -> str:
        if not text:
            return ""
        # Priority 1: \\boxed{}
        if "\\boxed{" in text:
            idx = text.rfind("\\boxed{")
            brace_count = 0
            start = idx + len("\\boxed{")
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    if brace_count == 0:
                        return text[start:i].strip()
                    brace_count -= 1
        # Priority 2: "The correct answer is ..." (possibly multiline)
        m = re.search(
            r"(?:correct answer is|answer is)\s*[:\s]*([\s\S]+?)(?:\n\n|\Z)",
            text, re.IGNORECASE,
        )
        if m:
            ans = m.group(1).strip().rstrip(".")
            ans = re.sub(r"^\\\[", "", ans)
            ans = re.sub(r"\\\]$", "", ans)
            ans = ans.strip().strip("$").strip()
            if ans:
                return ans
        # Fallback: return last non-empty line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else ""
