"""
ExpRAG agent for MATH-500 streaming evaluation.

Reproduces the ExpRAG baseline from:
  "Evo-Memory: Benchmarking LLM Agent Test-time Learning
   with Self-Evolving Memory" (Wei et al., 2025)

Core idea:
  - Each experience m_i = S(x_i, ŷ_i, f_i) is stored as structured text
  - At step t, retrieve top-k most similar experiences via embedding cosine sim
  - Inject retrieved experiences into prompt following the paper's template
  - Model answers conditioned on relevant past experiences (not ALL history)
"""

import logging
import os
import re
import sys
from typing import List, Optional

import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer

from .base_agent import MemoryAgent

logger = logging.getLogger("EvoMemory.ExpRAGAgent")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AMADEUS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))))


class _Embedder:
    """Lightweight sentence embedder using mean pooling."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def embed(self, text: str) -> Optional[np.ndarray]:
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            pooled = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            vec = pooled[0].cpu().numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None


class ExpRAGAgent(MemoryAgent):
    """
    ExpRAG: Experience Retrieval and Aggregation.

    Memory pool stores structured experience entries.
    At each step, top-k most similar experiences are retrieved
    via cosine similarity and injected into the prompt.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-32b-instruct",
        api_base: str = "http://localhost:8006/v1",
        api_key: str = "EMPTY",
        embedder_model_path: str = None,
        top_k: int = 4,
        **kwargs,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
        self.top_k = top_k

        # Embedder
        if embedder_model_path is None:
            embedder_model_path = os.path.join(
                _AMADEUS_ROOT, "amadeus", "models", "all-MiniLM-L6-v2"
            )
        self.embedder = _Embedder(embedder_model_path, device="cpu")

        # Memory pool: list of (embedding, experience_text)
        self.memory: List[tuple] = []
        # Cache the last raw LLM response so receive_feedback can store rationale
        self._last_raw_response: str = ""

    def answer(self, question: str) -> str:
        # Retrieve top-k similar experiences
        retrieved = self._retrieve(question, self.top_k)

        # Build prompt — always use paper's Appendix C template (unified format)
        memory_section = ""
        if retrieved:
            memory_block = "\n\n".join(
                f"[Experience #{i+1}]\n{exp}" for i, exp in enumerate(retrieved)
            )
            memory_section = (
                "Below are some retrieved LOCAL EXPERIENCE MEMORIES:\n\n"
                f"{memory_block}\n\n"
            )

        prompt = (
            "You are a helpful assistant with access to LOCAL EXPERIENCE MEMORY. "
            "Each memory may contain past experience, rationales, domains, and skills. "
            f"{memory_section}"
            "Now solve the following problem.\n"
            f"Question: {question}\n\n"
            "Provide your output in the following format:\n"
            "- Rationale: your short reasoning, may cite memory if useful\n"
            '- Final Answer: The correct answer is (insert answer here)'
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

        # Cache raw response so receive_feedback can store full rationale
        self._last_raw_response = raw
        return self._parse_answer(raw)

    def receive_feedback(
        self,
        question: str,
        model_answer: str,
        is_correct: bool,
    ) -> None:
        """Store structured experience m_i = S(x_i, ŷ_i, f_i) into memory pool.

        Includes the model's full rationale (ŷ_i), not just the answer.
        """
        status = "Correct" if is_correct else "Wrong"

        # Structured experience entry with rationale (paper Section 3.2)
        # m_i = S(x_i, ŷ_i, f_i): input + full model output + feedback
        experience = (
            f"Question: {question}\n"
            f"Model response:\n{self._last_raw_response}\n"
            f"Outcome: {status}"
        )

        # Embed using question text (retrieval key)
        vec = self.embedder.embed(question)
        if vec is not None:
            self.memory.append((vec, experience))
            logger.debug(f"Stored experience #{len(self.memory)} (pool size: {len(self.memory)})")
        else:
            logger.warning("Failed to embed experience, skipping storage")

    def reset(self) -> None:
        self.memory.clear()

    @property
    def name(self) -> str:
        return f"ExpRAG({self.model_name})"

    def _retrieve(self, query: str, k: int) -> List[str]:
        """Retrieve top-k most similar experiences via cosine similarity."""
        if not self.memory:
            return []

        query_vec = self.embedder.embed(query)
        if query_vec is None:
            return []

        # Compute cosine similarities (vectors are already normalized)
        scores = []
        for vec, text in self.memory:
            sim = float(np.dot(query_vec, vec))
            scores.append((sim, text))

        # Sort by similarity, take top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scores[:k]]

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
        # Priority 2: "Final Answer: ..." or "The correct answer is ..." (possibly multiline)
        m = re.search(
            r"(?:Final Answer|correct answer is|answer is)\s*[:\s]*([\s\S]+?)(?:\n\n|\Z)",
            text, re.IGNORECASE,
        )
        if m:
            ans = m.group(1).strip().rstrip(".")
            ans = re.sub(r"^(?:The correct answer is|the correct answer is)\s*", "", ans).strip()
            ans = re.sub(r"^\\\[", "", ans)
            ans = re.sub(r"\\\]$", "", ans)
            ans = ans.strip().strip("$").strip()
            if ans:
                return ans
        # Fallback: return last non-empty line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else ""
