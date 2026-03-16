"""
ReMem agent for GPQA streaming evaluation.

Strict reproduction of the ReMem method from:
  "Evo-Memory: Benchmarking LLM Agent Test-time Learning
   with Self-Evolving Memory" (Wei et al., 2025)

For single-turn QA (GPQA), ReMem and ExpRAG share the SAME answer prompt
(Appendix C single-turn template). The only difference is the Evolve phase:

  ExpRAG:  store experience (append)
  ReMem:   store experience (append) + Refine Memory (LLM prunes noisy memories)

Paper Figure 7: ~36.8% pruning rate on GPQA.
"""

import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from openai import OpenAI

from .base_agent import MemoryAgent
from .exprag_agent import _Embedder

logger = logging.getLogger("EvoMemory.ReMemAgent")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AMADEUS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))))


class ReMemAgent(MemoryAgent):
    """
    ReMem for single-turn QA.

    answer()           — identical to ExpRAG (Appendix C template)
    receive_feedback() — ExpRAG store + LLM-driven memory pruning (Refine Memory)
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-7b-instruct",
        api_base: str = "http://localhost:8006/v1",
        api_key: str = "EMPTY",
        embedder_model_path: str = None,
        top_k: int = 4,
        **kwargs,
    ):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
        self.top_k = top_k

        # Embedder (same as ExpRAG)
        if embedder_model_path is None:
            embedder_model_path = os.path.join(
                _AMADEUS_ROOT, "amadeus", "models", "all-MiniLM-L6-v2"
            )
        self.embedder = _Embedder(embedder_model_path, device="cpu")

        # Memory pool: ID-addressable for pruning
        # ExpRAG uses List[tuple(vec, text)], we need IDs for targeted deletion
        self.memory: List[Dict] = []
        self._next_id: int = 0

        # Cache for receive_feedback
        self._last_raw_response: str = ""
        self._last_retrieved_ids: List[int] = []

    # ------------------------------------------------------------------
    # answer() — IDENTICAL to ExpRAG (paper Appendix C single-turn template)
    # ------------------------------------------------------------------

    def answer(self, question: str, choices: List[str]) -> str:
        choice_block = "\n".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )

        # Retrieve top-k similar experiences
        retrieved = self._retrieve(question, self.top_k)
        self._last_retrieved_ids = [e["id"] for e in retrieved]

        # Build prompt — paper Appendix C single-turn template (same as ExpRAG)
        memory_section = ""
        if retrieved:
            memory_block = "\n\n".join(
                f"[Experience #{i+1}]\n{e['text']}"
                for i, e in enumerate(retrieved)
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
            f"Choices:\n{choice_block}\n\n"
            "Provide your output in the following format:\n"
            "- Rationale: your short reasoning, may cite memory if useful\n"
            "- Final Answer: The correct answer is (insert answer here)"
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
        return self._parse_letter(raw)

    # ------------------------------------------------------------------
    # receive_feedback() — ExpRAG store + ReMem Refine Memory (prune)
    # ------------------------------------------------------------------

    def receive_feedback(
        self,
        question: str,
        choices: List[str],
        model_answer: str,
        is_correct: bool,
    ) -> None:
        # --- Step 1: Store experience (same as ExpRAG) ---
        status = "Correct" if is_correct else "Wrong"
        choice_block = "\n".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )

        experience_text = (
            f"Question: {question}\n"
            f"Choices:\n{choice_block}\n"
            f"Model response:\n{self._last_raw_response}\n"
            f"Outcome: {status}"
        )

        vec = self.embedder.embed(question)
        if vec is not None:
            entry = {
                "id": self._next_id,
                "embedding": vec,
                "text": experience_text,
                "question": question,
                "is_correct": is_correct,
            }
            self.memory.append(entry)
            self._next_id += 1
            logger.debug(
                f"Stored experience #{entry['id']} ({status}), "
                f"pool size: {len(self.memory)}"
            )
        else:
            logger.warning("Failed to embed experience, skipping storage")

        # --- Step 2: Refine Memory — prune (ReMem only, not in ExpRAG) ---
        if len(self.memory) >= 5:
            self._refine_memory()

    # ------------------------------------------------------------------
    # Refine Memory (Section 3.3, Figure 3b, Figure 7)
    # ------------------------------------------------------------------

    def _refine_memory(self) -> None:
        """LLM reviews recent memories and prunes noisy ones.

        Paper Figure 7: GPQA has ~36.8% pruning rate.
        Only reviews a bounded window to keep prompt size manageable.
        """
        # Candidates: recent entries + entries retrieved for the last question
        recent_ids = {e["id"] for e in self.memory[-(self.top_k * 2):]}
        retrieved_ids = set(self._last_retrieved_ids)
        candidate_ids = recent_ids | retrieved_ids

        candidates = [e for e in self.memory if e["id"] in candidate_ids]
        if len(candidates) < 2:
            return

        # Build concise memory summaries for the LLM
        mem_lines = []
        for entry in candidates:
            short_q = entry["question"][:120]
            status = "Correct" if entry["is_correct"] else "Wrong"
            mem_lines.append(f"[ID={entry['id']}] ({status}) Q: {short_q}")
        mem_block = "\n".join(mem_lines)

        prompt = (
            f"You are managing a memory pool for a question-answering system.\n"
            f"The pool has {len(self.memory)} entries total. "
            f"Review these {len(candidates)} recent entries:\n\n"
            f"{mem_block}\n\n"
            f"Which memories should be PRUNED (removed) because they are:\n"
            f"- Wrong answers with flawed reasoning (likely to mislead)\n"
            f"- Redundant with another entry that has a better outcome\n"
            f"- Low quality or uninformative\n\n"
            f"Keep correct answers and useful lessons from mistakes.\n\n"
            f"Respond EXACTLY in this format:\n"
            f"Prune: <comma-separated IDs to remove, or \"none\">\n"
            f"Reason: <brief explanation>"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Refine Memory LLM call failed: {e}")
            return

        # Parse prune response
        prune_match = re.search(r"Prune:\s*(.+)", raw, re.IGNORECASE)
        if not prune_match:
            return

        prune_text = prune_match.group(1).strip().lower()
        if "none" in prune_text and not re.search(r"\d", prune_text):
            logger.debug("Refine Memory: no memories pruned")
            return

        prune_ids = {int(x) for x in re.findall(r"\d+", prune_text)}
        # Safety: only prune candidates we actually showed to the LLM
        valid_ids = {e["id"] for e in candidates}
        prune_ids = prune_ids & valid_ids

        if prune_ids:
            before = len(self.memory)
            self.memory = [e for e in self.memory if e["id"] not in prune_ids]
            logger.info(
                f"Refine Memory: pruned {before - len(self.memory)} entries "
                f"(IDs: {sorted(prune_ids)}), pool: {before} -> {len(self.memory)}"
            )

    # ------------------------------------------------------------------
    # Retrieval (same as ExpRAG)
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, k: int) -> List[Dict]:
        """Retrieve top-k most similar memory entries via cosine similarity."""
        if not self.memory:
            return []

        query_vec = self.embedder.embed(query)
        if query_vec is None:
            return []

        scored = []
        for entry in self.memory:
            sim = float(np.dot(query_vec, entry["embedding"]))
            scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    # ------------------------------------------------------------------
    # Boilerplate
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.memory.clear()
        self._next_id = 0
        self._last_raw_response = ""
        self._last_retrieved_ids = []

    @property
    def name(self) -> str:
        return f"ReMem({self.model_name})"

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
