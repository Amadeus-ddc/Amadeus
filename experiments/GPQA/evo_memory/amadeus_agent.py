"""
Amadeus memory agent adapter for Evo-Memory GPQA evaluation.

Uses the FULL Amadeus pipeline:
  - MemoryGraph        : knowledge graph (storage + retrieval)
  - BuilderAgent       : buffer -> graph write  (evolve / store)
  - AnswererAgent      : Graph RAG SEARCH->WALK->READ (answer)
  - QuestionerAgent    : adversarial question generation (attack)
  - AdversarialOptimizer : self-play loop (attack/defend/judge/evolve)

The LLM is accessed via an OpenAI-compatible API (vLLM / local server).
"""

import json
import logging
import os
import re
import sys
from typing import List, Optional

import networkx as nx
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Path setup — make sure `amadeus` package is importable
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AMADEUS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))))
if _AMADEUS_ROOT not in sys.path:
    sys.path.insert(0, _AMADEUS_ROOT)

from amadeus.code.core.graph import MemoryGraph
from amadeus.code.agents.builder import BuilderAgent
from amadeus.code.agents.answerer import AnswererAgent
from amadeus.code.agents.questioner import QuestionerAgent
from amadeus.code.engine.optimizer import AdversarialOptimizer

from .base_agent import MemoryAgent

logger = logging.getLogger("EvoMemory.AmadeusAgent")

# ---------------------------------------------------------------------------
# Embedder (reused from amadeus LoCoMo experiment)
# ---------------------------------------------------------------------------

class HuggingFaceEmbedder:
    """Thin wrapper so MemoryGraph can call embedder.embed(text)."""

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
            return pooled[0].cpu().numpy()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Amadeus MemoryAgent
# ---------------------------------------------------------------------------

class AmadeusMemoryAgent(MemoryAgent):
    """
    Wraps the full Amadeus pipeline for Evo-Memory evaluation.

    answer():
        Uses AnswererAgent's Graph RAG (SEARCH -> WALK -> READ)
        to explore the knowledge graph and produce an answer.

    receive_feedback():
        1. BuilderAgent.process_buffer  — store experience into graph
        2. AdversarialOptimizer.step    — self-play loop:
             Questioner generates questions (attack)
             Answerer answers from graph   (defend)
             Optimizer judges, patches graph, evolves agent prompts

    Nothing inside Amadeus core code is modified.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-7b-instruct",
        api_base: str = "http://localhost:8006/v1",
        api_key: str = "EMPTY",
        graph_storage_path: str = None,
        embedder_model_path: str = None,
        top_k_retrieve: int = 4,
        enable_self_play: bool = True,
        self_play_mode: str = "adaptive_buffer_fixed_sp",
        self_play_questions: int = 3,
        use_cot: bool = False,
        buffer_size: int = 3,
    ):
        # --- LLM client (OpenAI-compatible, e.g. vLLM) ---
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
        self.top_k = top_k_retrieve
        self.enable_self_play = enable_self_play
        self.self_play_questions = self_play_questions
        self.use_cot = use_cot

        # Map ablation-style mode to optimizer mode (same as run_locomo.py)
        if self_play_mode == "adaptive_buffer_fixed_sp":
            self.optimizer_mode = "fixed"
            self.optimizer_fixed_count = self_play_questions
        elif self_play_mode == "fixed_buffer_adaptive_sp":
            self.optimizer_mode = "adaptive"
            self.optimizer_fixed_count = None
        elif self_play_mode == "fixed_buffer_fixed_sp_cot":
            self.optimizer_mode = "fixed"
            self.optimizer_fixed_count = self_play_questions
            self.use_cot = True
        else:  # "none" -> adaptive buffer + adaptive self-play
            self.optimizer_mode = "adaptive"
            self.optimizer_fixed_count = None

        # --- Experience Buffer (matches LoCoMo's buffering pattern) ---
        self.buffer_size = buffer_size
        self._experience_buffer: List[str] = []
        self._kept_items: List[str] = []

        # --- Embedder ---
        if embedder_model_path is None:
            embedder_model_path = os.path.join(
                _AMADEUS_ROOT, "amadeus", "models", "all-MiniLM-L6-v2"
            )
        self.embedder = HuggingFaceEmbedder(embedder_model_path, device="cuda")

        # --- MemoryGraph ---
        if graph_storage_path is None:
            graph_storage_path = os.path.join(_THIS_DIR, "data", "gpqa_memory_graph.json")
        os.makedirs(os.path.dirname(graph_storage_path), exist_ok=True)
        self.graph_storage_path = graph_storage_path
        self.graph = MemoryGraph(storage_path=graph_storage_path, embedder=self.embedder)

        # --- Ensure env vars so Amadeus BaseAgent.__init__ can create OpenAI client ---
        os.environ["OPENAI_BASE_URL"] = api_base
        os.environ["OPENAI_API_KEY"] = api_key

        # --- BuilderAgent (memory write / evolve) ---
        self.builder = BuilderAgent(graph=self.graph, model_name=model_name)
        self.builder.client = OpenAI(base_url=api_base, api_key=api_key)
        self.builder.model_name = model_name

        # --- AnswererAgent (Graph RAG: SEARCH -> WALK -> READ) ---
        self.answerer = AnswererAgent(
            graph=self.graph, model_name=model_name,
            api_base=api_base, api_key=api_key,
        )

        # --- QuestionerAgent (adversarial attack) ---
        self.questioner = QuestionerAgent(
            model_name=model_name,
            api_base=api_base, api_key=api_key,
        )

        # --- AdversarialOptimizer (self-play judge + prompt evolution) ---
        self.optimizer = AdversarialOptimizer(
            questioner=self.questioner,
            builder=self.builder,
            answerer=self.answerer,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
        )

        # --- Bookkeeping ---
        self._question_idx = 0

    # ---- MemoryAgent interface ----

    def answer(self, question: str, choices: List[str]) -> str:
        self._question_idx += 1

        # Format GPQA question for AnswererAgent's Graph RAG
        choice_block = "\n".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )
        formatted_q = (
            f"{question}\n\n"
            f"Choices:\n{choice_block}\n\n"
            f"Which is the correct answer? Reply with the letter (A), (B), (C), or (D)."
        )

        # Use AnswererAgent's full Graph RAG pipeline
        try:
            raw_answer = self.answerer.answer(formatted_q)
        except Exception as e:
            logger.error(f"AnswererAgent.answer() failed: {e}")
            raw_answer = ""

        letter = self._parse_letter(raw_answer)
        logger.info(
            f"Q{self._question_idx}: AnswererAgent returned: {raw_answer!r} -> parsed={letter}  "
            f"(graph: {self.graph.graph.number_of_nodes()} nodes, "
            f"{self.graph.graph.number_of_edges()} edges)"
        )
        return letter

    def receive_feedback(
        self,
        question: str,
        choices: List[str],
        model_answer: str,
        is_correct: bool,
    ) -> None:
        status = "CORRECT" if is_correct else "WRONG"
        choice_block = "\n".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )
        answer_idx = ord(model_answer) - ord("A")
        answer_text = choices[answer_idx] if 0 <= answer_idx < len(choices) else model_answer

        experience = (
            f"--- Question #{self._question_idx} [{status}] ---\n"
            f"Q: {question}\n"
            f"Choices:\n{choice_block}\n"
            f"My answer: ({model_answer}) {answer_text}\n"
            f"Outcome: {status}\n"
        )

        # Accumulate experiences, flush when buffer is full
        self._experience_buffer.append(experience)
        if len(self._experience_buffer) >= self.buffer_size:
            self._flush_buffer()

    # Scene prefix to override Builder's "minimalism" heuristic for GPQA
    GPQA_SCENE_PREFIX = (
        "--- Scene: Scientific Knowledge Accumulation ---\n"
        "You are building a long-term scientific knowledge graph from exam questions.\n"
        "IMPORTANT: Every question below contains valuable domain knowledge. "
        "Do NOT ignore them as 'temporary test results'.\n"
        "For each question:\n"
        "  - ADD a node for the core concept/topic (e.g., 'Krebs Cycle', 'Maxwell Equations').\n"
        "  - ADD edges for key facts, correct reasoning, and common misconceptions.\n"
        "  - If the outcome is WRONG, store the correct reasoning as a corrective edge "
        "(e.g., subject='Maxwell Equations', object='Magnetic Monopoles', "
        "content='Monopoles change BOTH div(B) and curl(E), not just div(B)').\n"
        "  - Treat 'Subject' as the scientific concept, NOT a person.\n\n"
    )

    def _flush_buffer(self) -> None:
        """Flush accumulated experiences through Builder + Optimizer (mirrors LoCoMo flow)."""
        if not self._experience_buffer:
            return

        # Prepend scene context + WAIT carry-over items
        combined = self.GPQA_SCENE_PREFIX
        if self._kept_items:
            combined += "--- Deferred Context (from previous rounds) ---\n"
            combined += "\n".join(f"- {item}" for item in self._kept_items)
            combined += "\n\n"
        combined += "\n\n".join(self._experience_buffer)

        # ====== Step 1: Builder stores experiences into graph ======
        action_log = []
        try:
            kept, action_log, _ = self.builder.process_buffer(combined)
            self._kept_items = kept if kept else []
            logger.info(
                f"Q{self._question_idx} Buffer flush: {len(self._experience_buffer)} experiences, "
                f"{len(action_log)} ops, {len(self._kept_items)} WAIT items  "
                f"(graph: {self.graph.graph.number_of_nodes()} nodes, "
                f"{self.graph.graph.number_of_edges()} edges)"
            )
        except Exception as e:
            logger.error(f"Builder.process_buffer failed: {e}")
            self._kept_items = []

        # ====== Step 2: Self-play via AdversarialOptimizer (with empty-graph guard) ======
        if self.enable_self_play and self.graph.graph.number_of_nodes() > 0:
            try:
                logger.info(
                    f"Q{self._question_idx} Self-play starting "
                    f"(mode={self.optimizer_mode}, questions={self.optimizer_fixed_count}, "
                    f"cot={self.use_cot})"
                )
                self.optimizer.step(
                    buffer_content=combined,
                    action_log=action_log if action_log else None,
                    mode=self.optimizer_mode,
                    fixed_loops=self.optimizer_fixed_count,
                    use_cot=self.use_cot,
                )
                logger.info(
                    f"Q{self._question_idx} Self-play done  "
                    f"(graph: {self.graph.graph.number_of_nodes()} nodes, "
                    f"{self.graph.graph.number_of_edges()} edges)"
                )
            except Exception as e:
                logger.error(f"Optimizer.step failed: {e}")

        self._experience_buffer = []

    def finalize(self) -> None:
        """Flush any remaining buffered experiences (called after evaluation loop ends)."""
        if self._experience_buffer:
            logger.info(f"Final flush: {len(self._experience_buffer)} remaining experiences")
            self._flush_buffer()

    def reset(self) -> None:
        self.graph.graph = nx.MultiDiGraph()
        self.graph.save()
        self._question_idx = 0
        self._experience_buffer = []
        self._kept_items = []
        # Reset evolved guidelines so agents start fresh
        self.builder.operator_guidelines = {}
        self.answerer.operator_guidelines = {}
        self.questioner.operator_guidelines = {"GENERATE": []}
        logger.info("Memory graph, buffer, and agent guidelines reset.")

    @property
    def name(self) -> str:
        return f"Amadeus({self.model_name})"

    # ---- internal helpers ----

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
        # Last resort: look for standalone letter
        m = re.search(r"\b([A-D])\b", text)
        if m:
            return m.group(1)
        return "A"
