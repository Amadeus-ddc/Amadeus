"""
Amadeus MemoryGraph with full Builder + self-play for MATH.

Pipeline per problem:
  1. search(): semantic search on KG for similar prior solutions
  2. evolve(): Builder parses solution → KG ops; Optimizer runs self-play
"""

import sys
import logging
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from .base import MemoryModule

logger = logging.getLogger("MATH500")

_AMADEUS_PATH_ADDED = False

def _ensure_amadeus_path():
    global _AMADEUS_PATH_ADDED
    if not _AMADEUS_PATH_ADDED:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent  # amadeus/
        sys.path.insert(0, str(base_dir.parent))  # Amadeus/
        _AMADEUS_PATH_ADDED = True


class HuggingFaceEmbedder:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def embed(self, text):
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            mask = inputs["attention_mask"]
            tok_emb = outputs.last_hidden_state
            mask_ex = mask.unsqueeze(-1).expand(tok_emb.size()).float()
            emb = torch.sum(tok_emb * mask_ex, 1) / torch.clamp(mask_ex.sum(1), min=1e-9)
            return emb[0].cpu().numpy()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None


class AmadeusMemory(MemoryModule):
    """
    Full Amadeus: Builder + Optimizer self-play for math solutions.

    Required kwargs: output_dir, embedding_model, model_name, api_base, api_key
    Optional: top_k, selfplay_mode, selfplay_rounds, use_cot, no_selfplay
    """

    def __init__(self, **kwargs):
        output_dir = kwargs.get("output_dir")
        embedding_model = kwargs.get("embedding_model")
        model_name = kwargs.get("model_name")
        api_base = kwargs.get("api_base")
        api_key = kwargs.get("api_key")
        self.top_k = kwargs.get("top_k", 5)
        self.selfplay_mode = kwargs.get("selfplay_mode", "fixed")
        self.selfplay_rounds = kwargs.get("selfplay_rounds", 3)
        self.use_cot = kwargs.get("use_cot", False)
        self.no_selfplay = kwargs.get("no_selfplay", False)

        if not output_dir or not embedding_model:
            raise ValueError("AmadeusMemory requires 'output_dir' and 'embedding_model'.")

        _ensure_amadeus_path()
        from amadeus_tzx.code.core.graph import MemoryGraph
        from amadeus_tzx.code.agents.builder import BuilderAgent
        from amadeus_tzx.code.agents.answerer import AnswererAgent
        from amadeus_tzx.code.agents.questioner import QuestionerAgent
        from amadeus_tzx.code.engine.optimizer import AdversarialOptimizer

        logger.info(f"Loading embedding model: {embedding_model}")
        embedder = HuggingFaceEmbedder(embedding_model)
        graph_path = str(Path(output_dir) / "memory_graph.json")
        self.graph = MemoryGraph(graph_path, embedder=embedder)

        self.builder = BuilderAgent(self.graph, model_name=model_name)
        self.answerer = AnswererAgent(self.graph, model_name=model_name,
                                      api_base=api_base, api_key=api_key)
        self.questioner = QuestionerAgent(model_name=model_name,
                                          api_base=api_base, api_key=api_key)
        self.optimizer = AdversarialOptimizer(
            self.questioner, self.builder, self.answerer,
            model_name=model_name, api_base=api_base, api_key=api_key,
        )

        logger.info(f"Amadeus initialized: selfplay={'OFF' if self.no_selfplay else self.selfplay_mode}")

    def search(self, query: str) -> str:
        if self.graph.graph.number_of_nodes() == 0:
            return ""

        hits = (
            self.graph.semantic_search(query, top_k=self.top_k)
            if self.graph.embedder
            else self.graph.primitive_search(query)[:self.top_k]
        )
        if not hits:
            return ""

        parts = []
        for idx, node_name in enumerate(hits):
            if not self.graph.graph.has_node(node_name):
                continue
            desc = self.graph.graph.nodes[node_name].get("description", "")
            if desc:
                parts.append(f"[Prior Solution #{idx+1}]\n{desc}")
            for _, target, data in self.graph.graph.out_edges(node_name, data=True):
                rel = data.get("relation", "")
                if rel and self.graph.graph.has_node(target):
                    t_desc = self.graph.graph.nodes[target].get("description", "")[:200]
                    parts.append(f"  -> {rel}: {t_desc}")

        return "\n\n".join(parts[:15])

    def evolve(self, problem, reasoning, answer, is_correct, problem_idx):
        if not is_correct:
            return  # Only memorize correct solutions

        # Format as buffer for Builder
        reasoning_trunc = reasoning[:500] if len(reasoning) > 500 else reasoning
        buffer_content = (
            f"--- Math Problem {problem_idx} ---\n"
            f"Problem: {problem[:300]}\n"
            f"Answer: {answer}\n"
            f"Reasoning: {reasoning_trunc}"
        )

        # Builder parses into structured KG
        logger.info(f"  [Amadeus] Builder processing problem {problem_idx}...")
        try:
            kept_items, action_log = self.builder.process_buffer(buffer_content)
            logger.info(f"  [Amadeus] Builder completed: {len(action_log)} operations")
        except Exception as e:
            logger.error(f"  [Amadeus] Builder failed: {e}, falling back to direct add")
            self._fallback_add(problem, reasoning, answer, problem_idx)
            return

        # Self-play
        if not self.no_selfplay and self.graph.graph.number_of_nodes() > 0:
            logger.info(f"  [Amadeus] Starting self-play...")
            try:
                self.optimizer.step(
                    buffer_content,
                    action_log=action_log,
                    mode=self.selfplay_mode,
                    fixed_loops=self.selfplay_rounds,
                    use_cot=self.use_cot,
                )
                logger.info("  [Amadeus] Self-play completed")
            except Exception as e:
                logger.error(f"  [Amadeus] Self-play failed: {e}")

    def _fallback_add(self, problem, reasoning, answer, problem_idx):
        reasoning_summary = reasoning[:400] if len(reasoning) > 400 else reasoning
        node_name = f"Problem_{problem_idx}"
        node_desc = f"Q: {problem[:200]} | A: {answer} | Reasoning: {reasoning_summary}"
        self.graph.add_node(node_name, "MathProblem", node_desc)
        self.graph.save()

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument(
            "--embedding_model", type=str,
            default=str(Path(__file__).resolve().parent.parent.parent.parent / "models" / "all-MiniLM-L6-v2"),
            help="Path to sentence embedding model",
        )
        parser.add_argument("--top_k", type=int, default=5,
                            help="Number of prior solutions to retrieve")
        parser.add_argument("--selfplay_mode", type=str, default="fixed",
                            choices=["adaptive", "fixed"])
        parser.add_argument("--selfplay_rounds", type=int, default=3)
        parser.add_argument("--use_cot", action="store_true", default=False)
        parser.add_argument("--no_selfplay", action="store_true", default=False,
                            help="Disable self-play, only use Builder")
