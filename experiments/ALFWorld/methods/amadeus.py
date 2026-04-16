"""
Amadeus MemoryGraph with full Builder + self-play for ALFWorld.

Pipeline per episode:
  1. search(): semantic search on KG for similar prior task experiences
  2. evolve(): Builder parses trajectory → KG ops; Optimizer runs self-play
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel

from .base import MemoryModule

logger = logging.getLogger("ALFWorldStreaming")

_AMADEUS_PATH_ADDED = False


def _ensure_amadeus_path():
    global _AMADEUS_PATH_ADDED
    if not _AMADEUS_PATH_ADDED:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent  # repo root
        sys.path.insert(0, str(base_dir))
        _AMADEUS_PATH_ADDED = True


class HuggingFaceEmbedder:
    """Simple mean-pooling sentence embedder."""

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
    Full Amadeus pipeline: Builder → Questioner → Answerer → Optimizer self-play.

    Required kwargs: output_dir, embedding_model, model_name, api_base, api_key
    Optional: top_k, selfplay_mode, selfplay_rounds, use_cot, no_selfplay
    """

    def __init__(self, **kwargs):
        output_dir = kwargs.get("output_dir")
        embedding_model = kwargs.get("embedding_model")
        model_name = kwargs.get("model_name")
        api_base = kwargs.get("api_base")
        api_key = kwargs.get("api_key")
        self.top_k = kwargs.get("top_k", 4)
        self.selfplay_mode = kwargs.get("selfplay_mode", "fixed")
        self.selfplay_rounds = kwargs.get("selfplay_rounds", 3)
        self.use_cot = kwargs.get("use_cot", False)
        self.no_selfplay = kwargs.get("no_selfplay", False)

        if not output_dir or not embedding_model:
            raise ValueError(
                "AmadeusMemory requires 'output_dir' and 'embedding_model' kwargs. "
                "Pass --output_dir and --embedding_model on the command line."
            )

        _ensure_amadeus_path()
        from amadeus_collab.core.graph import MemoryGraph
        from amadeus_collab.agents.builder import BuilderAgent
        from amadeus_collab.agents.answerer import AnswererAgent
        from amadeus_collab.agents.questioner import QuestionerAgent
        from amadeus_collab.engine.optimizer import AdversarialOptimizer

        logger.info(f"Loading embedding model: {embedding_model}")
        embedder = HuggingFaceEmbedder(embedding_model)
        graph_path = str(Path(output_dir) / "memory_graph.json")
        self.graph = MemoryGraph(graph_path, embedder=embedder)

        self.builder = BuilderAgent(self.graph, model_name=model_name)
        self.answerer = AnswererAgent(
            self.graph, model_name=model_name, api_base=api_base, api_key=api_key
        )
        self.questioner = QuestionerAgent(
            model_name=model_name, api_base=api_base, api_key=api_key
        )
        self.optimizer = AdversarialOptimizer(
            self.questioner, self.builder, self.answerer,
            model_name=model_name, api_base=api_base, api_key=api_key,
        )

        logger.info(
            f"Amadeus initialized: selfplay={'OFF' if self.no_selfplay else self.selfplay_mode}, "
            f"rounds={self.selfplay_rounds}, cot={self.use_cot}"
        )

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

        experiences = []
        for idx, node_name in enumerate(hits):
            if not self.graph.graph.has_node(node_name):
                continue
            desc = self.graph.graph.nodes[node_name].get("description", "")
            if desc:
                experiences.append(f"[Experience #{idx+1}]\n{desc}")

        return "\n\n".join(experiences)

    def evolve(
        self,
        task_type: str,
        task_description: str,
        trajectory: List[Tuple[str, str]],
        success: bool,
        episode_idx: int,
    ) -> None:
        buffer_content = self._format_trajectory(
            task_type, task_description, trajectory, success, episode_idx
        )

        logger.info(f"  [Amadeus] Builder processing episode {episode_idx}...")
        try:
            kept_items, action_log = self.builder.process_buffer(buffer_content)
            logger.info(f"  [Amadeus] Builder completed: {len(action_log)} operations")
        except Exception as e:
            logger.error(f"  [Amadeus] Builder failed: {e}, falling back to direct add")
            self._fallback_add(task_type, task_description, trajectory, success, episode_idx)
            return

        if not self.no_selfplay and self.graph.graph.number_of_nodes() > 0:
            logger.info(f"  [Amadeus] Starting self-play (mode={self.selfplay_mode})...")
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

    def _format_trajectory(self, task_type, task_description, trajectory, success, episode_idx):
        lines = [
            f"--- Episode {episode_idx}: {task_type} ---",
            f"Goal: {task_description}",
            f"Result: {'SUCCESS' if success else 'FAILED'}",
            "",
        ]
        for obs, action in trajectory[-20:]:
            lines.append(f"Agent: {action}")
            lines.append(f"Environment: {obs[:200]}")

        text = "\n".join(lines)
        if len(text) > 3000:
            text = text[:3000] + "\n... (truncated)"
        return text

    def _fallback_add(self, task_type, task_description, trajectory, success, episode_idx):
        traj_text = "\n".join(
            f"Action: {act} | Obs: {obs[:150]}" for obs, act in trajectory[-15:]
        )
        result_str = "SUCCESS" if success else "FAILED"
        node_name = f"Task_{episode_idx}_{task_type}"
        node_desc = f"Goal: {task_description}\nResult: {result_str}\nTrajectory:\n{traj_text}"
        self.graph.add_node(node_name, "TaskExperience", node_desc)
        self.graph.save()

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument(
            "--embedding_model", type=str,
            default=str(Path(__file__).resolve().parent.parent.parent.parent / "models" / "all-MiniLM-L6-v2"),
            help="Path to sentence embedding model (for Amadeus method)",
        )
        parser.add_argument("--top_k", type=int, default=4,
                            help="Number of experiences to retrieve")
        parser.add_argument("--selfplay_mode", type=str, default="fixed",
                            choices=["adaptive", "fixed"])
        parser.add_argument("--selfplay_rounds", type=int, default=3)
        parser.add_argument("--use_cot", action="store_true", default=False)
        parser.add_argument("--no_selfplay", action="store_true", default=False,
                            help="Disable self-play, only use Builder")
