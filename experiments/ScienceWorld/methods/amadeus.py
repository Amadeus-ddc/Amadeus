"""
Amadeus MemoryGraph with full self-play (adversarial optimization).

This is the complete Amadeus pipeline:
  1. Builder agent parses episode trajectory into structured KG operations
  2. Questioner generates adversarial questions from the trajectory
  3. Answerer (Graph RAG) tries to answer from the KG
  4. Meta-Critic judges, assigns blame, generates textual gradients + graph patches
  5. Loop until convergence (2 consecutive all-pass rounds)
"""

import sys
import logging
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from .base import MemoryModule

logger = logging.getLogger("SciWorldStreaming")

# Lazy-import path setup (done once)
_AMADEUS_PATH_ADDED = False

def _ensure_amadeus_path():
    global _AMADEUS_PATH_ADDED
    if not _AMADEUS_PATH_ADDED:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent  # amadeus/
        sys.path.insert(0, str(base_dir.parent))  # Amadeus/
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

    Required kwargs:
        output_dir (str): Directory to store the memory graph
        embedding_model (str): Path to sentence embedding model
        model_name (str): LLM model name for the agents
        api_base (str): vLLM API base URL
        api_key (str): API key

    Optional kwargs:
        top_k (int): Number of experiences to retrieve (default: 4)
        selfplay_mode (str): "adaptive" or "fixed" (default: "fixed")
        selfplay_rounds (int): Number of Q&A rounds in fixed mode (default: 3)
        use_cot (bool): Use CoT evaluation in optimizer (default: False)
        no_selfplay (bool): Disable self-play, only use Builder (default: False)
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
            raise ValueError(
                "AmadeusMemory requires 'output_dir' and 'embedding_model' kwargs. "
                "Pass --output_dir and --embedding_model on the command line."
            )

        _ensure_amadeus_path()
        from amadeus_tzx.code.core.graph import MemoryGraph
        from amadeus_tzx.code.agents.builder import BuilderAgent
        from amadeus_tzx.code.agents.answerer import AnswererAgent
        from amadeus_tzx.code.agents.questioner import QuestionerAgent
        from amadeus_tzx.code.engine.optimizer import AdversarialOptimizer

        # Init embedder and graph
        logger.info(f"Loading embedding model: {embedding_model}")
        embedder = HuggingFaceEmbedder(embedding_model)
        graph_path = str(Path(output_dir) / "memory_graph.json")
        self.graph = MemoryGraph(graph_path, embedder=embedder)

        # Init the three agents
        self.builder = BuilderAgent(self.graph, model_name=model_name)
        self.answerer = AnswererAgent(self.graph, model_name=model_name,
                                      api_base=api_base, api_key=api_key)
        self.questioner = QuestionerAgent(model_name=model_name,
                                          api_base=api_base, api_key=api_key)

        # Init optimizer (orchestrates self-play)
        self.optimizer = AdversarialOptimizer(
            self.questioner, self.builder, self.answerer,
            model_name=model_name, api_base=api_base, api_key=api_key,
        )

        logger.info(
            f"Amadeus initialized: selfplay={'OFF' if self.no_selfplay else self.selfplay_mode}, "
            f"rounds={self.selfplay_rounds}, cot={self.use_cot}"
        )

    def search(self, query: str) -> str:
        """Retrieve relevant past experiences via Answerer's graph RAG."""
        if self.graph.graph.number_of_nodes() == 0:
            return ""

        # Use semantic search for retrieval (same as Answerer's entry point)
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
        task_name: str,
        goal: str,
        trajectory: List[dict],
        success: bool,
        progress: float,
        episode_idx: int,
    ) -> None:
        """
        Full Amadeus evolve pipeline:
          1. Format trajectory as a text buffer
          2. Builder processes the buffer → structured KG operations
          3. Optimizer runs self-play (Questioner attacks, Answerer defends)
        """
        # Format trajectory as a "conversation buffer" for the Builder
        buffer_content = self._format_trajectory_as_buffer(
            task_name, goal, trajectory, success, progress, episode_idx
        )

        # Step 1: Builder processes the buffer → graph operations
        logger.info(f"  [Amadeus] Builder processing episode {episode_idx}...")
        try:
            kept_items, action_log, _ = self.builder.process_buffer(buffer_content)
            logger.info(f"  [Amadeus] Builder completed: {len(action_log)} operations")
        except Exception as e:
            logger.error(f"  [Amadeus] Builder failed: {e}, falling back to direct add")
            self._fallback_add(task_name, goal, trajectory, success, progress, episode_idx)
            return

        # Step 2: Self-play optimization
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
        elif self.no_selfplay:
            logger.info("  [Amadeus] Self-play disabled, skipping")

    def _format_trajectory_as_buffer(
        self, task_name, goal, trajectory, success, progress, episode_idx
    ) -> str:
        """Format the ScienceWorld episode as a text buffer for the Builder."""
        lines = []
        lines.append(f"--- Episode {episode_idx}: {task_name} ---")
        lines.append(f"Goal: {goal}")
        result_str = "SUCCESS" if success else f"PARTIAL (progress={progress:.2f})"
        lines.append(f"Result: {result_str}")
        lines.append("")

        for item in trajectory:
            if "Goal" in item:
                lines.append(f"Goal: {item['Goal']}")
            elif "Action" in item:
                lines.append(f"Agent: {item['Action']}")
            elif "Observation" in item:
                obs = item["Observation"][:200]
                lines.append(f"Environment: {obs}")
            elif "Progress Rate" in item:
                lines.append(f"[Progress: {item['Progress Rate']:.2f}]")

        # Truncate to avoid overly long buffers
        text = "\n".join(lines)
        if len(text) > 3000:
            text = text[:3000] + "\n... (truncated)"
        return text

    def _fallback_add(self, task_name, goal, trajectory, success, progress, episode_idx):
        """Fallback: directly add a node if Builder fails."""
        traj_summary = []
        for item in trajectory:
            if "Action" in item:
                traj_summary.append(f"Action: {item['Action']}")
            elif "Observation" in item:
                traj_summary.append(f"Obs: {item['Observation'][:150]}")

        traj_text = "\n".join(traj_summary[-20:])
        result_str = "SUCCESS" if success else f"PARTIAL (progress={progress:.2f})"
        node_name = f"Task_{episode_idx}_{task_name}"
        node_desc = f"Goal: {goal}\nResult: {result_str}\nTrajectory:\n{traj_text}"
        self.graph.add_node(node_name, "TaskExperience", node_desc)
        self.graph.save()

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument(
            "--embedding_model", type=str,
            default=str(Path(__file__).resolve().parent.parent.parent.parent / "models" / "all-MiniLM-L6-v2"),
            help="Path to sentence embedding model (for Amadeus method)",
        )
        parser.add_argument(
            "--top_k", type=int, default=5,
            help="Number of experiences to retrieve (for Amadeus method)",
        )
        parser.add_argument(
            "--selfplay_mode", type=str, default="fixed", choices=["adaptive", "fixed"],
            help="Self-play mode: 'adaptive' (converge-based) or 'fixed' (N rounds)",
        )
        parser.add_argument(
            "--selfplay_rounds", type=int, default=3,
            help="Number of Q&A rounds per episode in fixed self-play mode",
        )
        parser.add_argument(
            "--use_cot", action="store_true", default=False,
            help="Use Chain-of-Thought evaluation in self-play optimizer",
        )
        parser.add_argument(
            "--no_selfplay", action="store_true", default=False,
            help="Disable self-play, only use Builder for memory (ablation)",
        )
