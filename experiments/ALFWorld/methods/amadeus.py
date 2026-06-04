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
        self.schema_exploration_buffer_limit = kwargs.get("schema_exploration_buffer_limit", 3)
        self.no_schema_emergence = kwargs.get("no_schema_emergence", False)
        self.enable_schema_replay = kwargs.get("enable_schema_replay", False)

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
        from amadeus_collab.core.schema import SchemaState
        from amadeus_collab.engine.optimizer import AdversarialOptimizer

        logger.info(f"Loading embedding model: {embedding_model}")
        embedder = HuggingFaceEmbedder(embedding_model)
        output_path = Path(output_dir)
        graph_path = str(output_path / "memory_graph.json")
        self.schema_path = str(output_path / "schema_state.json")
        self.graph = MemoryGraph(graph_path, embedder=embedder)
        self.schema_state = SchemaState.load(self.schema_path)
        self.flushed_buffers = []

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
            f"rounds={self.selfplay_rounds}, cot={self.use_cot}, "
            f"schema_emergence={'OFF' if self.no_schema_emergence else 'ON'}, "
            f"schema_limit={self.schema_exploration_buffer_limit}, replay={self.enable_schema_replay}"
        )

    def _apply_schema_proposals(
        self,
        proposals,
        *,
        buffers_seen: int,
        selected_n: int,
        source: str = "unknown",
    ) -> bool:
        from amadeus_collab.core.schema import SchemaProposalBundle, SchemaReviewResult

        if not isinstance(proposals, SchemaProposalBundle):
            logger.warning("  [Amadeus] Invalid schema proposals from %s; skipping apply.", source)
            return False

        if not (proposals.node_types or proposals.edge_types or proposals.rules):
            logger.info("  [Amadeus] No schema proposals to apply from %s.", source)
            self.schema_state.buffers_seen = max(self.schema_state.buffers_seen, buffers_seen)
            self.schema_state.n_selected = max(self.schema_state.n_selected, selected_n)
            self.schema_state.save(self.schema_path)
            return False

        review_raw = self.builder.review_schema_proposals(proposals, self.schema_state)
        review_results = []
        for item in review_raw:
            try:
                review_results.append(SchemaReviewResult(**item))
            except Exception as e:
                logger.warning("  [Amadeus] Invalid schema review result skipped: %s", e)

        old_version = self.schema_state.version
        changed = self.schema_state.apply_reviewed_proposals(
            proposals,
            review_results,
            buffers_seen=buffers_seen,
            selected_n=selected_n,
        )
        self.schema_state.save(self.schema_path)

        logger.info(
            "  [Amadeus] Schema after %s | changed=%s | version=%s->%s | n_selected=%s | "
            "active_node_types=%s | active_edge_types=%s | active_rules=%s",
            source,
            changed,
            old_version,
            self.schema_state.version,
            self.schema_state.n_selected,
            [item.name for item in self.schema_state.node_types if item.status == "active"],
            [item.name for item in self.schema_state.edge_types if item.status == "active"],
            [item.name for item in self.schema_state.rules if item.status == "active"],
        )
        return changed

    def _maybe_emerge_schema(self) -> bool:
        max_n = min(self.schema_exploration_buffer_limit, len(self.flushed_buffers))
        if max_n <= 0:
            return False

        logger.info("  [Amadeus] Starting schema emergence over first up to %s buffers.", max_n)
        last_output = None
        selected_n = 1
        for n in range(1, max_n + 1):
            selected_n = n
            logger.info("  [Amadeus] Emergence attempt with n=%s", n)
            last_output = self.builder.emerge_schema(self.flushed_buffers[:n], self.schema_state)
            if last_output.stable:
                logger.info("  [Amadeus] Emergence declared stable at n=%s", n)
                break

        if last_output is None:
            logger.warning("  [Amadeus] Emergence returned no output.")
            return False

        return self._apply_schema_proposals(
            last_output.proposals,
            buffers_seen=len(self.flushed_buffers),
            selected_n=selected_n,
            source="emergence",
        )

    def _replay_graph(self) -> None:
        logger.info(
            "  [Amadeus] Replay start | buffers=%s | schema_version=%s",
            len(self.flushed_buffers),
            self.schema_state.version,
        )
        self.graph.graph.clear()
        carry_items = []
        total_ops = 0
        for idx, buffer_content in enumerate(self.flushed_buffers):
            replay_input = buffer_content
            if carry_items:
                replay_input = "\n".join(carry_items + [buffer_content])
            carry_items, action_log, _ = self.builder.process_buffer(
                replay_input,
                schema_state=self.schema_state,
                buffer_index=idx,
                replay_mode=True,
            )
            total_ops += len(action_log)
        self.schema_state.mark_replayed_until(len(self.flushed_buffers) - 1)
        self.schema_state.save(self.schema_path)
        self.graph.save()
        logger.info(
            "  [Amadeus] Replay complete | total_ops=%s | graph_nodes=%s | graph_edges=%s",
            total_ops,
            self.graph.graph.number_of_nodes(),
            self.graph.graph.number_of_edges(),
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

        self.flushed_buffers.append(buffer_content)
        buffer_index = len(self.flushed_buffers) - 1

        if (
            not self.no_schema_emergence
            and len(self.flushed_buffers) <= self.schema_exploration_buffer_limit
        ):
            try:
                self._maybe_emerge_schema()
            except Exception as e:
                logger.error(f"  [Amadeus] Schema emergence failed: {e}")

        logger.info(f"  [Amadeus] Builder processing episode {episode_idx}...")
        try:
            kept_items, action_log, schema_proposals = self.builder.process_buffer(
                buffer_content,
                schema_state=self.schema_state,
                buffer_index=buffer_index,
                replay_mode=False,
            )
            logger.info(f"  [Amadeus] Builder completed: {len(action_log)} operations")
        except Exception as e:
            logger.error(f"  [Amadeus] Builder failed: {e}, falling back to direct add")
            self._fallback_add(task_type, task_description, trajectory, success, episode_idx)
            return

        if not self.no_schema_emergence:
            try:
                selected_n = min(self.schema_exploration_buffer_limit, len(self.flushed_buffers))
                changed = self._apply_schema_proposals(
                    schema_proposals,
                    buffers_seen=len(self.flushed_buffers),
                    selected_n=selected_n,
                    source="builder",
                )
                if changed and self.enable_schema_replay and self.schema_state.needs_replay():
                    self._replay_graph()
            except Exception as e:
                logger.error(f"  [Amadeus] Schema proposal handling failed: {e}")

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
        parser.add_argument("--schema_exploration_buffer_limit", type=int, default=3,
                            help="Number of early episode buffers used for schema emergence")
        parser.add_argument("--no_schema_emergence", action="store_true", default=False,
                            help="Disable schema emergence and only use broad default graph types")
        parser.add_argument("--enable_schema_replay", action="store_true", default=False,
                            help="Replay prior ALFWorld buffers after accepted schema changes")
