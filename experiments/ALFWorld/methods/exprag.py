"""ExpRAG memory method for ALFWorld streaming evaluation.

Experience Retrieval-Augmented Generation:
- Stores (Goal, Trajectory, Correctness) per episode with embedding
- At each step, retrieves top-k most similar past experiences via cosine similarity
- Aligned with Evo-Memory paper Section 3.2
"""

import os
import logging
from typing import List, Tuple, Optional

import numpy as np

from .base import MemoryModule

logger = logging.getLogger(__name__)

_AMADEUS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_EMBEDDER = os.path.join(_AMADEUS_ROOT, "models", "all-MiniLM-L6-v2")


class _Embedder:
    """Lightweight sentence embedder using mean pooling."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self._torch = torch

    def embed(self, text: str) -> Optional[np.ndarray]:
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.device)
            with self._torch.no_grad():
                outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(
                outputs.last_hidden_state.size()
            ).float()
            pooled = self._torch.sum(outputs.last_hidden_state * mask, 1) / \
                     self._torch.clamp(mask.sum(1), min=1e-9)
            vec = pooled[0].cpu().numpy()
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None


class ExpRAGMemory(MemoryModule):
    """
    ExpRAG: retrieve top-k most similar past episodes via embedding cosine similarity.

    Memory entry format (paper Section 3.2):
      Goal: <task_description>
      Trajectory: Action: X -> Action: Y -> ...
      Correctness: success/failure
    """

    def __init__(
        self,
        top_k: int = 4,
        embedder_model_path: str = None,
        **kwargs,
    ):
        self.top_k = top_k
        embedder_path = embedder_model_path or _DEFAULT_EMBEDDER
        self.embedder = _Embedder(embedder_path)
        # List of (embedding_vector, experience_text)
        self._memory: List[Tuple[np.ndarray, str]] = []

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument(
            "--exprag_top_k", type=int, default=4,
            help="ExpRAG: number of experiences to retrieve (default: 4)"
        )

    def search(self, query: str) -> str:
        """Retrieve top-k most similar experiences via cosine similarity."""
        if not self._memory:
            return ""

        query_vec = self.embedder.embed(query)
        if query_vec is None:
            return ""

        scores = [(float(np.dot(query_vec, vec)), text) for vec, text in self._memory]
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:self.top_k]

        parts = []
        for i, (_, text) in enumerate(top, 1):
            parts.append(f"[Experience #{i}]\n{text}")
        return "\n\n".join(parts)

    def evolve(
        self,
        task_type: str,
        task_description: str,
        trajectory: List[Tuple[str, str]],
        success: bool,
        episode_idx: int,
    ) -> None:
        """Store episode with embedding keyed on task_description."""
        traj_parts = [f"Action: {action}" for _, action in trajectory] if trajectory else []

        correctness = "success" if success else "failure"
        experience = (
            f"Goal: {task_description}\n"
            f"Trajectory: {' -> '.join(traj_parts)}\n"
            f"Correctness: {correctness}"
        )

        vec = self.embedder.embed(task_description)
        if vec is not None:
            self._memory.append((vec, experience))
            logger.debug(f"Stored experience #{len(self._memory)}")
