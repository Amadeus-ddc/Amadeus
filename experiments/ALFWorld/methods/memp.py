"""MemP memory method for ALFWorld offline evaluation.

Offline (cold-start) workflow:
  1. Cold-start: load training trajectories from alfworld_format_traj.json
  2. For each trajectory, call LLM to extract a workflow paragraph
  3. Store workflows in FAISS vector store (local all-MiniLM-L6-v2 embeddings)
  4. At test time: retrieve top-k workflows by query similarity, inject into prompt

This is a read-only memory at test time — no evolve() updates the store.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from openai import OpenAI

from .base import MemoryModule

logger = logging.getLogger(__name__)

_AMADEUS_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_DEFAULT_EMBEDDER = os.path.join(_AMADEUS_ROOT, "models", "all-MiniLM-L6-v2")
_DEFAULT_TRAJ_FILE = os.path.join(
    os.path.dirname(_AMADEUS_ROOT), "MemP", "ProcedureMem", "Alfworld", "alfworld_format_traj.json"
)
_DEFAULT_STORE_DIR = os.path.join(
    _AMADEUS_ROOT, "amadeus", "experiments", "ALFWorld", "logs", "memp_store"
)


# ---------------------------------------------------------------------------
# Local embedder (same as exprag.py)
# ---------------------------------------------------------------------------

class _Embedder:
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

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[Optional[np.ndarray]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                import torch
                inputs = self.tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(
                    outputs.last_hidden_state.size()
                ).float()
                pooled = torch.sum(outputs.last_hidden_state * mask, 1) / \
                         torch.clamp(mask.sum(1), min=1e-9)
                vecs = pooled.cpu().numpy()
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.where(norms > 0, norms, 1.0)
                results.extend(list(vecs))
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}")
                results.extend([None] * len(batch))
        return results


# ---------------------------------------------------------------------------
# Workflow extraction prompt (ALFWorld-specific)
# ---------------------------------------------------------------------------

def _workflow_prompt(query: str, trajectory: list) -> List[dict]:
    """Build prompt to extract a workflow paragraph from an ALFWorld trajectory."""
    # Convert trajectory list [{from, value}] to readable text
    traj_lines = []
    for turn in trajectory:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human":
            traj_lines.append(f"Observation: {value}")
        elif role == "gpt":
            traj_lines.append(f"Agent: {value}")
    traj_text = "\n".join(traj_lines)

    system = "You are a helpful assistant that summarizes agent trajectories into reusable workflows."
    user = f"""You are given an ALFWorld task goal and the agent's trajectory (observations and actions).
Your task is to write a concise workflow paragraph describing the key steps needed to solve this type of task.
The workflow should be general enough to help solve similar tasks in the future.
Focus on the sequence of actions (e.g., go to location, pick up object, heat/cool/clean object, place in receptacle).
Write as a natural paragraph, not a bullet list.

-----EXAMPLE WORKFLOW-----
To solve this task, first locate the target object by examining likely receptacles in the room. Pick up the object, then find the required appliance or receptacle. Interact with the appliance if needed (e.g., heat in microwave, cool in fridge, clean in sink), then place the object in the target receptacle.
-----EXAMPLE END-----

Task Goal: {query}

Trajectory:
{traj_text}

Output the workflow paragraph only, without any explanation:"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# MemP Memory Module
# ---------------------------------------------------------------------------

class MempMemory(MemoryModule):
    """
    MemP offline memory: cold-start from training trajectories, retrieve at test time.

    Args:
        api_base: vLLM server base URL (e.g. http://localhost:8003/v1)
        api_key: API key (default "EMPTY")
        model_name: served model name
        top_k: number of workflows to retrieve
        store_dir: directory to persist FAISS index and documents
        traj_file: path to alfworld_format_traj.json
        embedder_model_path: path to local sentence embedding model
        max_workers: parallel workers for cold-start LLM calls
        force_rebuild: if True, rebuild store even if it exists
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8003/v1",
        api_key: str = "EMPTY",
        model_name: str = "qwen2.5-7b-instruct",
        top_k: int = 3,
        store_dir: str = None,
        traj_file: str = None,
        embedder_model_path: str = None,
        max_workers: int = 16,
        force_rebuild: bool = False,
        **kwargs,
    ):
        self.top_k = top_k
        self.max_workers = max_workers
        self.store_dir = Path(store_dir or _DEFAULT_STORE_DIR)
        self.traj_file = traj_file or _DEFAULT_TRAJ_FILE
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name

        logger.info(f"Loading embedder from {embedder_model_path or _DEFAULT_EMBEDDER}")
        self.embedder = _Embedder(embedder_model_path or _DEFAULT_EMBEDDER)

        # In-memory store: list of (embedding, query, workflow)
        self._docs: List[dict] = []          # {"query": str, "workflow": str}
        self._embeddings: List[np.ndarray] = []

        self._load_or_build(force_rebuild)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def _docs_path(self):
        return self.store_dir / "documents.json"

    @property
    def _emb_path(self):
        return self.store_dir / "embeddings.pkl"

    def _save(self):
        with open(self._docs_path, "w") as f:
            json.dump(self._docs, f, indent=2, ensure_ascii=False)
        with open(self._emb_path, "wb") as f:
            pickle.dump(self._embeddings, f)
        logger.info(f"Saved {len(self._docs)} documents to {self.store_dir}")

    def _load(self) -> bool:
        if self._docs_path.exists() and self._emb_path.exists():
            with open(self._docs_path) as f:
                self._docs = json.load(f)
            with open(self._emb_path, "rb") as f:
                self._embeddings = pickle.load(f)
            logger.info(f"Loaded {len(self._docs)} documents from {self.store_dir}")
            return True
        return False

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------

    def _extract_workflow(self, item: dict) -> Optional[dict]:
        """Call LLM to extract workflow from one trajectory item."""
        query = item.get("query", "").split("\n\n")[0].strip()
        trajectory = item.get("trajectory", [])
        if not query or not trajectory:
            return None
        # Skip if already in store
        if any(d["query"] == query for d in self._docs):
            return None
        try:
            messages = _workflow_prompt(query, trajectory)
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=512,
            )
            workflow = resp.choices[0].message.content.strip()
            return {"query": query, "workflow": workflow}
        except Exception as e:
            logger.warning(f"Workflow extraction failed for '{query[:50]}': {e}")
            return None

    def _load_or_build(self, force_rebuild: bool):
        if not force_rebuild and self._load():
            return
        logger.info(f"Cold-starting MemP from {self.traj_file} ...")
        with open(self.traj_file) as f:
            traj_data = json.load(f)
        logger.info(f"Loaded {len(traj_data)} training trajectories")

        new_docs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._extract_workflow, item): item for item in traj_data}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting workflows"):
                result = future.result()
                if result:
                    new_docs.append(result)

        self._docs.extend(new_docs)
        logger.info(f"Extracted {len(new_docs)} new workflows, total={len(self._docs)}")

        # Embed all queries
        logger.info("Embedding all queries ...")
        queries = [d["query"] for d in self._docs]
        self._embeddings = self.embedder.embed_batch(queries)
        # Filter out failed embeddings
        valid = [(d, e) for d, e in zip(self._docs, self._embeddings) if e is not None]
        self._docs = [d for d, _ in valid]
        self._embeddings = [e for _, e in valid]

        self._save()

    # ------------------------------------------------------------------
    # MemoryModule interface
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        if not self._docs:
            return ""
        query_vec = self.embedder.embed(query)
        if query_vec is None:
            return ""

        emb_matrix = np.stack(self._embeddings)  # (N, D)
        scores = emb_matrix @ query_vec           # cosine similarity (normalized)
        top_idx = np.argsort(scores)[::-1][:self.top_k]

        parts = []
        for rank, idx in enumerate(top_idx, 1):
            doc = self._docs[idx]
            parts.append(
                f"[Workflow #{rank}]\n"
                f"Similar Task: {doc['query']}\n"
                f"Workflow: {doc['workflow']}"
            )
        return "\n\n".join(parts)

    def evolve(
        self,
        task_type: str,
        task_description: str,
        trajectory: List[Tuple[str, str]],
        success: bool,
        episode_idx: int,
    ) -> None:
        # MemP offline mode: no online updates
        pass

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument(
            "--memp_top_k", type=int, default=3,
            help="MemP: number of workflows to retrieve (default: 3)"
        )
        parser.add_argument(
            "--memp_store_dir", type=str, default=None,
            help="MemP: directory to persist workflow store"
        )
        parser.add_argument(
            "--memp_force_rebuild", action="store_true",
            help="MemP: force rebuild workflow store even if it exists"
        )
        parser.add_argument(
            "--memp_max_workers", type=int, default=16,
            help="MemP: parallel workers for cold-start LLM calls (default: 16)"
        )
