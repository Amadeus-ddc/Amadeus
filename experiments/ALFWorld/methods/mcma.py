"""MCMA memory method for ALFWorld evaluation.

Reproduces the core MCMA retrieval pipeline:
  1. Build mem_forest from training trajectories (MemP alfworld_format_traj.json)
     - Each entry: {canonical_goal: [{name, goal, trajectory:[{step,action,observation}], subtasks:{}}]}
     - Skips the LLM tree-decomposition step (make_tree_vllm.py); uses flat trajectories
  2. At test time: TF-IDF search over canonical goals → retrieve top-k trajectories
  3. LLM generates structured knowledge (Tree/Chain/KV/NL) from retrieved trajectories
  4. Knowledge injected as ### REFERENCE KNOWLEDGE ### in prompt

Reference: MCMA run_alf/run_alf_mcma.py + generate_knowledge.py + search_tasks.py
"""

import os
import re
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from .base import MemoryModule

logger = logging.getLogger(__name__)

_AMADEUS_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_DEFAULT_TRAJ_FILE = os.path.join(
    os.path.dirname(_AMADEUS_ROOT), "MemP", "ProcedureMem", "Alfworld", "alfworld_format_traj.json"
)
_DEFAULT_STORE_DIR = os.path.join(
    _AMADEUS_ROOT, "experiments", "ALFWorld", "logs", "mcma_store"
)

# Knowledge generation prompt (from MCMA run_alf/prompt/sum_knowledge.txt)
_KNOWLEDGE_PROMPT_TEMPLATE = """# ROLE AND GOAL

You are an advanced AI assistant specializing in induction and reasoning from task execution trajectories. Your objective is to extract common patterns and strategies from the given trajectory and summarize this knowledge in a flexible, structured format using trees, chains, key-value pairs, and natural language.

# INSTRUCTIONS

1. Analyze the provided task trajectory (goal + action sequence).
2. Identify key patterns: task decomposition, action sequences, object/location relationships.
3. Select the optimal structure for each piece of knowledge (Tree / Chain / Key-Value / Natural Language).
4. Output a single JSON object with a "knowledge" list.

# OUTPUT FORMAT

Answer: {
  "knowledge": [
    {
      "name": "knowledge name",
      "structured_storage": {
        "type": "chain",
        "nodes": [{"step": "..."}, ...]
      }
    }
  ]
}

# TASK

Analyze the following ALFWorld trajectory and extract reusable knowledge.

Goal: [GOAL]

Trajectory:
[TRAJECTORY]

# OUTPUT"""


def _keyword_search_tfidf(query: str, data_list: List[str], top_k: int = 3) -> List[str]:
    """TF-IDF cosine similarity search — from MCMA search_tasks.py."""
    if not data_list or not query.strip():
        return []
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    full_corpus = [query] + data_list
    tfidf_matrix = vectorizer.fit_transform(full_corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    scores = cosine_sim[0]
    if scores.max() == 0:
        return []
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [data_list[i] for i in top_indices if scores[i] > 0]


def _format_trajectory(task_node: Dict) -> str:
    """Format trajectory list into readable string for knowledge prompt."""
    lines = []
    for step in task_node.get("trajectory", []):
        action = step.get("action", "")
        obs = step.get("observation", "")
        lines.append(f"Action: {action} > Observation: {obs}")
    return "\n".join(lines)


def _parse_knowledge(llm_output: str) -> str:
    """Extract knowledge text from LLM output, return as formatted string."""
    try:
        _, sep, content = llm_output.rpartition("Answer:")
        if sep:
            json_str = content.strip()
            data = json.loads(json_str)
            knowledge_list = data.get("knowledge", [])
            parts = []
            for k in knowledge_list:
                name = k.get("name", "")
                storage = k.get("structured_storage", {})
                ktype = storage.get("type", "")
                if ktype == "chain":
                    steps = [n.get("step", "") for n in storage.get("nodes", [])]
                    parts.append(f"[{name}]\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)))
                elif ktype == "key_value":
                    data_kv = storage.get("data", {})
                    lines = [f"  {k}: {v}" for k, v in data_kv.items()]
                    parts.append(f"[{name}]\n" + "\n".join(lines))
                elif ktype == "natural_language":
                    parts.append(f"[{name}]\n  {storage.get('text', '')}")
                elif ktype == "tree":
                    root = storage.get("root", {})
                    parts.append(f"[{name}]\n  Root: {root.get('name', '')}")
                else:
                    parts.append(f"[{name}]")
            return "\n\n".join(parts)
    except Exception:
        pass
    # Fallback: return raw output trimmed
    return llm_output[:500].strip()


class MCMAMemory(MemoryModule):
    """
    MCMA memory method: TF-IDF retrieval + LLM-generated structured knowledge.

    Cold-start: builds mem_forest from training trajectories (MemP format).
    At test time: TF-IDF retrieves similar tasks → LLM generates knowledge → injected into prompt.
    No online updates during test (offline mode).

    Args:
        api_base: vLLM server URL
        api_key: API key
        model_name: served model name
        top_k: number of similar tasks to retrieve
        store_dir: directory to persist mem_forest
        traj_file: training trajectory JSON file
        force_rebuild: rebuild mem_forest even if cached
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8003/v1",
        api_key: str = "EMPTY",
        model_name: str = "qwen2.5-7b-instruct",
        top_k: int = 3,
        store_dir: str = None,
        traj_file: str = None,
        force_rebuild: bool = False,
        **kwargs,
    ):
        self.top_k = kwargs.get("mcma_top_k", top_k)
        self.store_dir = Path(kwargs.get("mcma_store_dir", None) or store_dir or _DEFAULT_STORE_DIR)
        self.traj_file = traj_file or _DEFAULT_TRAJ_FILE
        force_rebuild = kwargs.get("mcma_force_rebuild", force_rebuild)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name

        # mem_forest: {canonical_goal: [task_node_dict, ...]}
        self._mem_forest: Dict[str, List[Dict]] = {}
        self._mem_keys: List[str] = []  # sorted list of canonical goals for TF-IDF

        self._load_or_build(force_rebuild)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def _forest_path(self):
        return self.store_dir / "memory_forest.json"

    def _save(self):
        with open(self._forest_path, "w") as f:
            json.dump(self._mem_forest, f, ensure_ascii=False)
        logger.info(f"Saved mem_forest with {len(self._mem_forest)} goals to {self._forest_path}")

    def _load(self) -> bool:
        if self._forest_path.exists():
            with open(self._forest_path) as f:
                self._mem_forest = json.load(f)
            self._mem_keys = list(self._mem_forest.keys())
            logger.info(f"Loaded mem_forest: {len(self._mem_forest)} goals, {sum(len(v) for v in self._mem_forest.values())} instances")
            return True
        return False

    # ------------------------------------------------------------------
    # Build mem_forest from MemP training trajectories
    # ------------------------------------------------------------------

    def _build_from_traj(self):
        """Build mem_forest from alfworld_format_traj.json.

        MemP traj format: {query, trajectory:[{from:human/gpt, value}], source}
        MCMA mem_forest format: {canonical_goal: [{name, goal, trajectory:[{step,action,observation}], subtasks:{}}]}
        """
        logger.info(f"Building mem_forest from {self.traj_file}")
        with open(self.traj_file) as f:
            traj_data = json.load(f)
        logger.info(f"Loaded {len(traj_data)} training trajectories")

        for item in traj_data:
            query = item.get("query", "").split("\n\n")[0].strip()
            if not query:
                continue
            canonical_goal = query.strip().lower().rstrip(".")

            raw_traj = item.get("trajectory", [])
            # Convert [{from, value}] → [{step, action, observation}]
            # human turns: observations; gpt turns: thought+action
            observations = [t["value"] for t in raw_traj if t["from"] == "human"]
            gpt_turns = [t["value"] for t in raw_traj if t["from"] == "gpt"]

            trajectory_steps = []
            step_idx = 1
            for obs, gpt in zip(observations[1:], gpt_turns[1:]):  # skip system prompt
                action = ""
                for line in gpt.splitlines():
                    if line.strip().lower().startswith("action:"):
                        action = line.strip()[len("action:"):].strip()
                        break
                if action:
                    trajectory_steps.append({
                        "step": step_idx,
                        "action": action,
                        "observation": obs[:300],  # truncate long obs
                    })
                    step_idx += 1

            if not trajectory_steps:
                continue

            task_node = {
                "name": query,
                "goal": query,
                "steps_count": len(trajectory_steps),
                "trajectory": trajectory_steps,
                "subtasks": {},
            }

            if canonical_goal not in self._mem_forest:
                self._mem_forest[canonical_goal] = []
            self._mem_forest[canonical_goal].append(task_node)

        self._mem_keys = list(self._mem_forest.keys())
        logger.info(f"mem_forest built: {len(self._mem_forest)} goals, {sum(len(v) for v in self._mem_forest.values())} instances")
        self._save()

    def _load_or_build(self, force_rebuild: bool):
        if not force_rebuild and self._load():
            return
        self._build_from_traj()

    # ------------------------------------------------------------------
    # Knowledge generation via LLM
    # ------------------------------------------------------------------

    def _generate_knowledge(self, goal: str, task_nodes: List[Dict]) -> str:
        """Call LLM to generate structured knowledge from retrieved trajectories."""
        # Use first instance only (like MCMA's mem_forest[key][0])
        node = task_nodes[0]
        traj_str = _format_trajectory(node)

        prompt = _KNOWLEDGE_PROMPT_TEMPLATE.replace("[GOAL]", goal).replace("[TRAJECTORY]", traj_str)

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_knowledge(raw)
        except Exception as e:
            logger.warning(f"Knowledge generation failed for '{goal[:50]}': {e}")
            # Fallback: format trajectory directly
            steps = [f"{s['step']}. {s['action']}" for s in node.get("trajectory", [])[:10]]
            return f"[Action Sequence for: {goal}]\n" + "\n".join(steps)

    # ------------------------------------------------------------------
    # MemoryModule interface
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        """TF-IDF retrieval + LLM knowledge generation."""
        if not self._mem_keys:
            return ""

        top_matches = _keyword_search_tfidf(query, self._mem_keys, top_k=self.top_k)
        if not top_matches:
            return ""

        knowledge_parts = []
        for i, match_goal in enumerate(top_matches, 1):
            task_nodes = self._mem_forest.get(match_goal, [])
            if not task_nodes:
                continue
            knowledge = self._generate_knowledge(match_goal, task_nodes)
            if knowledge:
                knowledge_parts.append(f"[Reference Knowledge #{i} — Similar task: {match_goal}]\n{knowledge}")

        return "\n\n".join(knowledge_parts)

    def evolve(
        self,
        task_type: str,
        task_description: str,
        trajectory: List[Tuple[str, str]],
        success: bool,
        episode_idx: int,
    ) -> None:
        # MCMA offline mode: no online updates during test
        pass

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument(
            "--mcma_top_k", type=int, default=3,
            help="MCMA: number of similar tasks to retrieve (default: 3)"
        )
        parser.add_argument(
            "--mcma_store_dir", type=str, default=None,
            help="MCMA: directory to persist mem_forest"
        )
        parser.add_argument(
            "--mcma_force_rebuild", action="store_true",
            help="MCMA: force rebuild mem_forest even if cached"
        )
