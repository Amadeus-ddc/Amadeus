#!/usr/bin/env python3
"""
Amadeus × ALFWorld
==================
Use the Amadeus memory system (Knowledge Graph + Builder/Answerer/Optimizer)
as the backbone of an LLM agent to solve ALFWorld tasks.

Architecture
------------
For each environment instance:
  1. Builder  – observes (obs, action, reward) at every step and maintains
                a per-episode knowledge graph (rooms, objects, relations).
  2. Answerer – given the current observation + task, navigates the KG to
                produce context that helps the LLM pick the next action.
  3. Optimizer (optional) – runs adversarial self-play after each episode
                to refine Builder/Answerer prompts.

The interaction loop mirrors verl-agent's GPT-4o prompt agent:
  for each test round:
      reset all envs
      for step in range(max_steps):
          for each active env:
              - Builder ingests last (obs, action) into graph
              - Answerer retrieves relevant memory context
              - LLM decides the next action (with <think>/<action> tags)
          envs.step(actions)
      evaluate success rates
"""

import sys
import os
import re
import json
import yaml
import logging
import argparse
import datetime
import time
import numpy as np
from collections import defaultdict

import ray
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Path setup  (amadeus repo root = two levels up from this file)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AMADEUS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))       # amadeus/
WORKSPACE_ROOT = os.path.dirname(AMADEUS_ROOT)                     # /data/hzy/Amadeus
sys.path.insert(0, WORKSPACE_ROOT)

load_dotenv(os.path.join(AMADEUS_ROOT, "experiments", ".env"))
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE")

# Amadeus components  (run_locomo.py 同样从 amadeus_tzx 导入)
from amadeus_tzx.code.core.graph import MemoryGraph
from amadeus_tzx.code.core.buffer import TimeWindowBuffer
from amadeus_tzx.code.agents.builder import BuilderAgent
from amadeus_tzx.code.agents.answerer import AnswererAgent
from amadeus_tzx.code.agents.questioner import QuestionerAgent
from amadeus_tzx.code.engine.optimizer import AdversarialOptimizer

# verl-agent ALFWorld environment  (reuse its env package directly)
# 注意: 不能通过包路径导入，因为 agent_system.environments.__init__ 会拉入
# env_manager → omegaconf 等不必要的依赖。这里预注册 stub 包绕过 __init__ 链。
VERL_AGENT_ROOT = os.path.join(WORKSPACE_ROOT, "verl-agent")
sys.path.insert(0, VERL_AGENT_ROOT)

# Pre-register stub packages so that envs.py's internal imports
# (e.g. agent_system.environments.env_package.alfworld.alfworld.agents.environment)
# don't trigger the real agent_system/environments/__init__.py (which needs omegaconf).
# Only stub the first two levels; deeper packages have real __init__.py with actual code.
import types
_stub_packages = [
    "agent_system",
    "agent_system.environments",
]
for _pkg_name in _stub_packages:
    if _pkg_name not in sys.modules:
        _stub = types.ModuleType(_pkg_name)
        _stub.__path__ = [os.path.join(VERL_AGENT_ROOT, _pkg_name.replace(".", "/"))]
        _stub.__package__ = _pkg_name
        sys.modules[_pkg_name] = _stub

# Now safe to do standard imports — the stubs prevent __init__.py chains
from agent_system.environments.env_package.alfworld.envs import (
    AlfworldEnvs, build_alfworld_envs, load_config_file,
)
from agent_system.environments.env_package.alfworld.projection import alfworld_projection

# Optional: HuggingFace embedder for semantic search in KG
try:
    import torch
    from transformers import AutoTokenizer, AutoModel

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
                    truncation=True, max_length=256,
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                mask = inputs["attention_mask"]
                tok_emb = outputs.last_hidden_state
                mask_exp = mask.unsqueeze(-1).expand(tok_emb.size()).float()
                emb = torch.sum(tok_emb * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
                return emb[0].cpu().numpy()
            except Exception as e:
                logging.warning(f"Embedding failed: {e}")
                return None
except ImportError:
    HuggingFaceEmbedder = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("Amadeus.ALFWorld")

# ---------------------------------------------------------------------------
# ALFWorld 6 sub-task types
# ---------------------------------------------------------------------------
TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

# ---------------------------------------------------------------------------
# Builder Prompt — ALFWorld 专用
# ---------------------------------------------------------------------------
ALFWORLD_BUILDER_PROMPT = """You are 'The Builder', the memory manager of an embodied agent in ALFWorld.
Your goal is to maintain a **Knowledge Graph** that tracks objects, locations, and the agent's state.

**COGNITIVE PRIMITIVES:**

1. **ADD(subject, object?, content, timestamp?)**
   - Trigger: A NEW fact discovered from the observation.
   - Node: object=null → create entity (e.g., room, object).
   - Edge: object=Target → create relationship.
   - Example: Observation says "You see a mug on the desk." → ADD("mug", "desk", "ON")

2. **UPDATE(subject, object?, content, timestamp?)**
   - Trigger: An entity's state or location changed.
   - Example: Agent picked up the mug → UPDATE("mug", "agent", "HELD_BY") + DELETE("mug", "desk")

3. **DELETE(subject, object?)**
   - Trigger: A relationship is no longer true (object moved, state changed).

4. **WAIT(subject, content)**
   - Trigger: Ambiguous or partial information; defer to next observation.

**KEY RULES:**
- Track every object and its location (on/in which receptacle).
- Track the agent's current room/location.
- Track which receptacles have been examined/opened.
- Track the task goal and progress toward it.
- Use step number as timestamp (e.g., "step_3").
- Keep entity names simple and consistent (lowercase, e.g., "mug 1", "countertop 1").

**SUBJECT RESOLUTION:**
- "You" or "I" always refers to the Agent.
- Parse object names exactly as ALFWorld provides them.

**OUTPUT SCHEMA (JSON):**
{
  "chain_of_thought": "Step analysis...",
  "operations": [
    {
      "action": "ADD" | "UPDATE" | "DELETE" | "WAIT",
      "subject": "EntityName",
      "object": "TargetName" or null,
      "content": "Relation or Description",
      "timestamp": "step_N",
      "reason": "Why this operation."
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# Action-Decision Prompt — 让 LLM 基于 KG 上下文选择动作
# ---------------------------------------------------------------------------
ALFWORLD_ACTION_PROMPT_NO_HISTORY = """You are an expert agent operating in the ALFRED Embodied Environment.

**Memory Context from Knowledge Graph:**
{memory_context}

Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation, your task goal, and what you know from memory. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_ACTION_PROMPT = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

**Memory Context from Knowledge Graph:**
{memory_context}

Prior to this step, you have already taken {step_count} step(s).
Recent action history:
{action_history}

You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation, your task goal, and what you know from memory. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

# ---------------------------------------------------------------------------
# Core: Amadeus ALFWorld Agent
# ---------------------------------------------------------------------------
class AmadeusALFWorldAgent:
    """
    One agent instance per environment.
    Maintains its own MemoryGraph + Builder + Answerer.
    """

    def __init__(
        self,
        env_id: int,
        model_name: str,
        graphs_dir: str,
        embedder=None,
        api_base: str = None,
        api_key: str = None,
        history_length: int = 5,
        use_optimizer: bool = True,
        optimizer_sp_count: int = 3,
    ):
        self.env_id = env_id
        self.model_name = model_name
        self.history_length = history_length
        self.use_optimizer = use_optimizer
        self.api_base = api_base
        self.api_key = api_key

        # Per-episode graph
        graph_path = os.path.join(graphs_dir, f"graph_env{env_id}.json")
        if os.path.exists(graph_path):
            os.remove(graph_path)
        self.graph = MemoryGraph(graph_path, embedder=embedder)

        # Builder — custom prompt for ALFWorld
        # NOTE: BuilderAgent.__init__ doesn't accept api_base/api_key,
        # so we override its client after construction.
        self.builder = BuilderAgent(self.graph, model_name=model_name)
        self.builder.client = OpenAI(
            base_url=api_base or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
        self.builder.static_prompt = ALFWORLD_BUILDER_PROMPT

        # Answerer — for graph-based context retrieval
        self.answerer = AnswererAgent(
            self.graph, model_name=model_name,
            api_base=api_base, api_key=api_key,
        )

        # Optimizer (optional)
        self.optimizer = None
        self.optimizer_sp_count = optimizer_sp_count
        if use_optimizer:
            questioner = QuestionerAgent(model_name=model_name, api_base=api_base, api_key=api_key)
            self.optimizer = AdversarialOptimizer(
                questioner, self.builder, self.answerer,
                model_name=model_name, api_base=api_base, api_key=api_key,
            )

        # LLM client for action selection
        self.client = OpenAI(
            base_url=api_base or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

        # Step-level state
        self.step_count = 0
        self.task_description = ""
        self.action_history = []  # list of (obs, action)

        # Buffer accumulation: Builder decides when to flush
        self._pending_buffer = ""       # accumulated text not yet processed
        self._sp_round = 0               # how many self-play rounds have been triggered

    def reset(self, initial_obs: str):
        """Reset agent state for a new episode."""
        self.step_count = 0
        self.action_history = []
        self._pending_buffer = ""
        self._sp_round = 0

        # Reset graph
        graph_path = self.graph.storage_path
        if os.path.exists(graph_path):
            os.remove(graph_path)
        self.graph = MemoryGraph(graph_path, embedder=self.graph.embedder)
        self.builder.graph = self.graph
        self.answerer.graph = self.graph

        # Extract task description
        marker = "Your task is to: "
        idx = initial_obs.find(marker)
        if idx != -1:
            self.task_description = initial_obs[idx + len(marker):].strip()
        else:
            self.task_description = "Complete the task."

        # Ingest initial observation into graph
        self._ingest_observation(initial_obs, action=None)

    def act(self, observation: str, admissible_actions: list) -> str:
        """
        Given the current observation and admissible actions, decide an action.
        Returns the raw LLM response (with <think>/<action> tags).
        """
        self.step_count += 1

        # 1) Builder: ingest the new observation
        if self.step_count > 1:
            self._ingest_observation(observation, action=self.action_history[-1][1] if self.action_history else None)

        # 2) Answerer: retrieve memory context from KG
        memory_context = self._get_memory_context(observation)

        # 3) Build action-selection prompt
        admissible_str = "\n ".join(f"'{a}'" for a in admissible_actions if a != "help")

        if self.step_count <= 1 or not self.action_history:
            prompt = ALFWORLD_ACTION_PROMPT_NO_HISTORY.format(
                memory_context=memory_context,
                current_observation=observation,
                admissible_actions=admissible_str,
            )
        else:
            recent = self.action_history[-self.history_length:]
            history_str = "\n".join(
                f"[Step {i+1}: Obs='{obs[:100]}...', Action='{act}']"
                for i, (obs, act) in enumerate(recent)
            )
            prompt = ALFWORLD_ACTION_PROMPT.format(
                task_description=self.task_description,
                memory_context=memory_context,
                step_count=len(self.action_history),
                action_history=history_str,
                current_step=self.step_count,
                current_observation=observation,
                admissible_actions=admissible_str,
            )

        # 4) Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1024,
            )
            action_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[Env {self.env_id}] LLM call failed: {e}")
            action_text = "<think>Error</think><action>look</action>"

        return action_text

    def record_action(self, observation: str, action: str):
        """Record the (obs, action) pair for history tracking."""
        self.action_history.append((observation, action))

    def end_episode(self, final_obs: str, won: bool):
        """Called at episode end. Flush remaining buffer and run final self-play."""
        # Flush any remaining buffer content
        if self._pending_buffer.strip():
            try:
                self.builder.process_buffer(self._pending_buffer)
            except Exception as e:
                logger.warning(f"[Env {self.env_id}] Final buffer flush failed: {e}")
            self._pending_buffer = ""

        # Save final graph
        self.graph.save()

        # Final self-play with episode summary
        if self.use_optimizer and self.optimizer and self.graph.graph.number_of_nodes() > 0:
            try:
                summary = f"Task: {self.task_description}\nResult: {'SUCCESS' if won else 'FAILED'}\nFinal observation: {final_obs}"
                action_log = [f"Step {i+1}: {act}" for i, (_, act) in enumerate(self.action_history)]
                self.optimizer.step(
                    summary, action_log,
                    mode="fixed", fixed_loops=self.optimizer_sp_count,
                )
                self._sp_round += 1
                logger.info(f"[Env {self.env_id}] Final self-play (round {self._sp_round}) at episode end")
            except Exception as e:
                logger.warning(f"[Env {self.env_id}] Final optimizer step failed: {e}")

    # ----- Internal helpers -----

    def _check_alfworld_flush(self, current_buffer: str, new_chunk: str) -> bool:
        """
        ALFWorld-specific flush decision.
        In ALFWorld, the agent navigates rooms and manipulates objects.
        Flush triggers when the agent enters a distinct new *phase* of the task.
        """
        # 1. Hard limit: buffer too long → force flush
        if len(current_buffer) > 2000:
            return True

        # 2. Too short to judge → keep accumulating
        if len(current_buffer) < 150:
            return False

        # 3. Rule-based heuristics (fast, no LLM call)
        chunk_lower = new_chunk.lower()
        buffer_lower = current_buffer.lower()

        # 3a. Room transition: "you arrive at" / "on the <furniture>"
        room_signals = ["you arrive at", "on the ", "you open the", "you are in the"]
        for sig in room_signals:
            if sig in chunk_lower and sig not in buffer_lower[-300:]:
                return True

        # 3b. Object state change: picked up / put down / opened / closed
        action_signals = ["you pick up", "you put the", "you open the",
                          "you close the", "you take the", "you heat the",
                          "you cool the", "you clean the", "you slice the",
                          "you toggle the", "you use the"]
        for sig in action_signals:
            if sig in chunk_lower:
                return True

        # 3c. Repeated failure: "nothing happens" appearing multiple times
        fail_count = buffer_lower.count("nothing happens") + chunk_lower.count("nothing happens")
        if fail_count >= 2:
            return True

        # 4. If many steps accumulated (≥5 steps in buffer), flush regardless
        step_markers = buffer_lower.count("--- step context:")
        if step_markers >= 5:
            return True

        # 5. LLM-based semantic check for ambiguous cases
        prompt = f"""You are a Memory Buffer Manager for an embodied AI agent navigating a household in ALFWorld.

The agent explores rooms, picks up objects, and places them at target locations.
Decide if the current exploration buffer should be FLUSHED (= consolidated into long-term memory) now.

**Current Buffer** (last 400 chars):
"{current_buffer[-400:]}"

**Incoming Step**:
"{new_chunk}"

**FLUSH when:**
1. The agent has moved to a NEW ROOM or LOCATION (navigation phase ended).
2. The agent has COMPLETED a sub-goal (e.g., found target object, reached destination).
3. The agent's STRATEGY has shifted (e.g., from searching to executing, or retrying after failure).
4. Enough exploration steps have accumulated that consolidation would help clarity.

**KEEP when:**
1. The agent is still searching the same area or doing sequential examine/look actions.
2. Only 1-2 trivial steps have happened since last flush.

Output JSON: {{"decision": "FLUSH" | "KEEP", "reason": "brief reason"}}
"""
        try:
            response = self.builder.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=128,
            )
            result = json.loads(response.choices[0].message.content)
            decision = result.get("decision", "KEEP").upper() == "FLUSH"
            if decision:
                logger.debug(f"[Env {self.env_id}] LLM flush reason: {result.get('reason', '')}")
            return decision
        except Exception as e:
            logger.debug(f"[Env {self.env_id}] LLM flush check failed: {e}")
            # Fallback: flush if buffer has ≥3 steps
            return buffer_lower.count("--- step context:") >= 3

    def _ingest_observation(self, observation: str, action: str = None):
        """
        Accumulate observation into pending buffer. Builder's check_flush_condition()
        decides whether to flush (process the buffer + trigger self-play).
        """
        step_label = f"step_{self.step_count}"
        new_chunk = f"--- Step Context: {step_label} ---\n"
        if action:
            new_chunk += f"Agent Action: {action}\n"
        new_chunk += f"Observation: {observation}\n"

        # Ask ALFWorld-specific flush logic
        should_flush = False
        try:
            should_flush = self._check_alfworld_flush(self._pending_buffer, new_chunk)
        except Exception as e:
            logger.debug(f"[Env {self.env_id}] flush check failed: {e}")

        if should_flush and self._pending_buffer.strip():
            # --- Flush: process accumulated buffer ---
            logger.info(f"[Env {self.env_id}] Builder triggered FLUSH at {step_label} "
                        f"(buffer len={len(self._pending_buffer)})")
            try:
                kept_items, action_log = self.builder.process_buffer(self._pending_buffer)
            except Exception as e:
                logger.warning(f"[Env {self.env_id}] Builder process_buffer failed at flush: {e}")

            # --- Self-play after each flush (if optimizer is enabled) ---
            if self.use_optimizer and self.optimizer and self.graph.graph.number_of_nodes() > 0:
                try:
                    summary = (f"Task: {self.task_description}\n"
                               f"Progress: step {self.step_count}/{len(self.action_history)} actions taken\n"
                               f"Latest observation: {observation[:200]}")
                    recent_actions = [f"Step {i+1}: {act}" for i, (_, act) in enumerate(self.action_history[-10:])]
                    self.optimizer.step(
                        summary, recent_actions,
                        mode="fixed", fixed_loops=self.optimizer_sp_count,
                    )
                    self._sp_round += 1
                    logger.info(f"[Env {self.env_id}] Self-play round {self._sp_round} triggered at {step_label}")
                except Exception as e:
                    logger.warning(f"[Env {self.env_id}] Optimizer step failed at {step_label}: {e}")

            # Reset buffer, start fresh with the new chunk
            self._pending_buffer = new_chunk
        else:
            # --- Keep accumulating ---
            self._pending_buffer += new_chunk

            # Still need to process the latest chunk into KG for immediate use
            try:
                kept_items, action_log = self.builder.process_buffer(new_chunk)
            except Exception as e:
                logger.warning(f"[Env {self.env_id}] Builder failed at {step_label}: {e}")

    def _get_memory_context(self, current_observation: str) -> str:
        """Retrieve relevant context from the KG for the current situation."""
        if self.graph.graph.number_of_nodes() == 0:
            return "Knowledge graph is empty (first step)."

        # Strategy: Use graph.get_full_state() for small graphs,
        # or use Answerer's search for larger ones.
        num_nodes = self.graph.graph.number_of_nodes()
        num_edges = self.graph.graph.number_of_edges()

        if num_nodes <= 30:
            # Small graph — dump everything
            return self.graph.get_full_state()
        else:
            # Larger graph — use targeted search
            try:
                # Search for relevant nodes
                query = f"{self.task_description} {current_observation[:200]}"
                results = self.graph.primitive_search(query)
                if results:
                    node_info = self.graph.primitive_read(results[:10])
                    neighbor_info = self.graph.primitive_get_neighbors(results[:5])
                    return f"Relevant Entities:\n{node_info}\n\nConnections:\n{neighbor_info}"
                else:
                    return self.graph.get_full_state()
            except Exception:
                return self.graph.get_full_state()


# ---------------------------------------------------------------------------
# Action extraction (reuse verl-agent's projection logic)
# ---------------------------------------------------------------------------
def extract_action(raw_response: str) -> tuple:
    """
    Extract action from LLM response. Returns (action_str, is_valid).
    """
    text = raw_response.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx == -1 or end_idx == -1:
        # Fallback: use last 30 chars
        return text[-30:].strip(), False

    action = text[start_idx + len(start_tag):end_idx].strip()

    # Check for <think> tag
    is_valid = "<think>" in text and "</think>" in text
    return action, is_valid


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(args):
    """Run Amadeus agent on ALFWorld environments."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "experiment.log")
    setup_logging(log_path)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Args: {vars(args)}")

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Initialize embedder (optional)
    embedder = None
    if args.embedding_model and HuggingFaceEmbedder is not None:
        logger.info(f"Loading embedding model: {args.embedding_model}")
        embedder = HuggingFaceEmbedder(args.embedding_model)

    # Initialize ALFWorld environments (reuse verl-agent's env package)
    alf_config_path = os.path.join(
        VERL_AGENT_ROOT,
        "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    if not os.path.exists(alf_config_path):
        logger.error(f"ALFWorld config not found: {alf_config_path}")
        sys.exit(1)

    # Ensure ALFWORLD_DATA is set (needed for game files + PDDL logic)
    if not os.environ.get("ALFWORLD_DATA"):
        # Try standard alfworld cache location
        default_data = os.path.expanduser("~/.cache/alfworld")
        if os.path.isdir(default_data):
            os.environ["ALFWORLD_DATA"] = default_data
            logger.info(f"Auto-detected ALFWORLD_DATA: {default_data}")
        else:
            logger.error("ALFWORLD_DATA env var not set and ~/.cache/alfworld not found. "
                         "Run: alfworld-download to get the data.")
            sys.exit(1)
    else:
        logger.info(f"ALFWORLD_DATA: {os.environ['ALFWORLD_DATA']}")

    # Pre-init Ray with runtime_env so worker processes can find agent_system
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": VERL_AGENT_ROOT,
                    "ALFWORLD_DATA": os.environ["ALFWORLD_DATA"],
                },
            },
        )

    eval_dataset = "eval_in_distribution" if args.eval_in_domain else "eval_out_of_distribution"
    env_kwargs = {"eval_dataset": eval_dataset}
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}

    logger.info(f"Building ALFWorld envs: env_num={args.env_num}, eval_dataset={eval_dataset}")
    envs = build_alfworld_envs(
        alf_config_path,
        seed=args.seed,
        env_num=args.env_num,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )

    # Statistics accumulators
    overall_success_rates = []
    task_success_history = defaultdict(list)

    # ========================= Main Loop =========================
    for test_idx in range(args.test_times):
        logger.info(f"\n{'='*50}\nTest round {test_idx + 1}/{args.test_times}\n{'='*50}")
        start_time = time.time()

        # Reset environments
        # AlfworldEnvs.reset() -> (text_obs_list, image_obs_list, info_list)
        text_obs_list, image_obs_list, infos = envs.reset()
        env_dones = [False] * args.env_num

        # Parse gamefile for task-type tracking
        gamefiles = []
        for info in infos:
            gamefiles.append(info.get("extra.gamefile", ""))

        # Create one Amadeus agent per environment
        graphs_dir = os.path.join(run_dir, f"test{test_idx}", "graphs")
        os.makedirs(graphs_dir, exist_ok=True)

        agents = []
        for i in range(args.env_num):
            agent = AmadeusALFWorldAgent(
                env_id=i,
                model_name=args.model_name,
                graphs_dir=graphs_dir,
                embedder=embedder,
                api_base=args.api_base,
                api_key=args.api_key,
                history_length=args.history_length,
                use_optimizer=args.use_optimizer,
                optimizer_sp_count=args.sp_count,
            )
            agent.reset(text_obs_list[i])
            agents.append(agent)

        # Per-round stats
        success_this_round = np.zeros(args.env_num, dtype=bool)
        task_success_cnt = defaultdict(int)
        task_total_cnt = defaultdict(int)

        # Step loop
        for step_idx in range(args.max_steps):
            active_count = sum(1 for d in env_dones if not d)
            current_sr = success_this_round.mean()
            logger.info(
                f"Step {step_idx + 1}/{args.max_steps} | "
                f"Active: {active_count}/{args.env_num} | "
                f"SR so far: {current_sr:.4f}"
            )

            # Collect raw LLM responses (with <think>/<action> tags)
            raw_responses = []
            for i in range(args.env_num):
                if env_dones[i]:
                    raw_responses.append("None")
                    continue

                admissible = envs.get_admissible_commands[i]
                raw_response = agents[i].act(text_obs_list[i], admissible)

                # Record the extracted action in agent's history
                action_str, is_valid = extract_action(raw_response)
                agents[i].record_action(text_obs_list[i], action_str)

                if not is_valid:
                    logger.debug(f"[Env {i}] Invalid action format, using fallback: {action_str[:50]}")

                raw_responses.append(raw_response)  # pass raw to projection

            # Project raw responses to valid actions
            # Note: alfworld_projection modifies list in-place, so pass a copy
            projected_actions = list(raw_responses)
            projected_actions, valids = alfworld_projection(
                projected_actions, envs.get_admissible_commands
            )

            # Step environments
            # AlfworldEnvs.step() -> (text_obs_list, image_obs_list, rewards, dones, infos)
            text_obs_list, image_obs_list, rewards, dones, infos = envs.step(projected_actions)

            # Check completions
            for i in range(args.env_num):
                if env_dones[i]:
                    continue
                if dones[i]:
                    env_dones[i] = True
                    won = bool(infos[i].get("won", False))
                    success_this_round[i] = won

                    # End episode for agent
                    agents[i].end_episode(text_obs_list[i], won)

                    # Task-type stats
                    gamefile = gamefiles[i] if gamefiles[i] else infos[i].get("extra.gamefile", "")
                    matched = False
                    for task in TASKS:
                        if task in gamefile:
                            task_total_cnt[task] += 1
                            if won:
                                task_success_cnt[task] += 1
                            matched = True
                            break
                    if not matched:
                        task_total_cnt["other"] += 1
                        if won:
                            task_success_cnt["other"] += 1

                    icon = "✅" if won else "❌"
                    logger.info(f"[Env {i}] {icon} Episode done (won={won})")

            if all(env_dones):
                logger.info("All environments finished early!")
                break

        # Handle environments that didn't finish
        for i in range(args.env_num):
            if not env_dones[i]:
                agents[i].end_episode(text_obs_list[i], False)
                gamefile = gamefiles[i] if gamefiles[i] else ""
                for task in TASKS:
                    if task in gamefile:
                        task_total_cnt[task] += 1
                        break

        # Round summary
        round_sr = success_this_round.mean()
        overall_success_rates.append(round_sr)
        logger.info(f"\nRound {test_idx + 1} Success Rate: {round_sr:.4f}")

        for task in TASKS + ["other"]:
            total = task_total_cnt.get(task, 0)
            if total > 0:
                rate = task_success_cnt[task] / total
                task_success_history[task].append(rate)
                logger.info(f"  {task:<40s}: {rate:.4f} ({task_success_cnt[task]}/{total})")

        elapsed = time.time() - start_time
        logger.info(f"Round elapsed: {elapsed:.1f}s\n")

        # Save round results
        round_result = {
            "test_idx": test_idx,
            "success_rate": float(round_sr),
            "task_breakdown": {
                task: {
                    "rate": task_success_cnt.get(task, 0) / max(task_total_cnt.get(task, 0), 1),
                    "success": task_success_cnt.get(task, 0),
                    "total": task_total_cnt.get(task, 0),
                }
                for task in TASKS + ["other"]
            },
            "elapsed_seconds": elapsed,
        }
        with open(os.path.join(run_dir, f"test{test_idx}", "result.json"), "w") as f:
            json.dump(round_result, f, indent=2)

    # ========================= Final Summary =========================
    logger.info(f"\n{'='*50}\nFinal Summary\n{'='*50}")
    logger.info(f"Total tests: {args.test_times} | Envs/test: {args.env_num}")
    logger.info(
        f"Overall SR: {np.mean(overall_success_rates):.4f} ± {np.std(overall_success_rates):.4f}"
    )

    for task in TASKS + ["other"]:
        if task_success_history.get(task):
            rates = task_success_history[task]
            logger.info(f"  {task:<40s}: {np.mean(rates):.4f} ± {np.std(rates):.4f}")

    # Save final summary
    summary = {
        "timestamp": timestamp,
        "config": vars(args),
        "test_times": args.test_times,
        "env_num": args.env_num,
        "overall_success_rate_mean": float(np.mean(overall_success_rates)),
        "overall_success_rate_std": float(np.std(overall_success_rates)),
        "per_round_success_rates": [float(x) for x in overall_success_rates],
        "task_breakdown": {
            task: {
                "mean": float(np.mean(task_success_history[task])) if task_success_history.get(task) else 0.0,
                "std": float(np.std(task_success_history[task])) if task_success_history.get(task) else 0.0,
            }
            for task in TASKS + ["other"]
        },
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {run_dir}")

    # Cleanup
    envs.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Amadeus × ALFWorld Evaluation")

    # Environment
    parser.add_argument("--env_num", type=int, default=134,
                        help="Number of parallel ALFWorld environments")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--test_times", type=int, default=1,
                        help="Number of test rounds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for environment")
    parser.add_argument("--eval_in_domain", action="store_true", default=True,
                        help="Use in-distribution eval set (default: True)")
    parser.add_argument("--eval_out_domain", dest="eval_in_domain", action="store_false",
                        help="Use out-of-distribution eval set")

    # Model
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct",
                        help="LLM model name (must be served via OpenAI-compatible API)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key")

    # Amadeus Memory
    parser.add_argument("--history_length", type=int, default=5,
                        help="Number of recent (obs, action) pairs to include in prompt")
    parser.add_argument("--embedding_model", type=str, default=None,
                        help="Path to HuggingFace embedding model for semantic search in KG")
    parser.add_argument("--use_optimizer", action="store_true", default=False,
                        help="Enable adversarial self-play optimizer after each episode")
    parser.add_argument("--sp_count", type=int, default=3,
                        help="Number of self-play questions per optimizer round")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(AMADEUS_ROOT, "experiments", "ALFWorld", "logs"),
                        help="Base directory for experiment outputs")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run directory name")

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
