#!/usr/bin/env python3
"""
ALFWorld offline evaluation — cold-start from training trajectories, then test.

Protocol:
  1. COLD-START: Read pre-collected training trajectories (alfworld_format_traj.json),
     call memory.evolve() for each — Amadeus builds its memory graph from these.
     No LLM inference, no environment interaction in this phase.
  2. TEST: Run test environments sequentially (streaming-style, one at a time),
     calling memory.search() before each episode. Memory does NOT update during test.

Usage:
  python run_alfworld_offline.py --method amadeus --api_base http://localhost:8003/v1
"""

import sys
import os
import re
import json
import logging
import argparse
import datetime
import time
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import ray
from dotenv import load_dotenv
from openai import OpenAI

from methods import METHODS, load_method


# ---------------------------------------------------------------------------
# Tracked OpenAI client — records API calls, tokens, memory size
# ---------------------------------------------------------------------------
class TrackedOpenAI:
    """Thin wrapper around OpenAI client that counts calls and tokens."""

    def __init__(self, client: OpenAI):
        self._client = client
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.chat = self  # make self.chat.completions.create work

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        resp = self._client.chat.completions.create(**kwargs)
        self.total_calls += 1
        if resp.usage:
            self.total_prompt_tokens += resp.usage.prompt_tokens or 0
            self.total_completion_tokens += resp.usage.completion_tokens or 0
        return resp

    @property
    def total_tokens(self):
        return self.total_prompt_tokens + self.total_completion_tokens

    def stats(self) -> dict:
        return {
            "api_calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AMADEUS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
WORKSPACE_ROOT = os.path.dirname(AMADEUS_ROOT)
sys.path.insert(0, AMADEUS_ROOT)
sys.path.insert(0, WORKSPACE_ROOT)

load_dotenv(os.path.join(AMADEUS_ROOT, ".env"))
load_dotenv(os.path.join(AMADEUS_ROOT, "experiments", ".env"))
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE")

_VERL_CANDIDATES = [
    os.environ.get("VERL_AGENT_ROOT"),
    os.path.join(WORKSPACE_ROOT, "verl-agent"),
    os.path.join(AMADEUS_ROOT, "verl-agent"),
]
VERL_AGENT_ROOT = next(
    (os.path.abspath(path) for path in _VERL_CANDIDATES if path and os.path.isdir(os.path.join(path, "agent_system"))),
    os.path.abspath(os.environ.get("VERL_AGENT_ROOT", os.path.join(WORKSPACE_ROOT, "verl-agent"))),
)
sys.path.insert(0, VERL_AGENT_ROOT)

import types
_stub_packages = ["agent_system", "agent_system.environments"]
for _pkg_name in _stub_packages:
    if _pkg_name not in sys.modules:
        _stub = types.ModuleType(_pkg_name)
        _stub.__path__ = [os.path.join(VERL_AGENT_ROOT, _pkg_name.replace(".", "/"))]
        _stub.__package__ = _pkg_name
        sys.modules[_pkg_name] = _stub

from agent_system.environments.env_package.alfworld.envs import (
    AlfworldEnvs, build_alfworld_envs, load_config_file,
)
from agent_system.environments.env_package.alfworld.projection import alfworld_projection

# ---------------------------------------------------------------------------
# Optional default training trajectory file for cold-start
# ---------------------------------------------------------------------------
def find_default_traj_file():
    candidates = [
        os.environ.get("ALFWORLD_TRAJ_FILE"),
        os.path.join(AMADEUS_ROOT, "dataset", "ALFWorld", "alfworld_format_traj.json"),
        os.path.join(AMADEUS_ROOT, "data", "ALFWorld", "alfworld_format_traj.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None

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
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("ALFWorldOffline")

TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

# ---------------------------------------------------------------------------
# Prompt (same as streaming harness)
# ---------------------------------------------------------------------------
EXAMPLE_DEMONSTRATIONS = """==================================================
EXAMPLE DEMONSTRATIONS
==================================================
Example 1:
Goal: put a mug in coffee table.
Observation: You are in the middle of a room. You see a coffeetable 1, a sidetable 1, a mug 1 on sidetable 1.
Think: I need to pick up the mug from the sidetable and put it on the coffee table.
Action: go to sidetable 1
Observation: You arrive at sidetable 1. On the sidetable, you see a mug 1.
Action: pick up mug 1 from sidetable 1
Observation: You pick up the mug 1.
Action: go to coffeetable 1
Observation: You arrive at coffeetable 1.
Action: put mug 1 in/on coffeetable 1
Observation: You put the mug 1 in/on the coffeetable 1.

Example 2:
Goal: look at pillow under the desklamp.
Observation: You are in the middle of a room. You see a desklamp 1 on sidetable 1, a sofa 1 with a pillow 1.
Think: I need to find the desklamp, turn it on, then examine the pillow under it.
Action: go to sidetable 1
Observation: You arrive at sidetable 1. You see a desklamp 1.
Action: use desklamp 1
Observation: You turn on the desklamp 1.
Action: go to sofa 1
Observation: You arrive at sofa 1. You see a pillow 1.
Action: examine pillow 1
Observation: You examine the pillow 1 under the desklamp.
"""

ACTION_PROMPT_TEMPLATE = """==================================================
ENVIRONMENT INSTRUCTIONS
==================================================
You are an agent in the ALFWorld text-based environment. Your goal is to complete household tasks by taking actions.
- Actions must be chosen from the admissible actions list

{example_demonstrations}=================================================={experience_section}==================================================
YOUR CURRENT TASK
==================================================
Goal: {task_description}
Help: type 'check valid actions' if action fails
Help: type 'inventory' to check items

==================================================
RECENT HISTORY
==================================================
{recent_history}
Current Observation: {current_observation}
Admissible actions: {admissible_actions}

==================================================
OUTPUT FORMAT
==================================================
You MUST respond in EXACTLY ONE of these formats:

Format 1 - Prune irrelevant experiences (use ONLY when retrieved experiences are present):
Think-Prune: <reasoning about which experiences are relevant>
Pruned Experience IDs: [list of experience IDs to remove, e.g., #1, #3]

Format 2 - Internal reasoning:
Think: <your reasoning about what to do next>

Format 3 - Execute action:
Action: <exact action from admissible actions list>

Now respond with a Think-Prune, Think, or Action:"""


def extract_action(raw_response: str) -> Tuple[str, bool]:
    for line in raw_response.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("action:"):
            action = line[len("action:"):].strip()
            if action:
                return action, True
    lines = [l.strip() for l in raw_response.strip().splitlines() if l.strip()]
    return (lines[-1] if lines else "look"), False


def is_valid_action(action: str, admissible: List[str]) -> bool:
    return not admissible or action in admissible


def format_admissible_actions(admissible: List[str]) -> str:
    return ", ".join(f"'{a}'" for a in admissible if a != "help")


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------
def run_single_episode(
    client: OpenAI,
    model_name: str,
    envs: AlfworldEnvs,
    env_idx: int,
    initial_obs: str,
    experience_context: str,
    gamefile: str = "",
    max_steps: int = 50,
    history_length: int = 10,
) -> Tuple[bool, str, List[Tuple[str, str]], str, float]:
    """Run one episode. Returns (success, task_type, trajectory, task_desc, progress)."""

    marker = "Your task is to: "
    idx = initial_obs.find(marker)
    task_desc = initial_obs[idx + len(marker):].strip() if idx != -1 else ""

    task_type = "unknown"
    for t in TASKS:
        if t in (gamefile or ""):
            task_type = t
            break

    # Build experience section for prompt
    if experience_context:
        experience_section = f"""
==================================================
RELEVANT EXPERIENCE
==================================================
{experience_context}
"""
    else:
        experience_section = "\n"

    trajectory: List[Tuple[str, str]] = []
    recent_history_lines: List[str] = []
    active_experiences = experience_context

    current_obs = initial_obs
    success = False
    max_progress = 0.0
    current_admissible = list(envs.get_admissible_commands[env_idx] or [])

    for step in range(max_steps):
        admissible = current_admissible
        admissible_str = format_admissible_actions(admissible)
        history_str = "\n".join(recent_history_lines[-history_length:]) if recent_history_lines else "(none)"

        prompt = ACTION_PROMPT_TEMPLATE.format(
            example_demonstrations=EXAMPLE_DEMONSTRATIONS,
            experience_section=experience_section if active_experiences else "\n",
            task_description=task_desc,
            recent_history=history_str,
            current_observation=current_obs,
            admissible_actions=admissible_str,
        )

        action_str = None
        think_count = 0
        messages = [{"role": "user", "content": prompt}]

        while action_str is None and think_count < 3:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                )
                raw = resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"    LLM error step {step}: {e}")
                raw = "Action: look"

            first_line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
            logger.info(f"    [env {env_idx} step {step} think {think_count}] LLM: {first_line[:220]}")
            extracted_action, found_action = extract_action(raw)
            if found_action:
                action_str = extracted_action
            elif first_line.lower().startswith("think-prune:") and active_experiences:
                think_count += 1
                prune_match = re.search(r"Pruned Experience IDs:\s*\[([^\]]*)\]", raw, re.IGNORECASE)
                if prune_match:
                    for pid in re.findall(r"#(\d+)", prune_match.group(1)):
                        active_experiences = re.sub(
                            rf"\[Experience #{pid}\].*?(?=\[Experience #|\Z)", "",
                            active_experiences, flags=re.DOTALL
                        ).strip()
                    experience_section = (
                        f"\n==================================================\nRELEVANT EXPERIENCE\n"
                        f"==================================================\n{active_experiences}\n"
                        if active_experiences else "\n"
                    )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Now respond with Think or Action."})
            else:
                think_count += 1
                think_text = raw[len("think:"):].strip() if raw.lower().startswith("think:") else raw
                recent_history_lines.append(f"Think: {think_text}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Now execute an action. Respond with:\nAction: <exact action from admissible actions list>"})

        if action_str is None:
            action_str = "look"

        action_to_take = action_str
        retry_count = 0
        while not is_valid_action(action_to_take, admissible) and retry_count < 2:
            retry_count += 1
            logger.info(
                f"    [env {env_idx} step {step}] Invalid action rejected: {action_to_take!r} "
                f"(retry {retry_count}/2)"
            )
            retry_prompt = (
                "Your previous action was not in the admissible actions list and was NOT executed.\n"
                "Choose exactly one action by copying it verbatim from this admissible actions list.\n"
                f"Admissible actions: {format_admissible_actions(admissible)}\n"
                "Respond with only:\nAction: <exact admissible action>"
            )
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": f"Action: {action_to_take}"},
                        {"role": "user", "content": retry_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=128,
                )
                retry_raw = resp.choices[0].message.content.strip()
                retry_first = retry_raw.strip().splitlines()[0].strip() if retry_raw.strip() else ""
                logger.info(f"    [env {env_idx} step {step}] Retry LLM: {retry_first[:220]}")
                retry_action, found_retry = extract_action(retry_raw)
                action_to_take = retry_action if found_retry else retry_raw.strip()
            except Exception as e:
                logger.warning(f"    LLM retry error step {step}: {e}")
                break

        if not is_valid_action(action_to_take, admissible):
            logger.info(f"    [env {env_idx} step {step}] Falling back to 'look' after invalid action: {action_to_take!r}")
            action_to_take = "look"

        logger.info(f"    [env {env_idx} step {step}] Action: {action_to_take!r}")

        try:
            obs, scores, dones, info = ray.get(
                envs.workers[env_idx].step.remote(action_to_take)
            )
            obs = obs[0] if isinstance(obs, (list, tuple)) else obs
            done = dones[0] if isinstance(dones, (list, tuple)) else dones
            for k in list(info.keys()):
                if isinstance(info[k], (list, tuple)) and info[k]:
                    info[k] = info[k][0]
            new_adm = info.get("admissible_commands", None)
            if new_adm is not None:
                current_admissible = list(new_adm) if isinstance(new_adm, (list, tuple)) else [new_adm]
            gcsr = info.get("extra.goal_condition_success_rate", None)
            if gcsr is not None:
                try:
                    max_progress = max(max_progress, float(gcsr))
                except (TypeError, ValueError):
                    pass
            logger.info(
                f"    [env {env_idx} step {step}] done={bool(done)} won={bool(info.get('won', False))} "
                f"progress={max_progress:.2f} obs={str(obs)[:220]!r}"
            )
        except Exception as e:
            logger.warning(f"    Env step error: {e}")
            break

        trajectory.append((current_obs, action_to_take))
        recent_history_lines.append(f"Action: {action_to_take}\nObservation: {obs}")
        current_obs = obs

        if done:
            success = bool(info.get("won", False))
            if success:
                max_progress = 1.0
            break

    return success, task_type, trajectory, task_desc, max_progress


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint(output_dir: Path) -> dict:
    ckpt_file = output_dir / "checkpoint.json"
    if ckpt_file.exists():
        try:
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            logger.info(f"[Checkpoint] Loaded: cold_start_traj={ckpt.get('cold_start_done', 0)}, "
                        f"test_done={len(ckpt.get('completed_env_ids', []))}")
            return ckpt
        except Exception as e:
            logger.warning(f"[Checkpoint] Load failed: {e}")
    return {"cold_start_done": 0, "completed_env_ids": [], "phase": None}


def save_checkpoint(output_dir: Path, cold_start_done: int, completed_env_ids: list, phase: str = None):
    ckpt_file = output_dir / "checkpoint.json"
    ckpt = {
        "cold_start_done": cold_start_done,
        "completed_env_ids": sorted(completed_env_ids),
        "phase": phase,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    try:
        with open(ckpt_file, "w") as f:
            json.dump(ckpt, f, indent=2)
        logger.info(f"[Checkpoint] Saved: cold_start_done={cold_start_done}, "
                    f"test_done={len(completed_env_ids)}, phase={phase}, "
                    f"ts={ckpt['timestamp']}")
    except Exception as e:
        logger.warning(f"[Checkpoint] Save failed: {e}")


# ---------------------------------------------------------------------------
# Phase 1: Cold-start from training trajectories
# ---------------------------------------------------------------------------
def cold_start_from_traj(memory, traj_file: str, max_traj: int = None,
                          resume_from: int = 0, output_dir: Path = None):
    """Feed training trajectories into memory.evolve() without running envs."""
    logger.info(f"=== PHASE 1: Cold-start from {traj_file} ===")
    with open(traj_file) as f:
        traj_data = json.load(f)
    if max_traj:
        traj_data = traj_data[:max_traj]
    total = len(traj_data)
    logger.info(f"  Loaded {total} training trajectories, resuming from {resume_from}")

    for i, item in enumerate(traj_data):
        if i < resume_from:
            continue
        query = item.get("query", "").strip()
        raw_traj = item.get("trajectory", [])

        # Convert [{from, value}] → [(obs, action)] pairs
        observations = [t["value"] for t in raw_traj if t["from"] == "human"]
        gpt_turns = [t["value"] for t in raw_traj if t["from"] == "gpt"]

        trajectory_pairs = []
        for obs, gpt in zip(observations[1:], gpt_turns[1:]):  # skip system prompt / OK
            # Extract action from gpt turn (format: "Thought: ...\nAction: ...")
            action = ""
            for line in gpt.splitlines():
                if line.strip().lower().startswith("action:"):
                    action = line.strip()[len("action:"):].strip()
                    break
            if action:
                trajectory_pairs.append((obs, action))

        # Infer task type from query
        task_type = "unknown"
        q_lower = query.lower()
        if "heat" in q_lower:
            task_type = "pick_heat_then_place_in_recep"
        elif "cool" in q_lower:
            task_type = "pick_cool_then_place_in_recep"
        elif "clean" in q_lower:
            task_type = "pick_clean_then_place_in_recep"
        elif "look" in q_lower or "light" in q_lower or "lamp" in q_lower:
            task_type = "look_at_obj_in_light"
        elif "two" in q_lower or "both" in q_lower:
            task_type = "pick_two_obj_and_place"
        else:
            task_type = "pick_and_place"

        memory.evolve(
            task_type=task_type,
            task_description=query,
            trajectory=trajectory_pairs,
            success=True,   # cold-start trajectory files are expected to contain successful trajectories
            episode_idx=i,
        )

        if (i + 1) % 50 == 0:
            logger.info(f"  Cold-start: {i+1}/{total} done")
            if output_dir:
                save_checkpoint(output_dir, i + 1, [], "cold_start")

    logger.info(f"  Cold-start complete. Graph has {memory.graph.graph.number_of_nodes()} nodes, "
                f"{memory.graph.graph.number_of_edges()} edges.")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ALFWorld offline eval: cold-start from train trajs, then test",
    )
    parser.add_argument("--method", type=str, default="amadeus", choices=list(METHODS.keys()))
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--env_num", type=int, default=134)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--history_length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traj_file", type=str, default=find_default_traj_file(),
                        help="Training trajectory file for cold-start. Can also be set via ALFWORLD_TRAJ_FILE.")
    parser.add_argument("--max_traj", type=int, default=None,
                        help="Max training trajectories to use (default: all)")
    parser.add_argument("--output_dir", type=str, default=None)

    for method_cls in METHODS.values():
        method_cls.add_args(parser)

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(SCRIPT_DIR) / "logs" / f"offline_{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir / "experiment.log"))

    checkpoint = load_checkpoint(output_dir)
    cold_start_done = checkpoint.get("cold_start_done", 0)
    completed_env_ids = checkpoint.get("completed_env_ids", [])
    resume_phase = checkpoint.get("phase")

    logger.info(f"Mode: OFFLINE (cold-start → test)")
    logger.info(f"Method: {args.method}")
    logger.info(f"Model:  {args.model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Traj file: {args.traj_file}")
    logger.info(f"Checkpoint: cold_start_done={cold_start_done}, test_done={len(completed_env_ids)}, phase={resume_phase}")

    if not args.traj_file or not os.path.exists(args.traj_file):
        logger.error(
            "A valid ALFWorld cold-start trajectory file is required for offline mode. "
            "Pass --traj_file or set ALFWORLD_TRAJ_FILE."
        )
        sys.exit(1)

    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_base:
        logger.error("OPENAI_BASE_URL is required. Set it in .env or pass --api_base.")
        sys.exit(1)
    if not api_key:
        logger.error("OPENAI_API_KEY is required. Set it in .env or pass --api_key.")
        sys.exit(1)

    method_kwargs = vars(args).copy()
    method_kwargs["output_dir"] = str(output_dir)
    method_kwargs["api_base"] = api_base
    method_kwargs["api_key"] = api_key
    method_kwargs["model_name"] = args.model_name
    memory = load_method(args.method, **method_kwargs)

    # Phase 1: cold-start
    if resume_phase != "test":
        cold_start_from_traj(memory, args.traj_file, args.max_traj,
                             resume_from=cold_start_done, output_dir=output_dir)
        cold_start_done = args.max_traj if args.max_traj else len(json.load(open(args.traj_file)))
        save_checkpoint(output_dir, cold_start_done, [], "cold_start_done")
    else:
        logger.info(f"[Cold-start] Already completed (checkpoint: {cold_start_done}), skipping.")

    # Phase 2: test
    logger.info(f"\n=== PHASE 2: Test evaluation ===")

    if not os.environ.get("ALFWORLD_DATA"):
        data_candidates = [
            os.path.join(AMADEUS_ROOT, "dataset", "ALFWorld"),
            os.path.expanduser("~/.cache/alfworld"),
        ]
        default_data = next((path for path in data_candidates if os.path.isdir(os.path.join(path, "json_2.1.1"))), None)
        if default_data:
            os.environ["ALFWORLD_DATA"] = default_data
            logger.info(f"Auto-detected ALFWORLD_DATA: {default_data}")
        else:
            logger.error("ALFWORLD_DATA not set and no json_2.1.1 data found")
            sys.exit(1)

    if not ray.is_initialized():
        alfworld_pythonpath = os.path.join(
            VERL_AGENT_ROOT, "agent_system", "environments", "env_package", "alfworld"
        )
        ray_pythonpath = f"{alfworld_pythonpath}:{VERL_AGENT_ROOT}"
        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": ray_pythonpath,
                    "ALFWORLD_DATA": os.environ["ALFWORLD_DATA"],
                },
            },
        )

    alf_config_path = os.path.join(
        VERL_AGENT_ROOT,
        "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    env_kwargs = {"eval_dataset": "eval_in_distribution"}
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}

    logger.info(f"Building {args.env_num} test envs ...")
    envs = build_alfworld_envs(
        alf_config_path, seed=args.seed, env_num=args.env_num,
        group_n=1, is_train=False, env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )

    text_obs_list, _, infos = envs.reset()

    client = TrackedOpenAI(OpenAI(base_url=api_base, api_key=api_key))

    all_sr, all_pr = [], []
    task_success = defaultdict(list)
    task_progress = defaultdict(list)
    results = []
    results_file = output_dir / "results.jsonl"
    if results_file.exists() and completed_env_ids:
        logger.info(f"[Resume] Rebuilding stats from existing results.jsonl ...")
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r["id"] in completed_env_ids:
                        results.append(r)
                        all_sr.append(1.0 if r["success"] else 0.0)
                        all_pr.append(r["progress"])
                        task_success[r["task_type"]].append(1.0 if r["success"] else 0.0)
                        task_progress[r["task_type"]].append(r["progress"])
                except Exception:
                    continue
        logger.info(f"[Resume] Rebuilt stats: {len(all_sr)} completed envs, running SR so far: {sum(all_sr)/max(len(all_sr),1):.3f}")

    total_envs = args.env_num
    completed_env_ids_set = set(completed_env_ids)
    for env_idx in range(total_envs):
        if env_idx in completed_env_ids_set:
            logger.info(f"[{env_idx+1}/{total_envs}] SKIPPED (already completed)")
            continue
        initial_obs = text_obs_list[env_idx]
        marker = "Your task is to: "
        idx = initial_obs.find(marker)
        task_desc = initial_obs[idx + len(marker):].strip() if idx != -1 else ""

        gamefile = infos[env_idx].get("extra.gamefile", "") or ""
        if isinstance(gamefile, list):
            gamefile = gamefile[0] if gamefile else ""
        task_type = "unknown"
        for t in TASKS:
            if t in gamefile:
                task_type = t
                break

        logger.info(f"\n{'='*60}")
        logger.info(f"[{env_idx+1}/{total_envs}] {task_type}")
        logger.info(f"Task: {task_desc[:120]}")

        # Search memory (no update during test)
        experience_context = memory.search(task_desc)
        if experience_context:
            logger.info(f"  Retrieved {len(experience_context)} chars of experience")

        try:
            success, detected_type, trajectory, detected_desc, progress = run_single_episode(
                client, args.model_name, envs, env_idx,
                initial_obs, experience_context,
                gamefile=gamefile,
                max_steps=args.max_steps,
                history_length=args.history_length,
            )
        except Exception as e:
            import traceback
            logger.error(f"  Episode failed: {e}\n{traceback.format_exc()}")
            success, detected_type, trajectory, detected_desc, progress = False, task_type, [], task_desc, 0.0

        all_sr.append(1.0 if success else 0.0)
        all_pr.append(progress)
        task_success[task_type].append(1.0 if success else 0.0)
        task_progress[task_type].append(progress)

        icon = "+" if success else "x"
        running_sr = sum(all_sr) / len(all_sr)
        running_pr = sum(all_pr) / len(all_pr)
        logger.info(f"  [{icon}] SR={1 if success else 0}, P={progress:.2f}, running_SR={running_sr:.3f}, running_PR={running_pr:.3f}")

        results.append({
            "id": env_idx,
            "task_type": task_type,
            "task_description": task_desc,
            "success": success,
            "progress": progress,
            "num_steps": len(trajectory),
            "actions": [action for _, action in trajectory],
        })

        # Save incrementally
        with open(output_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(results[-1]) + "\n")

        completed_env_ids_set.add(env_idx)
        save_checkpoint(output_dir, cold_start_done, list(completed_env_ids_set), "test")

    # Summary
    overall_sr = sum(all_sr) / len(all_sr) if all_sr else 0.0
    overall_pr = sum(all_pr) / len(all_pr) if all_pr else 0.0
    llm_stats = client.stats()
    mem_size = len(getattr(memory, "_memory", getattr(memory, "_docs", [])))

    logger.info(f"\n{'='*60}")
    logger.info(f"OFFLINE EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Method:       {args.method}")
    logger.info(f"Overall SR:   {overall_sr:.3f} ({sum(all_sr):.0f}/{len(all_sr)})")
    logger.info(f"Overall PR:   {overall_pr:.3f}")
    logger.info(f"Memory size:  {mem_size} entries")
    logger.info(f"API calls:    {llm_stats['api_calls']}")
    logger.info(f"Total tokens: {llm_stats['total_tokens']} (prompt={llm_stats['prompt_tokens']}, completion={llm_stats['completion_tokens']})")
    logger.info(f"{'='*60}")
    for task in TASKS:
        srs = task_success.get(task, [])
        prs = task_progress.get(task, [])
        if srs:
            logger.info(f"  {task:<42}: S={np.mean(srs):.3f} P={np.mean(prs):.3f} ({sum(srs):.0f}/{len(srs)})")

    summary = {
        "method": args.method,
        "model": args.model_name,
        "mode": "offline",
        "overall_sr": overall_sr,
        "overall_pr": overall_pr,
        "memory_size": mem_size,
        "llm_stats": llm_stats,
        "task_breakdown": {t: {"sr": float(np.mean(task_success[t])), "pr": float(np.mean(task_progress[t])), "n": len(task_success[t])}
                           for t in TASKS if task_success.get(t)},
        "timestamp": timestamp,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_dir}")

    envs.close()


if __name__ == "__main__":
    main()
