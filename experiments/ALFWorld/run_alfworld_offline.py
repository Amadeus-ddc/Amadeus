#!/usr/bin/env python3
"""
ALFWorld offline evaluation — cold-start from training trajectories, then test.

Protocol:
  1. COLD-START: Read pre-collected training trajectories (alfworld_format_traj.json),
     call memory.evolve() for each — exprag builds its embedding store from these.
     No LLM inference, no environment interaction in this phase.
  2. TEST: Run test environments sequentially (streaming-style, one at a time),
     calling memory.search() before each episode. Memory does NOT update during test.

Usage:
  python run_alfworld_offline.py --method exprag --api_base http://localhost:8003/v1
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
sys.path.insert(0, WORKSPACE_ROOT)

load_dotenv(os.path.join(AMADEUS_ROOT, "experiments", ".env"))
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE")

VERL_AGENT_ROOT = os.path.join(WORKSPACE_ROOT, "verl-agent")
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
# Default training trajectory file
# ---------------------------------------------------------------------------
DEFAULT_TRAJ_FILE = os.path.join(
    WORKSPACE_ROOT, "MemP", "ProcedureMem", "Alfworld", "alfworld_format_traj.json"
)

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


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def fuzzy_match_action(action: str, admissible: List[str]) -> str:
    if not admissible:
        return action
    if action in admissible:
        return action
    scored = [(jaccard_similarity(action, a), a) for a in admissible]
    scored.sort(reverse=True)
    return scored[0][1] if scored[0][0] > 0.3 else action


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
    max_steps: int = 50,
    history_length: int = 10,
) -> Tuple[bool, str, List[Tuple[str, str]], str, float]:
    """Run one episode. Returns (success, task_type, trajectory, task_desc, progress)."""

    marker = "Your task is to: "
    idx = initial_obs.find(marker)
    task_desc = initial_obs[idx + len(marker):].strip() if idx != -1 else ""

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
    task_type = "unknown"
    progress = 0.0
    max_progress = 0.0

    for step in range(max_steps):
        try:
            admissible = ray.get(envs.workers[env_idx].get_admissible_commands.remote())
        except Exception:
            admissible = []

        admissible_str = ", ".join(f"'{a}'" for a in admissible if a != "help")
        history_str = "\n".join(recent_history_lines[-history_length:]) if recent_history_lines else "(none)"

        prompt = ACTION_PROMPT_TEMPLATE.format(
            example_demonstrations=EXAMPLE_DEMONSTRATIONS,
            experience_section=experience_section if active_experiences else "\n",
            task_description=task_desc,
            recent_history=history_str,
            current_observation=current_obs,
            admissible_actions=admissible_str,
        )

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"    LLM error step {step}: {e}")
            raw = "Action: look"

        # Handle Think-Prune
        if raw.lower().startswith("think-prune:") and active_experiences:
            prune_match = re.search(r"Pruned Experience IDs:\s*\[([^\]]*)\]", raw, re.IGNORECASE)
            if prune_match:
                ids_str = prune_match.group(1)
                prune_ids = re.findall(r"#(\d+)", ids_str)
                if prune_ids and active_experiences:
                    lines = active_experiences.split("\n")
                    for pid in sorted(prune_ids, reverse=True):
                        active_experiences = re.sub(
                            rf"\[Experience #{pid}\].*?(?=\[Experience #|\Z)", "",
                            active_experiences, flags=re.DOTALL
                        ).strip()
                    if active_experiences:
                        experience_section = f"""
==================================================
RELEVANT EXPERIENCE
==================================================
{active_experiences}
"""
                    else:
                        experience_section = "\n"
            continue  # don't step env on prune turn

        # Handle Think
        if raw.lower().startswith("think:"):
            think_text = raw[len("think:"):].strip()
            recent_history_lines.append(f"Think: {think_text}")
            continue  # don't step env on think turn

        # Handle Action
        action_str, _ = extract_action(raw)
        action_to_take = fuzzy_match_action(action_str, admissible)

        try:
            obs, scores, dones, info = ray.get(
                envs.workers[env_idx].step.remote(action_to_take)
            )
            obs = obs[0] if isinstance(obs, list) else obs
            done = dones[0] if isinstance(dones, list) else dones
            gcsr = info.get("score", None)
            max_score = info.get("max_score", None)
            if gcsr is not None and max_score and float(max_score) > 0:
                max_progress = max(max_progress, float(gcsr) / float(max_score))
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
            gamefile = info.get("extra.gamefile", "")
            for t in TASKS:
                if t in gamefile:
                    task_type = t
                    break
            break

    return success, task_type, trajectory, task_desc, max_progress


# ---------------------------------------------------------------------------
# Phase 1: Cold-start from training trajectories
# ---------------------------------------------------------------------------
def cold_start_from_traj(memory, traj_file: str, max_traj: int = None):
    """Feed training trajectories into memory.evolve() without running envs."""
    logger.info(f"=== PHASE 1: Cold-start from {traj_file} ===")
    with open(traj_file) as f:
        traj_data = json.load(f)
    if max_traj:
        traj_data = traj_data[:max_traj]
    logger.info(f"  Loaded {len(traj_data)} training trajectories")

    for i, item in enumerate(traj_data):
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
            success=True,   # MemP traj file contains successful trajectories only
            episode_idx=i,
        )

        if (i + 1) % 500 == 0:
            logger.info(f"  Cold-start: {i+1}/{len(traj_data)} done")

    logger.info(f"  Cold-start complete. Memory has {len(getattr(memory, '_memory', []))} entries.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ALFWorld offline eval: cold-start from train trajs, then test",
    )
    parser.add_argument("--method", type=str, default="exprag", choices=list(METHODS.keys()))
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--env_num", type=int, default=134)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--history_length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traj_file", type=str, default=DEFAULT_TRAJ_FILE,
                        help="Training trajectory file for cold-start")
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

    logger.info(f"Mode: OFFLINE (cold-start → test)")
    logger.info(f"Method: {args.method}")
    logger.info(f"Model:  {args.model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Traj file: {args.traj_file}")

    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "token-abc123")

    method_kwargs = vars(args).copy()
    method_kwargs["api_base"] = api_base
    method_kwargs["api_key"] = api_key
    method_kwargs["model_name"] = args.model_name
    memory = load_method(args.method, **method_kwargs)

    # Phase 1: cold-start
    cold_start_from_traj(memory, args.traj_file, args.max_traj)

    # Phase 2: test
    logger.info(f"\n=== PHASE 2: Test evaluation ===")

    if not os.environ.get("ALFWORLD_DATA"):
        default_data = os.path.expanduser("~/.cache/alfworld")
        if os.path.isdir(default_data):
            os.environ["ALFWORLD_DATA"] = default_data
        else:
            logger.error("ALFWORLD_DATA not set")
            sys.exit(1)

    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": VERL_AGENT_ROOT,
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

    total_envs = args.env_num
    for env_idx in range(total_envs):
        initial_obs = text_obs_list[env_idx]
        marker = "Your task is to: "
        idx = initial_obs.find(marker)
        task_desc = initial_obs[idx + len(marker):].strip() if idx != -1 else ""

        gamefile = infos[env_idx].get("extra.gamefile", "")
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
                max_steps=args.max_steps,
                history_length=args.history_length,
            )
        except Exception as e:
            logger.error(f"  Episode failed: {e}")
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
        })

        # Save incrementally
        with open(output_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(results[-1]) + "\n")

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
