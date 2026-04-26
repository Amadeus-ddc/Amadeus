#!/usr/bin/env python3
"""
ALFWorld batch (non-streaming) evaluation — method-agnostic harness.

This runs ALL environments in parallel without cross-episode memory evolution.
Each episode is independent — no streaming memory accumulation.

For streaming evaluation (with memory evolution), use run_alfworld_streaming.py.

The memory module is PLUGGABLE — controlled by --method flag.
To add a new method, see methods/base.py for the interface.

Usage:
  # 1. Start vLLM server
  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
      --model /data/hzy/models/Qwen2.5-7B-Instruct --port 8000 \
      --max-model-len 8192 --gpu-memory-utilization 0.9 --trust-remote-code

  # 2. Run batch evaluation (no memory)
  python run_alfworld_batch.py --method none

  # 3. Run batch evaluation (with per-episode memory, no cross-episode transfer)
  python run_alfworld_batch.py --method amadeus
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
    os.path.join(os.path.dirname(AMADEUS_ROOT), "amadeus", "experiments", "verl-agent"),
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

logger = logging.getLogger("ALFWorldBatch")


TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


# ---------------------------------------------------------------------------
# Action Prompt (same as streaming, but no experience section)
# ---------------------------------------------------------------------------
ACTION_PROMPT_NO_HISTORY = """You are an expert agent operating in the ALFRED Embodied Environment.

Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation, your task goal, and what you know from memory. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ACTION_PROMPT = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Recent action history:
{action_history}

You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation, your task goal, and what you know from memory. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def extract_action(raw_response: str) -> Tuple[str, bool]:
    text = raw_response.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx == -1 or end_idx == -1:
        return text[-30:].strip(), False

    action = text[start_idx + len(start_tag):end_idx].strip()
    is_valid = "<think>" in text and "</think>" in text
    return action, is_valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ALFWorld batch (non-streaming) eval — method-agnostic harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available methods:
  none      No memory (baseline)
  amadeus   Amadeus (per-episode only, no cross-episode transfer)

For streaming evaluation with cross-episode memory, use run_alfworld_streaming.py.
        """,
    )

    parser.add_argument("--method", type=str, default="none", choices=list(METHODS.keys()),
                        help="Memory method (default: none). In batch mode, memory is NOT shared across episodes.")
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--env_num", type=int, default=134)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--test_times", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_in_domain", action="store_true", default=True)
    parser.add_argument("--eval_out_domain", dest="eval_in_domain", action="store_false")
    parser.add_argument("--history_length", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)

    for method_cls in METHODS.values():
        method_cls.add_args(parser)

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(SCRIPT_DIR) / "logs" / f"batch_{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir / "experiment.log"))

    logger.info(f"Mode: BATCH (non-streaming)")
    logger.info(f"Method: {args.method}")
    logger.info(f"Model:  {args.model_name}")
    logger.info(f"Output: {output_dir}")

    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "token-abc123")
    os.environ["OPENAI_BASE_URL"] = api_base
    os.environ["OPENAI_API_KEY"] = api_key

    client = OpenAI(base_url=api_base, api_key=api_key)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Ensure ALFWORLD_DATA is set
    if not os.environ.get("ALFWORLD_DATA"):
        data_candidates = [
            os.path.join(AMADEUS_ROOT, "dataset", "ALFWorld"),
            os.path.join(os.path.dirname(AMADEUS_ROOT), "amadeus", "dataset", "ALFWorld"),
            os.path.expanduser("~/.cache/alfworld"),
        ]
        default_data = next((path for path in data_candidates if os.path.isdir(os.path.join(path, "json_2.1.1"))), None)
        if default_data:
            os.environ["ALFWORLD_DATA"] = default_data
            logger.info(f"Auto-detected ALFWORLD_DATA: {default_data}")
        else:
            logger.error("ALFWORLD_DATA env var not set and no json_2.1.1 data found.")
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
    eval_dataset = "eval_in_distribution" if args.eval_in_domain else "eval_out_of_distribution"
    env_kwargs = {"eval_dataset": eval_dataset}
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}

    logger.info(f"Building ALFWorld envs: env_num={args.env_num}")
    envs = build_alfworld_envs(
        alf_config_path, seed=args.seed, env_num=args.env_num,
        group_n=1, is_train=False, env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )

    overall_success_rates = []
    task_success_history = defaultdict(list)

    for test_idx in range(args.test_times):
        logger.info(f"\n{'='*50}\nTest round {test_idx + 1}/{args.test_times}\n{'='*50}")
        start_time = time.time()

        text_obs_list, _, infos = envs.reset()
        env_dones = [False] * args.env_num

        gamefiles = [info.get("extra.gamefile", "") for info in infos]

        # Extract task descriptions
        task_descriptions = []
        for obs in text_obs_list:
            marker = "Your task is to: "
            idx = obs.find(marker)
            task_descriptions.append(obs[idx + len(marker):].strip() if idx != -1 else "")

        # Track per-env action history
        action_histories = [[] for _ in range(args.env_num)]

        success_this_round = np.zeros(args.env_num, dtype=bool)
        task_success_cnt = defaultdict(int)
        task_total_cnt = defaultdict(int)
        task_progress_sum = defaultdict(float)
        env_max_progress = np.zeros(args.env_num, dtype=float)

        # Step loop — all environments in parallel
        for step_idx in range(args.max_steps):
            active_count = sum(1 for d in env_dones if not d)
            if active_count == 0:
                break

            logger.info(f"Step {step_idx + 1}/{args.max_steps} | Active: {active_count}/{args.env_num}")

            raw_responses = []
            for i in range(args.env_num):
                if env_dones[i]:
                    raw_responses.append("None")
                    continue

                admissible = envs.get_admissible_commands[i]
                admissible_str = "\n ".join(f"'{a}'" for a in admissible if a != "help")

                if step_idx == 0 or not action_histories[i]:
                    prompt = ACTION_PROMPT_NO_HISTORY.format(
                        current_observation=text_obs_list[i],
                        admissible_actions=admissible_str,
                    )
                else:
                    recent = action_histories[i][-args.history_length:]
                    history_str = "\n".join(
                        f"[Step {j+1}: Obs='{o[:100]}...', Action='{a}']"
                        for j, (o, a) in enumerate(recent)
                    )
                    prompt = ACTION_PROMPT.format(
                        task_description=task_descriptions[i],
                        step_count=len(action_histories[i]),
                        action_history=history_str,
                        current_step=step_idx + 1,
                        current_observation=text_obs_list[i],
                        admissible_actions=admissible_str,
                    )

                try:
                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=1024,
                    )
                    raw_response = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"[Env {i}] LLM call failed: {e}")
                    raw_response = "<think>Error</think><action>look</action>"

                action_str, _ = extract_action(raw_response)
                action_histories[i].append((text_obs_list[i], action_str))
                raw_responses.append(raw_response)

            projected, _ = alfworld_projection(raw_responses, envs.get_admissible_commands)
            text_obs_list, _, rewards, dones, infos = envs.step(projected)

            for i in range(args.env_num):
                if env_dones[i]:
                    continue
                # track progress via score/max_score
                score = float(infos[i].get("score", 0.0))
                max_score = float(infos[i].get("max_score", 1.0))
                pr = score / max_score if max_score > 0 else 0.0
                env_max_progress[i] = max(env_max_progress[i], pr)
                if dones[i]:
                    env_dones[i] = True
                    won = bool(infos[i].get("won", False))
                    success_this_round[i] = won
                    if won:
                        env_max_progress[i] = 1.0

                    gamefile = gamefiles[i] or infos[i].get("extra.gamefile", "")
                    for task in TASKS:
                        if task in gamefile:
                            task_total_cnt[task] += 1
                            if won:
                                task_success_cnt[task] += 1
                            break
                    else:
                        task_total_cnt["other"] += 1
                        if won:
                            task_success_cnt["other"] += 1

        # Handle unfinished
        for i in range(args.env_num):
            if not env_dones[i]:
                gamefile = gamefiles[i]
                matched = False
                for task in TASKS:
                    if task in gamefile:
                        task_total_cnt[task] += 1
                        matched = True
                        break
                if not matched:
                    task_total_cnt["other"] += 1

        round_sr = success_this_round.mean()
        round_pr = env_max_progress.mean()
        overall_success_rates.append(round_sr)

        logger.info(f"\nRound {test_idx + 1} SR: {round_sr:.4f}  PR: {round_pr:.4f}")
        for task in TASKS + ["other"]:
            total = task_total_cnt.get(task, 0)
            if total > 0:
                rate = task_success_cnt[task] / total
                task_success_history[task].append(rate)
                logger.info(f"  {task:<40s}: {rate:.4f} ({task_success_cnt[task]}/{total})")

        elapsed = time.time() - start_time
        logger.info(f"Round elapsed: {elapsed:.1f}s")

    # Final Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"ALFWorld Batch Evaluation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Method:     {args.method}")
    logger.info(f"Model:      {args.model_name}")
    logger.info(f"Mode:       BATCH (non-streaming)")
    logger.info(f"SR:         {np.mean(overall_success_rates):.4f}")
    logger.info(f"{'='*60}")

    summary = {
        "method": args.method,
        "model": args.model_name,
        "mode": "batch",
        "test_times": args.test_times,
        "env_num": args.env_num,
        "overall_success_rate": float(np.mean(overall_success_rates)),
        "per_round": [float(x) for x in overall_success_rates],
        "task_breakdown": {
            task: {
                "mean": float(np.mean(task_success_history[task])) if task_success_history.get(task) else 0.0,
            }
            for task in TASKS + ["other"]
        },
        "timestamp": timestamp,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    envs.close()


if __name__ == "__main__":
    main()
