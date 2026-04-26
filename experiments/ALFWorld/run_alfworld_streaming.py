#!/usr/bin/env python3
"""
ALFWorld streaming evaluation — method-agnostic harness.

Follows the Evo-Memory (arXiv:2511.20857) evaluation protocol:
  - 134 ALFWorld environments processed sequentially as a task stream
  - Search → Predict → Evolve loop per episode
  - Memory evolves after each episode; later tasks benefit from earlier experiences
  - Metrics: Success Rate per task type + overall

The memory module is PLUGGABLE — controlled by --method flag.
To add a new method, see methods/base.py for the interface.

Usage:
  # 1. Start vLLM server
  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
      --model /data/hzy/models/Qwen2.5-7B-Instruct --port 8000 \
      --max-model-len 8192 --gpu-memory-utilization 0.9 --trust-remote-code

  # 2. Run with Amadeus memory (streaming)
  python run_alfworld_streaming.py --method amadeus

  # 3. Run baseline (no memory)
  python run_alfworld_streaming.py --method none
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
from typing import List, Tuple, Optional
from collections import defaultdict

import ray
from dotenv import load_dotenv
from openai import OpenAI

from methods import METHODS, load_method


# ---------------------------------------------------------------------------
# Tracked OpenAI client — records API calls and tokens
# ---------------------------------------------------------------------------
class TrackedOpenAI:
    def __init__(self, client: OpenAI):
        self._client = client
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.chat = self

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
SCRIPT_DIR = Path(__file__).resolve().parent


def _find_workspace_root() -> Path:
    env_verl = os.environ.get("VERL_AGENT_ROOT")
    if env_verl:
        return Path(env_verl).expanduser().resolve().parent
    for candidate in SCRIPT_DIR.parents:
        if (candidate / "verl-agent").is_dir():
            return candidate
    return SCRIPT_DIR.parents[2]


AMADEUS_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = _find_workspace_root()

sys.path.insert(0, str(AMADEUS_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT))

load_dotenv(AMADEUS_ROOT / ".env")
load_dotenv(AMADEUS_ROOT / "experiments" / ".env")
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE")

# verl-agent ALFWorld environment (reuse its env package directly)
_VERL_CANDIDATES = [
    Path(os.environ["VERL_AGENT_ROOT"]).expanduser() if os.environ.get("VERL_AGENT_ROOT") else None,
    WORKSPACE_ROOT / "verl-agent",
    AMADEUS_ROOT / "verl-agent",
    AMADEUS_ROOT.parent / "amadeus" / "experiments" / "verl-agent",
]
VERL_AGENT_ROOT = next(
    (p.resolve() for p in _VERL_CANDIDATES if p and (p / "agent_system").is_dir()),
    Path(os.environ.get("VERL_AGENT_ROOT", WORKSPACE_ROOT / "verl-agent")).expanduser().resolve(),
)
sys.path.insert(0, str(VERL_AGENT_ROOT))

# Pre-register stub packages to avoid omegaconf dependency chain
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

logger = logging.getLogger("ALFWorldStreaming")


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
# Static few-shot demonstrations (shared by all methods, aligned with Evo-Memory)
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

# ---------------------------------------------------------------------------
# Action Prompt (aligned with Evo-Memory paper Appendix C)
# ---------------------------------------------------------------------------
ACTION_PROMPT_TEMPLATE = """==================================================
ENVIRONMENT INSTRUCTIONS
==================================================
You are an agent in the ALFWorld text-based environment. Your goal is to complete household tasks by taking actions.
- Actions must be chosen from the admissible actions list

{example_demonstrations}==================================================
{experience_section}==================================================
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
    """Extract action from LLM response (Action: format). Returns (action_str, is_valid)."""
    for line in raw_response.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("action:"):
            action = line[len("action:"):].strip()
            if action:
                return action, True
    # Fallback: last non-empty line
    lines = [l.strip() for l in raw_response.strip().splitlines() if l.strip()]
    return (lines[-1] if lines else "look"), False


# ---------------------------------------------------------------------------
# Single episode runner (method-agnostic)
# ---------------------------------------------------------------------------
def run_single_episode(
    client: OpenAI,
    model_name: str,
    envs: AlfworldEnvs,
    env_idx: int,
    initial_obs: str,
    experience_context: str,
    max_steps: int = 50,
    history_length: int = 5,
) -> Tuple[bool, str, List[Tuple[str, str]], str, float]:
    """
    Run one ALFWorld episode for a single environment.

    Returns: (success, task_type, trajectory, task_description, progress)
    """
    # Extract task description
    marker = "Your task is to: "
    idx = initial_obs.find(marker)
    task_description = initial_obs[idx + len(marker):].strip() if idx != -1 else "Complete the task."

    # Get gamefile for task type
    gamefile = ""
    try:
        info = envs.get_infos[env_idx] if hasattr(envs, 'get_infos') else {}
        gamefile = info.get("extra.gamefile", "")
    except Exception:
        pass

    task_type = "other"
    for t in TASKS:
        if t in gamefile:
            task_type = t
            break

    # Build experience section
    if experience_context:
        experience_section = (
            "RELEVANT EXPERIENCE FROM SIMILAR TASKS\n"
            "==================================================\n"
            f"{experience_context}\n"
        )
    else:
        experience_section = ""

    recent_history_lines = []  # list of "Observation: ...\nAction: ..." strings
    trajectory = []
    obs = initial_obs
    max_progress = 0.0

    for step in range(max_steps):
        admissible = envs.get_admissible_commands[env_idx]
        admissible_str = ", ".join(f"'{a}'" for a in admissible if a != "help")

        # Build RECENT HISTORY block
        if recent_history_lines:
            recent_block = "\n".join(recent_history_lines[-history_length * 2:])
        else:
            recent_block = "(No previous actions in this episode)"

        prompt = ACTION_PROMPT_TEMPLATE.format(
            example_demonstrations=EXAMPLE_DEMONSTRATIONS,
            experience_section=experience_section,
            task_description=task_description,
            recent_history=recent_block,
            current_observation=obs,
            admissible_actions=admissible_str,
        )

        # Think-Prune/Think/Action loop: allow up to 3 rounds before forcing Action
        action_str = None
        think_count = 0
        messages = [{"role": "user", "content": prompt}]

        while action_str is None and think_count < 3:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256,
                )
                raw_response = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"  LLM call failed at step {step}: {e}")
                raw_response = "Action: look"

            # Check if it's Think-Prune, Think, or Action
            first_line = raw_response.strip().splitlines()[0].strip() if raw_response.strip() else ""
            if first_line.lower().startswith("action:"):
                action_str, _ = extract_action(raw_response)
            elif first_line.lower().startswith("think-prune:"):
                # Parse pruned experience IDs and rebuild experience section
                think_count += 1
                pruned_ids = set()
                for line in raw_response.strip().splitlines():
                    if line.strip().lower().startswith("pruned experience ids:"):
                        pruned_ids = set(int(x) for x in re.findall(r'#(\d+)', line))
                        break
                if pruned_ids and experience_context:
                    # Rebuild experience_context removing pruned experiences
                    exp_blocks = re.split(r'(?=\[Experience #\d+\])', experience_context)
                    kept = []
                    for block in exp_blocks:
                        block = block.strip()
                        if not block:
                            continue
                        m = re.match(r'\[Experience #(\d+)\]', block)
                        if m and int(m.group(1)) in pruned_ids:
                            continue
                        kept.append(block)
                    experience_context = "\n\n".join(kept)
                    experience_section = (
                        "RELEVANT EXPERIENCE FROM SIMILAR TASKS\n"
                        "==================================================\n"
                        f"{experience_context}\n"
                    ) if experience_context.strip() else ""
                    # Rebuild prompt with pruned experiences
                    prompt = ACTION_PROMPT_TEMPLATE.format(
                        example_demonstrations=EXAMPLE_DEMONSTRATIONS,
                        experience_section=experience_section,
                        task_description=task_description,
                        recent_history=recent_block,
                        current_observation=obs,
                        admissible_actions=admissible_str,
                    )
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append({"role": "user", "content": "Now respond with Think or Action."})
            else:
                # It's a Think — append to messages and ask for next step
                think_count += 1
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": "Now execute an action. Respond with:\nAction: <exact action from admissible actions list>"})

        if action_str is None:
            action_str = "look"

        trajectory.append((obs, action_str))
        recent_history_lines.append(f"Observation:\n{obs}")
        recent_history_lines.append(f"Action:\n{action_str}")

        # Project action for this single env only (avoid stepping all 134 envs)
        admissible_single = [envs.get_admissible_commands[env_idx]]

        # Fuzzy-match extracted action to closest admissible command
        admissible_list = admissible_single[0]
        extracted_lower = action_str.lower().strip()
        action_to_take = extracted_lower  # fallback
        best_score = -1
        for cand in admissible_list:
            cand_lower = cand.lower().strip()
            if cand_lower == extracted_lower:
                action_to_take = cand
                best_score = 1.0
                break
            e_words = set(extracted_lower.split())
            c_words = set(cand_lower.split())
            if not e_words:
                continue
            score = len(e_words & c_words) / len(e_words | c_words)
            if score > best_score:
                best_score = score
                action_to_take = cand
        if best_score < 0.3 and admissible_list:
            action_to_take = admissible_list[0]

        # Step only the target worker directly (bypasses shared envs.step)
        import ray as _ray
        result = _ray.get(envs.workers[env_idx].step.remote(action_to_take))
        step_obs, step_scores, step_dones, step_info = result
        for k in list(step_info.keys()):
            step_info[k] = step_info[k][0]
        envs.prev_admissible_commands[env_idx] = step_info.get("admissible_commands", admissible_single[0])

        obs = step_obs[0]
        # Track progress via score/max_score (TextWorld native progress signal)
        score = step_info.get("score", None)
        max_score = step_info.get("max_score", None)
        if score is not None and max_score and float(max_score) > 0:
            max_progress = max(max_progress, float(score) / float(max_score))

        if step_dones[0]:
            won = bool(step_info.get("won", False))
            if won:
                max_progress = 1.0
            return won, task_type, trajectory, task_description, max_progress

    return False, task_type, trajectory, task_description, max_progress


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ALFWorld streaming eval — method-agnostic harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available methods:
  none      No memory (baseline)
  amadeus   Amadeus MemoryGraph (Builder + self-play)

To add a new method:
  1. Create methods/your_method.py implementing MemoryModule (see methods/base.py)
  2. Register it in methods/__init__.py METHODS dict
  3. Run: python run_alfworld_streaming.py --method your_method
        """,
    )

    # Harness arguments (method-agnostic)
    parser.add_argument("--method", type=str, default="none", choices=list(METHODS.keys()),
                        help="Memory method to use (default: none)")
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--history_length", type=int, default=5,
                        help="Number of recent (obs, action) pairs in prompt")
    parser.add_argument("--env_num", type=int, default=134,
                        help="Number of ALFWorld environments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_in_domain", action="store_true", default=True)
    parser.add_argument("--eval_out_domain", dest="eval_in_domain", action="store_false")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--max_tasks", type=int, default=-1,
                        help="Max tasks to evaluate (-1 = all)")

    # Let each method add its own arguments
    for method_cls in METHODS.values():
        method_cls.add_args(parser)

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(SCRIPT_DIR) / "logs" / f"streaming_{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir / "experiment.log"))

    logger.info(f"Mode: STREAMING")
    logger.info(f"Method: {args.method}")
    logger.info(f"Model:  {args.model_name}")
    logger.info(f"Output: {output_dir}")

    # Set env vars
    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "token-abc123")
    os.environ["OPENAI_BASE_URL"] = api_base
    os.environ["OPENAI_API_KEY"] = api_key

    # Init memory method
    method_kwargs = vars(args).copy()
    method_kwargs["output_dir"] = str(output_dir)
    method_kwargs["api_base"] = api_base
    method_kwargs["api_key"] = api_key
    memory = load_method(args.method, **method_kwargs)
    logger.info(f"Memory module loaded: {memory.__class__.__name__}")

    # Init OpenAI client
    client = TrackedOpenAI(OpenAI(base_url=api_base, api_key=api_key))

    # Ensure ALFWORLD_DATA is set
    if not os.environ.get("ALFWORLD_DATA"):
        data_candidates = [
            AMADEUS_ROOT / "dataset" / "ALFWorld",
            AMADEUS_ROOT.parent / "amadeus" / "dataset" / "ALFWorld",
            Path.home() / ".cache" / "alfworld",
        ]
        default_data = next((p for p in data_candidates if (p / "json_2.1.1").is_dir()), None)
        if default_data:
            os.environ["ALFWORLD_DATA"] = str(default_data)
            logger.info(f"Auto-detected ALFWORLD_DATA: {default_data}")
        else:
            logger.error("ALFWORLD_DATA env var not set and no json_2.1.1 data found.")
            sys.exit(1)

    # Init Ray
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": str(VERL_AGENT_ROOT),
                    "ALFWORLD_DATA": os.environ["ALFWORLD_DATA"],
                },
            },
        )

    # Build environments
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

    # Reset all envs to get initial observations and gamefiles
    text_obs_list, _, infos = envs.reset()

    gamefiles = [info.get("extra.gamefile", "") for info in infos]
    task_types = []
    for gf in gamefiles:
        tt = "other"
        for t in TASKS:
            if t in gf:
                tt = t
                break
        task_types.append(tt)

    total_envs = args.env_num
    if args.max_tasks > 0:
        total_envs = min(total_envs, args.max_tasks)

    # Resume support
    results_file = output_dir / "results.jsonl"
    completed = set()
    if args.resume and results_file.exists():
        with open(results_file) as f:
            for line in f:
                entry = json.loads(line)
                completed.add(entry["id"])
        logger.info(f"Resuming: {len(completed)} tasks already done")

    # ── Streaming Evaluation Loop ──
    # Process environments sequentially (streaming)
    all_sr = []
    all_pr = []
    task_success = defaultdict(list)
    task_progress = defaultdict(list)

    for env_idx in range(total_envs):
        if env_idx in completed:
            continue

        task_type = task_types[env_idx]
        initial_obs = text_obs_list[env_idx]

        # Extract task description for search
        marker = "Your task is to: "
        idx = initial_obs.find(marker)
        task_desc = initial_obs[idx + len(marker):].strip() if idx != -1 else ""

        logger.info(f"\n{'='*60}")
        logger.info(f"[{env_idx+1}/{total_envs}] Env {env_idx}: {task_type}")
        logger.info(f"Task: {task_desc[:120]}")

        # Step 1: SEARCH — retrieve relevant experience
        experience_context = memory.search(task_desc)
        if experience_context:
            logger.info(f"  Retrieved {len(experience_context)} chars of experience")

        # Step 2: PREDICT — run episode
        # For streaming, we need to reset just this env and run it
        # Since build_alfworld_envs creates all envs, we run one at a time
        try:
            success, detected_task_type, trajectory, detected_task_desc, progress = run_single_episode(
                client, args.model_name, envs, env_idx,
                initial_obs, experience_context,
                max_steps=args.max_steps,
                history_length=args.history_length,
            )
        except Exception as e:
            logger.error(f"  Episode failed: {e}")
            success = False
            trajectory = []
            detected_task_type = task_type
            detected_task_desc = task_desc
            progress = 0.0

        sr = 1.0 if success else 0.0
        all_sr.append(sr)
        all_pr.append(progress)
        task_success[task_type].append(sr)
        task_progress[task_type].append(progress)

        icon = "+" if success else "x"
        running_sr = sum(all_sr) / len(all_sr)
        running_pr = sum(all_pr) / len(all_pr)
        logger.info(
            f"  [{icon}] SR={sr:.0f}, P={progress:.2f}, running_SR={running_sr:.3f}, running_PR={running_pr:.3f}"
        )

        # Step 3: EVOLVE — store experience into memory
        memory.evolve(task_type, detected_task_desc or task_desc, trajectory, success, env_idx)

        # Save result incrementally
        result = {
            "id": env_idx,
            "task_type": task_type,
            "task_description": task_desc,
            "success": success,
            "progress": progress,
            "num_steps": len(trajectory),
            "method": args.method,
            "memory_used": bool(experience_context),
        }
        with open(results_file, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Summary ──
    total = len(all_sr)
    if total == 0:
        logger.info("No tasks evaluated.")
        return

    overall_sr = sum(all_sr) / total
    overall_pr = sum(all_pr) / total
    llm_stats = client.stats()
    mem_size = len(getattr(memory, "_memory", getattr(memory, "_docs", [])))

    logger.info(f"\n{'='*60}")
    logger.info(f"ALFWorld Streaming Evaluation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Method:         {args.method}")
    logger.info(f"Model:          {args.model_name}")
    logger.info(f"Total tasks:    {total}")
    logger.info(f"Success Rate:   {overall_sr:.3f}")
    logger.info(f"Progress Rate:  {overall_pr:.3f}")
    logger.info(f"Memory size:    {mem_size} entries")
    logger.info(f"API calls:      {llm_stats['api_calls']}")
    logger.info(f"Total tokens:   {llm_stats['total_tokens']} (prompt={llm_stats['prompt_tokens']}, completion={llm_stats['completion_tokens']})")

    for task in TASKS + ["other"]:
        rates = task_success.get(task, [])
        progs = task_progress.get(task, [])
        if rates:
            rate = sum(rates) / len(rates)
            prog = sum(progs) / len(progs) if progs else 0
            logger.info(f"  {task:<40s}: S={rate:.3f} P={prog:.3f} ({int(sum(rates))}/{len(rates)})")

    logger.info(f"{'='*60}")

    summary = {
        "method": args.method,
        "model": args.model_name,
        "mode": "streaming",
        "total_tasks": total,
        "success_rate": overall_sr,
        "progress_rate": overall_pr,
        "memory_size": mem_size,
        "llm_stats": llm_stats,
        "task_breakdown": {
            task: {
                "success_rate": sum(task_success[task]) / len(task_success[task]) if task_success.get(task) else 0,
                "progress_rate": sum(task_progress[task]) / len(task_progress[task]) if task_progress.get(task) else 0,
                "success": int(sum(task_success.get(task, []))),
                "total": len(task_success.get(task, [])),
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
