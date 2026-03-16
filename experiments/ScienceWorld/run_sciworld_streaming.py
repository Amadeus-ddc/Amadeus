#!/usr/bin/env python3
"""
ScienceWorld streaming evaluation — method-agnostic harness.

Follows the Evo-Memory (arXiv:2511.20857) evaluation protocol:
  - Uses AgentBoard's ScienceWorld test set (90 tasks)
  - Metrics: Success Rate (S), Progress Rate (P), Grounding Accuracy
  - Streaming: tasks processed sequentially; memory evolves after each episode
  - Simplifications: selfWateringFlowerPots, openContainers, openDoors, noElectricalAction
  - Max 30 steps per episode

The memory module is PLUGGABLE — controlled by --method flag.
To add a new method, see methods/base.py for the interface.

Usage:
  # 1. Start vLLM server
  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
      --model /data/hzy/models/Qwen2.5-7B-Instruct --port 8000 \
      --max-model-len 8192 --gpu-memory-utilization 0.9 --trust-remote-code

  # 2. Run with Amadeus memory
  python run_sciworld_streaming.py --method amadeus

  # 3. Run baseline (no memory)
  python run_sciworld_streaming.py --method none

  # 4. Run with your own method
  #    (1) Create methods/your_method.py implementing MemoryModule
  #    (2) Register in methods/__init__.py
  #    (3) python run_sciworld_streaming.py --method your_method
"""

import sys
import os
import json
import re
import logging
import argparse
import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import jsonlines
from openai import OpenAI
from scienceworld import ScienceWorldEnv

from methods import METHODS, load_method

# ── Logging ─────────────────────────────────────────────────────
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

logger = logging.getLogger("SciWorldStreaming")


# ── AgentBoard ScienceWorld Environment ─────────────────────────
SIMPLIFICATIONS = "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction"


class SciWorldEnvWrapper:
    """Wraps ScienceWorldEnv with AgentBoard-style subgoal tracking."""

    def __init__(self, env_step_limit: int = 30):
        self.env = ScienceWorldEnv("", envStepLimit=env_step_limit)
        self.finished_sub_goal = []
        self.subgoals = []
        self.modified_goal = ""
        self.difficulty = ""

    def load_task(self, task_name: str, var: int, label: dict):
        self.env.load(task_name, var, simplificationStr=SIMPLIFICATIONS, generateGoldPath=True)
        self.subgoals = label["subgoals"]
        self.modified_goal = label["modified_goal"]
        self.difficulty = label["difficulty"]
        self.finished_sub_goal = [0] * len(self.subgoals)

    def reset(self):
        obs, info = self.env.reset()
        inventory = self.env.inventory()
        init_obs = obs + f"\n{inventory}"
        return init_obs

    def step(self, action: str):
        action = action.strip()
        if action == "check valid actions":
            valid_actions = self.get_valid_actions()
            obs = f"Choose an action from these valid actions: {', '.join(valid_actions)}"
            return obs, self.get_progress(), self.is_done(), False
        obs, _, done_env, info = self.env.step(action)
        self._check_subgoals(obs)
        return obs, self.get_progress(), self.is_done(), self._is_valid_action(action)

    def get_valid_actions(self) -> List[str]:
        try:
            combos = self.env.getValidActionObjectCombinationsWithTemplates()
            forbidden = {"teleport", "connect", "dunk", "eat", "flush", "close door"}
            actions = []
            for c in combos:
                v = c["action"]
                if not any(fw in v for fw in forbidden):
                    actions.append(v)
            if "check valid actions" not in actions:
                actions.append("check valid actions")
            return actions
        except Exception:
            return ["check valid actions"]

    def _is_valid_action(self, action: str) -> bool:
        try:
            valid = self.get_valid_actions()
            return action in valid
        except Exception:
            return False

    def _check_subgoals(self, observation: str):
        for i, pattern in enumerate(self.subgoals):
            if re.search(pattern, observation):
                self.finished_sub_goal[i] = 1

    def get_progress(self) -> float:
        if not self.finished_sub_goal:
            return 0.0
        return sum(self.finished_sub_goal) / len(self.finished_sub_goal)

    def is_done(self) -> bool:
        return sum(self.finished_sub_goal) >= len(self.subgoals)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ── Agent Prompt ───────────────────────────────────────────────
AGENT_INSTRUCTION = """You are an agent in a virtual science school environment, tasked to interact with various elements. Here are the commands you can use:

- **Manipulation**:
  - `open {OBJ}` / `close {OBJ}`: Interact with a container.
  - `pick up {OBJ}`: Add an object to your inventory.
  - `put down {OBJ}`: Remove an object from your inventory.
  - `move {OBJ} to {OBJ}`: Transfer an object.
  - `pour {OBJ} into {OBJ}`: Pour a substance.
  - `dunk {OBJ} into {OBJ}`: Immerse a container in a liquid.
  - `mix {OBJ}`: Chemically combine contents.

- **Inspection**:
  - `look around`: Survey your surroundings.
  - `look at {OBJ}`: Examine an object closely.
  - `look in {OBJ}`: Peek inside a container.
  - `read {OBJ}`: Review written content.

- **Device Operations**:
  - `activate {OBJ}` / `deactivate {OBJ}`: Toggle a device.
  - `use {OBJ} [on {OBJ}]`: Utilize a device or item.

- **Movement**:
  - `go to {LOC}`: Relocate.

- **Miscellaneous**:
  - `focus on {OBJ}`: Direct attention to a particular object.
  - `wait [DURATION]`: Pause for a specified period.

- **Information**:
  - `task`: Recap your current objective.
  - `inventory`: Display items you're carrying.
  - `check valid actions`: Show all currently valid actions."""

EXAMPLE_TRAJECTORY = (
    "Task Description: Your task is to boil water.\n\n"
    "ACTION: look around\n"
    "OBSERVATION: This room is called the hallway. In it, you see: a door to the kitchen (that is open)\n\n"
    "ACTION: go to kitchen\n"
    "OBSERVATION: You move to the kitchen.\n\n"
    "ACTION: look around\n"
    "OBSERVATION: This room is called the kitchen. In it, you see: a sink, a stove, a cupboard, a metal pot.\n\n"
    "ACTION: pick up metal pot\n"
    "OBSERVATION: You move the metal pot to the inventory.\n\n"
    "ACTION: move metal pot to sink\n"
    "OBSERVATION: You move the metal pot to the sink.\n\n"
    "ACTION: activate sink\n"
    "OBSERVATION: The sink is now activated.\n\n"
    "ACTION: deactivate sink\n"
    "OBSERVATION: The sink is now deactivated.\n\n"
    "ACTION: pick up metal pot\n"
    "OBSERVATION: You move the metal pot to the inventory.\n\n"
    "ACTION: focus on substance in metal pot\n"
    "OBSERVATION: You focus on the water.\n\n"
    "ACTION: move metal pot to stove\n"
    "OBSERVATION: You move the metal pot to the stove.\n\n"
    "ACTION: activate stove\n"
    "OBSERVATION: The stove is now activated."
)


def build_agent_prompt(
    goal: str,
    history: List[Tuple[str, str]],
    init_obs: str,
    experience_context: str = "",
) -> List[dict]:
    """Build the chat prompt following Evo-Memory / AgentBoard format."""

    system_msg = "You are a helpful agent that interacts with the virtual science school environment to solve the given task."

    parts = []

    # 1. Environment instructions
    parts.append("==================================================")
    parts.append("ENVIRONMENT INSTRUCTIONS")
    parts.append("==================================================")
    parts.append(AGENT_INSTRUCTION)

    # 2. Example
    parts.append("\n==================================================")
    parts.append("EXAMPLE DEMONSTRATIONS")
    parts.append("==================================================")
    parts.append(EXAMPLE_TRAJECTORY)

    # 3. Retrieved experience (streaming memory)
    if experience_context:
        parts.append("\n==================================================")
        parts.append("RELEVANT EXPERIENCE FROM SIMILAR TASKS")
        parts.append("==================================================")
        parts.append(experience_context)

    # 4. Current task
    parts.append("\n==================================================")
    parts.append("YOUR CURRENT TASK")
    parts.append("==================================================")
    parts.append(f"Goal: {goal}")
    parts.append("Help: type 'check valid actions' if action fails")
    parts.append("Help: type 'inventory' to check items")

    # 5. Recent history
    parts.append("\n==================================================")
    parts.append("RECENT HISTORY")
    parts.append("==================================================")
    parts.append(f"Observation:\n{init_obs}")
    for action, obs in history:
        parts.append(f"Action:\n{action}")
        parts.append(f"Observation:\n{obs}")

    # 6. Output format
    parts.append("\n==================================================")
    parts.append("OUTPUT FORMAT")
    parts.append("==================================================")
    parts.append("You MUST respond in EXACTLY ONE of these formats:")
    parts.append("")
    if experience_context:
        parts.append("Format 1 - Prune irrelevant experiences (use ONLY when retrieved experiences are present):")
        parts.append("Think-Prune: <reasoning about which experiences are relevant>")
        parts.append("Pruned Experience IDs: [list of experience IDs to remove, e.g., #1, #3]")
        parts.append("")
        parts.append("Format 2 - Internal reasoning:")
        parts.append("Think: <your reasoning>")
        parts.append("")
        parts.append("Format 3 - Execute action:")
    else:
        parts.append("Format 1 - Internal reasoning:")
        parts.append("Think: <your reasoning>")
        parts.append("")
        parts.append("Format 2 - Execute action:")
    parts.append("Action: <exact command>")
    parts.append("Must be a valid command from ENVIRONMENT INSTRUCTIONS.")
    parts.append("\nRespond with a Think-Prune, Think, or Action:" if experience_context else "\nRespond with a Think or Action:")

    user_msg = "\n".join(parts)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def extract_action(llm_output: str) -> Optional[str]:
    """Extract action from LLM response."""
    match = re.search(r"Action:\s*(.+)", llm_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if re.match(r"Think:", llm_output, re.IGNORECASE):
        return None
    if re.match(r"Think-Prune:", llm_output, re.IGNORECASE):
        return None
    lines = [l.strip() for l in llm_output.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        for prefix in ["Action:", "action:", ">", "- "]:
            if last.startswith(prefix):
                last = last[len(prefix):].strip()
        return last
    return None


# ── Episode Runner ──────────────────────────────────────────────
def run_episode(
    client: OpenAI,
    model_name: str,
    env: SciWorldEnvWrapper,
    goal: str,
    init_obs: str,
    experience_context: str,
    max_steps: int = 30,
    temperature: float = 0.0,
) -> Tuple[float, bool, float, List[dict], int]:
    """
    Run one ScienceWorld episode.
    Returns: (progress_rate, success, grounding_acc, trajectory, num_steps)
    """
    history = []
    trajectory = [{"Goal": goal, "id": 0}, {"Observation": init_obs, "id": 0}]
    grounding_count = 0
    total_actions = 0
    num_steps = 0

    for step in range(max_steps):
        recent_history = history[-15:]
        prompt_messages = build_agent_prompt(goal, recent_history, init_obs, experience_context)

        # Think-Prune/Think/Action loop: allow up to 3 rounds before forcing Action
        action = None
        think_count = 0
        messages = list(prompt_messages)

        while action is None and think_count < 3:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=256,
                )
                llm_output = response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"  LLM call failed at step {step}: {e}")
                llm_output = "Action: look around"

            first_line = llm_output.strip().splitlines()[0].strip() if llm_output.strip() else ""
            if first_line.lower().startswith("action:"):
                action = extract_action(llm_output)
            elif first_line.lower().startswith("think-prune:"):
                think_count += 1
                pruned_ids = set()
                for line in llm_output.strip().splitlines():
                    if line.strip().lower().startswith("pruned experience ids:"):
                        pruned_ids = set(int(x) for x in re.findall(r'#(\d+)', line))
                        break
                if pruned_ids and experience_context:
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
                    # Rebuild prompt with pruned experiences
                    prompt_messages = build_agent_prompt(goal, recent_history, init_obs, experience_context)
                    messages = list(prompt_messages)
                else:
                    messages.append({"role": "assistant", "content": llm_output})
                    messages.append({"role": "user", "content": "Now respond with Think or Action."})
            else:
                # It's a Think — append to messages and ask for next step
                think_count += 1
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({"role": "user", "content": "Now execute an action. Respond with:\nAction: <exact command>"})

        if action is None:
            action = "look around"

        logger.debug(f"  Step {step}: Action={action}")
        trajectory.append({"Action": action, "id": step})

        obs, progress, done, is_valid = env.step(action)
        total_actions += 1
        if is_valid:
            grounding_count += 1

        history.append((action, obs))
        trajectory.append({"Observation": obs, "id": step})
        trajectory.append({"Progress Rate": progress, "id": step})
        num_steps = step + 1

        if done:
            break

    progress_rate = env.get_progress()
    success = env.is_done()
    grounding_acc = grounding_count / total_actions if total_actions > 0 else 0.0

    return progress_rate, success, grounding_acc, trajectory, num_steps


# ── Data Loading ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent


def load_test_data(label_path: str) -> List[dict]:
    labels = []
    with open(label_path, "r") as f:
        for item in jsonlines.Reader(f):
            task_name = item["additional_info"]["env_name"]
            var = item["additional_info"]["var"]
            labels.append({
                "task_name": task_name,
                "var": var,
                "modified_goal": item["goal"],
                "subgoals": item["subgoals"],
                "difficulty": item["difficulty"],
            })
    return labels


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ScienceWorld streaming eval — method-agnostic harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available methods:
  none      No memory (baseline)
  amadeus   Amadeus MemoryGraph (KG + semantic search)

To add a new method:
  1. Create methods/your_method.py implementing MemoryModule (see methods/base.py)
  2. Register it in methods/__init__.py METHODS dict
  3. Run: python run_sciworld_streaming.py --method your_method
        """,
    )

    # Harness arguments (method-agnostic)
    parser.add_argument("--method", type=str, default="none", choices=list(METHODS.keys()),
                        help="Memory method to use (default: none)")
    parser.add_argument("--model_name", type=str, default="/data/hzy/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="token-abc123")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--env_step_limit", type=int, default=100,
                        help="Java-side step limit (should be > max_steps)")
    parser.add_argument("--label_path", type=str,
                        default=str(SCRIPT_DIR / "data" / "test.jsonl"))
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
        output_dir = SCRIPT_DIR / "logs" / f"run_{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir / "experiment.log"))

    logger.info(f"Method: {args.method}")
    logger.info(f"Model:  {args.model_name}")
    logger.info(f"Output: {output_dir}")

    # Set env vars for any sub-modules that need OpenAI config
    os.environ["OPENAI_BASE_URL"] = args.api_base
    os.environ["OPENAI_API_KEY"] = args.api_key

    # Load test data
    labels = load_test_data(args.label_path)
    if args.max_tasks > 0:
        labels = labels[:args.max_tasks]
    logger.info(f"Loaded {len(labels)} tasks from {args.label_path}")

    # Init OpenAI client
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # Init memory method — pass all args as kwargs so each method can pick what it needs
    method_kwargs = vars(args).copy()
    method_kwargs["output_dir"] = str(output_dir)
    memory = load_method(args.method, **method_kwargs)
    logger.info(f"Memory module loaded: {memory.__class__.__name__}")

    # Init environment
    env = SciWorldEnvWrapper(env_step_limit=args.env_step_limit)

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
    all_sr = []
    all_pr = []
    all_gr = []
    difficulties = []

    for idx, label in enumerate(labels):
        if idx in completed:
            continue

        task_name = label["task_name"]
        var = label["var"]
        goal = label["modified_goal"]
        difficulty = label["difficulty"]

        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx+1}/{len(labels)}] Task: {task_name}, var={var}, difficulty={difficulty}")
        logger.info(f"Goal: {goal[:120]}")

        # Step 1: SEARCH — retrieve relevant experience
        experience_context = memory.search(goal)
        if experience_context:
            logger.info(f"  Retrieved {len(experience_context)} chars of experience")

        # Step 2: PREDICT — run episode
        try:
            env.load_task(task_name, var, label)
            init_obs = env.reset()

            progress, success, grounding_acc, trajectory, num_steps = run_episode(
                client, args.model_name, env, goal, init_obs,
                experience_context, args.max_steps, args.temperature,
            )
        except Exception as e:
            logger.error(f"  Episode failed: {e}")
            progress, success, grounding_acc, trajectory, num_steps = 0.0, False, 0.0, [], 0

        sr = 1.0 if success else 0.0
        all_sr.append(sr)
        all_pr.append(progress)
        all_gr.append(grounding_acc)
        difficulties.append(difficulty)

        icon = "+" if success else "x"
        logger.info(
            f"  [{icon}] SR={sr:.0f}, PR={progress:.2f}, GR={grounding_acc:.2f}, "
            f"steps={num_steps}, "
            f"running_SR={sum(all_sr)/len(all_sr):.3f}, "
            f"running_PR={sum(all_pr)/len(all_pr):.3f}"
        )

        # Step 3: EVOLVE — store experience into memory
        memory.evolve(task_name, goal, trajectory, success, progress, idx)

        # Save result incrementally
        result = {
            "id": idx,
            "task_name": task_name,
            "var": var,
            "goal": goal,
            "difficulty": difficulty,
            "success": success,
            "progress_rate": progress,
            "grounding_acc": grounding_acc,
            "num_steps": num_steps,
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
    overall_gr = sum(all_gr) / total

    hard_sr = [s for s, d in zip(all_sr, difficulties) if d == "hard"]
    hard_pr = [p for p, d in zip(all_pr, difficulties) if d == "hard"]
    easy_sr = [s for s, d in zip(all_sr, difficulties) if d == "easy"]
    easy_pr = [p for p, d in zip(all_pr, difficulties) if d == "easy"]

    logger.info(f"\n{'='*60}")
    logger.info(f"ScienceWorld Streaming Evaluation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Method:             {args.method}")
    logger.info(f"Model:              {args.model_name}")
    logger.info(f"Total tasks:        {total}")
    logger.info(f"Success Rate (S):   {overall_sr:.3f}")
    logger.info(f"Progress Rate (P):  {overall_pr:.3f}")
    logger.info(f"Grounding Acc:      {overall_gr:.3f}")
    if easy_sr:
        logger.info(f"Easy SR/PR:         {sum(easy_sr)/len(easy_sr):.3f} / {sum(easy_pr)/len(easy_pr):.3f}")
    if hard_sr:
        logger.info(f"Hard SR/PR:         {sum(hard_sr)/len(hard_sr):.3f} / {sum(hard_pr)/len(hard_pr):.3f}")
    logger.info(f"{'='*60}")

    summary = {
        "method": args.method,
        "model": args.model_name,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "total_tasks": total,
        "success_rate": overall_sr,
        "progress_rate": overall_pr,
        "grounding_acc": overall_gr,
        "easy_sr": sum(easy_sr) / len(easy_sr) if easy_sr else 0,
        "easy_pr": sum(easy_pr) / len(easy_pr) if easy_pr else 0,
        "hard_sr": sum(hard_sr) / len(hard_sr) if hard_sr else 0,
        "hard_pr": sum(hard_pr) / len(hard_pr) if hard_pr else 0,
        "timestamp": timestamp,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
