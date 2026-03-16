#!/usr/bin/env python3
"""
Standalone ScienceWorld evaluation script for reproducing
the naive-mode baseline from the EMPO2/Agent-Lightning paper.

Target: Qwen2.5-7B-Instruct, naive mode, single prompt_type,
        average return ~-61.3 across ScienceWorld tasks.

This script replicates the exact evaluation loop from:
  agent-lightning/contrib/recipes/envs/
including prompt templates, instruction formatting, action extraction,
and reward accumulation.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import OpenAI
from scienceworld import ScienceWorldEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# Data paths (relative to this script's location)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DATA_DIR = SCRIPT_DIR / "agl-envs" / "task_data" / "scienceworld"
SPLIT_SETS_DIR = TASK_DATA_DIR / "split_sets"

# ============================================================
# Prompt templates — exact copies from agl_envs/scienceworld/__init__.py
# ============================================================

NAIVE_INSTRUCTION = (
    "Please response with only one line with one sentence, "
    "following the possible action format shown above. No extra words are allowed."
)


def get_single_prompt_template(mission: str):
    """Return (template_without_history, template_with_history) for single-mode prompts."""
    template_wo_his = (
        "You are an expert agent operating in the ScienceWorld environment, "
        "which is a text-based virtual environment centered around accomplishing "
        "tasks from the elementary science curriculum.\n"
        f"Your current task is: {mission}\n\n"
        "Your current observation is: {current_observation}\n\n"
        "Current available actions:\n"
        "{admissible_actions}"
    )

    template = (
        "You are an expert agent operating in the ScienceWorld environment, "
        "which is a text-based virtual environment centered around accomplishing "
        "tasks from the elementary science curriculum.\n"
        f"Your current task is: {mission}\n\n"
        "Prior to this step, you have already taken {step_count} step(s). "
        "Below are the most recent {history_length} observations and the "
        "corresponding actions you took: {history}\n"
        "You are now at step {current_step} and your current observation is: "
        "{current_observation}\n\n"
        "Current available actions:\n"
        "{admissible_actions}"
    )

    return template_wo_his, template


# ============================================================
# Prompt builder — mirrors prompt_builder.py from the original repo
# ============================================================

class HistoryPromptBuilder:
    """Builds single-mode prompts with history tracking."""

    def __init__(self, max_history: int = 2):
        self.max_history = max_history
        self._events = []
        self.admissible_actions = None
        self.step_count = 1
        self.template_wo_his = None
        self.template = None

    def init(self, mission: str):
        self._events.clear()
        self.step_count = 1
        self.template_wo_his, self.template = get_single_prompt_template(mission)

    def update_observation(self, obs: str):
        self._events.append({"type": "observation", "text": obs})

    def update_action(self, action: str):
        self._events.append({"type": "action", "action": action})

    def update_admissible_actions(self, hint: str):
        self.admissible_actions = hint

    def update_step_count(self):
        self.step_count += 1

    def get_prompt(self) -> str:
        if self.max_history != -1:
            events = self._events[-(self.max_history * 2 + 1):]
        else:
            events = self._events

        current_obs = events[-1]["text"]

        if len(events) == 1:
            # No history
            kwargs = {"current_observation": current_obs}
            template = self.template_wo_his
            if "{admissible_actions}" in template:
                kwargs["admissible_actions"] = self.admissible_actions
            return template.format(**kwargs)
        else:
            # With history
            history = ""
            obs_count = 0
            for idx, event in enumerate(events):
                if event["type"] == "observation" and idx != len(events) - 1:
                    next_event = events[idx + 1]
                    step_num = max(self.step_count - self.max_history + obs_count, 1)
                    history += (
                        f"[Observation {step_num}: '{event['text']}', "
                        f"Action {step_num}: '{next_event['action']}']\n "
                    )
                    obs_count += 1

            kwargs = {
                "step_count": self.step_count - 1,
                "history_length": min(self.step_count - 1, self.max_history),
                "history": history,
                "current_step": self.step_count,
                "current_observation": current_obs,
            }
            if "{admissible_actions}" in self.template:
                kwargs["admissible_actions"] = self.admissible_actions
            return self.template.format(**kwargs)


# ============================================================
# ScienceWorld environment wrapper — mirrors base.py
# ============================================================

def build_simplification_str():
    """Build the simplification string matching the original parse_args defaults."""
    # From scienceworld_env.py parse_args: simplifications_preset="easy"
    # open_containers=True, open_doors=True, others False
    return "easy"


def jaccard_similarity(s1, s2):
    set1, set2 = set(s1.split()), set(s2.split())
    if len(set1 | set2) == 0:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


class SciWorldEnvWrapper:
    """Wraps ScienceWorldEnv to match the original base.py SciWorldEnv."""

    def __init__(self, env: ScienceWorldEnv, use_action_correction: bool = False):
        self.env = env
        self.step_count = 0
        self.success = False
        self.max_steps = 100
        self.use_action_correction = use_action_correction
        self.available_actions_hint = ""
        self.sub_task_name = ""

    def set_max_steps(self, sub_task_name: str, max_steps_dict: dict):
        self.sub_task_name = sub_task_name
        if sub_task_name in max_steps_dict:
            self.max_steps = max_steps_dict[sub_task_name]
        else:
            self.max_steps = 100

    def reset(self):
        observation, info = self.env.reset()
        self.step_count = 0
        self.success = False

        valid_actions = self.env.get_possible_actions()
        valid_objs = self.env.get_possible_objects()
        self.available_actions_hint = (
            f"Valid_actions: {valid_actions}, OBJ needs to be replaced with "
            f"one of the following objects: {valid_objs}\n example: focus on door"
        )
        return observation, info

    def step(self, action: str):
        if self.use_action_correction:
            validActions = self.env.get_valid_action_object_combinations_with_templates()
            if action not in [va["action"] for va in validActions]:
                sim_list = []
                for valid_action in validActions:
                    sim_list.append(jaccard_similarity(valid_action["action"], action))
                if sim_list:
                    highest_idx = np.argmax(sim_list)
                    action = validActions[highest_idx]["action"]

        observation, reward, done, info = self.env.step(action)

        valid_actions = self.env.get_possible_actions()
        valid_objs = self.env.get_possible_objects()
        self.available_actions_hint = (
            f"Valid_actions: {valid_actions}, OBJ needs to be replaced with "
            f"one of the following objects: {valid_objs}\n example: focus on door"
        )

        self.success = done and info["score"] > 70

        if self.step_count >= self.max_steps:
            done = True
        self.step_count += 1

        return observation, reward, done, info

    def get_success_score(self):
        return 1.0 if self.success else 0.0

    def close(self):
        self.env.close()


# ============================================================
# Action extraction for naive mode — mirrors base.py extract_action
# ============================================================

def extract_action_naive(llm_output: str):
    """
    For naive mode (use_reasoning=False), the entire output is the action.
    Matches base.py extract_action with use_reasoning=False:
        action = llm_output
        action_valid = True
    But we also check for Chinese characters (invalid).
    """
    llm_output = llm_output.lower()
    action = llm_output
    is_valid = True

    # Check for Chinese characters
    if re.search(r"[\u4e00-\u9fff]", llm_output):
        is_valid = False

    return action, is_valid


# ============================================================
# Main evaluation loop
# ============================================================

def run_episode(
    client: OpenAI,
    model_name: str,
    env_wrapper: SciWorldEnvWrapper,
    mission: str,
    initial_obs: str,
    max_history: int = 2,
    temperature: float = 0.4,
    use_success_rate: bool = True,
    max_tokens: int = 256,
):
    """
    Run a single ScienceWorld episode in naive+single mode.
    The environment should already be reset and initial_obs provided.
    Returns (episode_reward, final_score, success, steps, valid_action_count).
    """
    prompt_builder = HistoryPromptBuilder(max_history=max_history)
    prompt_builder.init(mission)

    # Use the already-reset observation (no second reset!)
    prompt_builder.update_observation(initial_obs)
    prompt_builder.update_admissible_actions(env_wrapper.available_actions_hint)

    episode_reward = 0.0
    done = False
    step_count = 0
    valid_action_count = 0
    final_score = 0.0

    while not done:
        # Build prompt
        prompt_text = prompt_builder.get_prompt()
        # Append naive instruction
        instructed_prompt = prompt_text + "\n\n" + NAIVE_INSTRUCTION

        # Call LLM via OpenAI-compatible API
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": instructed_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            llm_output = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed at step {step_count}: {e}")
            break

        # Extract action (naive mode)
        action, is_valid = extract_action_naive(llm_output)
        if is_valid:
            valid_action_count += 1

        # Step environment
        obs, reward, done, info = env_wrapper.step(action)
        final_score = info.get("score", 0.0)

        # Compute reward
        if use_success_rate:
            step_reward = env_wrapper.get_success_score()
        else:
            step_reward = reward

        episode_reward += float(step_reward)

        # Update prompt builder
        prompt_builder.update_step_count()
        prompt_builder.update_action(action)
        prompt_builder.update_observation(obs)
        prompt_builder.update_admissible_actions(env_wrapper.available_actions_hint)

        step_count += 1

    success = env_wrapper.get_success_score() > 0.5
    return episode_reward, final_score, success, step_count, valid_action_count


def load_test_episodes():
    """Load test episodes from the multi_data/test.parquet file.

    This matches the original evaluation protocol which uses:
      data.val_files: agl_envs/task_data/scienceworld/multi_data/test.parquet
    with 144 episodes (29 tasks x ~5 variations each), all with max_steps=30.
    """
    import pandas as pd
    test_parquet = TASK_DATA_DIR / "multi_data" / "test.parquet"
    df = pd.read_parquet(test_parquet)
    episodes = []
    for _, row in df.iterrows():
        episodes.append({
            "sub_task_name": row["sub_task_name"],
            "variation_idx": int(row["variation_idx"]),
            "max_steps": int(row["max_steps"]),
        })
    return episodes


def main():
    parser = argparse.ArgumentParser(description="ScienceWorld evaluation in naive mode")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name/path for vLLM")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                        help="vLLM server API base URL")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (0.4 for eval, matching original)")
    parser.add_argument("--max-history", type=int, default=2,
                        help="Max history steps in prompt")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens for LLM response")
    parser.add_argument("--use-success-rate", action="store_true", default=False,
                        help="Use binary success rate as reward (original uses True for training)")
    parser.add_argument("--use-action-correction", action="store_true", default=False,
                        help="Use action correction via Jaccard similarity")
    parser.add_argument("--task-ids", type=str, default=None,
                        help="Comma-separated task IDs to evaluate (default: all)")
    parser.add_argument("--max-variations", type=int, default=-1,
                        help="Max variations per task (-1 = all)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from existing results")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load task metadata
    with open(SPLIT_SETS_DIR / "id2taskname.json") as f:
        id2taskname = json.load(f)
    # Build reverse mapping: taskname -> task_id
    taskname2id = {v: int(k) for k, v in id2taskname.items()}

    # Load test episodes from parquet (matching original evaluation protocol)
    test_episodes = load_test_episodes()

    # Filter by task_ids if specified
    if args.task_ids:
        allowed_names = set()
        for tid_str in args.task_ids.split(","):
            tid = int(tid_str)
            if str(tid) in id2taskname:
                allowed_names.add(id2taskname[str(tid)])
        test_episodes = [ep for ep in test_episodes if ep["sub_task_name"] in allowed_names]

    # Limit variations per task if requested
    if args.max_variations > 0:
        from collections import Counter
        task_count = Counter()
        filtered = []
        for ep in test_episodes:
            if task_count[ep["sub_task_name"]] < args.max_variations:
                filtered.append(ep)
                task_count[ep["sub_task_name"]] += 1
        test_episodes = filtered

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup OpenAI client (pointing to vLLM server)
    client = OpenAI(base_url=args.api_base, api_key="token-abc123")

    # Determine model name for API calls
    model_name = args.model

    # Load existing results if resuming
    results_file = output_dir / "results.jsonl"
    completed_episodes = set()
    if args.resume and results_file.exists():
        with open(results_file) as f:
            for line in f:
                entry = json.loads(line)
                key = (entry["task_name"], entry["variation_idx"])
                completed_episodes.add(key)
        logger.info(f"Resuming: found {len(completed_episodes)} completed episodes")

    # Run evaluation
    all_results = []
    total_episodes = len(test_episodes)
    simplification_str = build_simplification_str()

    logger.info(f"Starting evaluation: {total_episodes} episodes")

    for episode_idx, episode in enumerate(test_episodes, 1):
        task_name = episode["sub_task_name"]
        var_idx = episode["variation_idx"]
        max_steps = episode["max_steps"]
        task_id = taskname2id.get(task_name, -1)

        # Skip if already completed
        if (task_name, var_idx) in completed_episodes:
            logger.info(f"  [{episode_idx}/{total_episodes}] Skipping {task_name} var {var_idx} (already done)")
            continue

        logger.info(f"  [{episode_idx}/{total_episodes}] {task_name} var {var_idx} (max_steps={max_steps})")

        # Create environment
        try:
            sciworld_env = ScienceWorldEnv("", envStepLimit=200)
            env_wrapper = SciWorldEnvWrapper(
                sciworld_env,
                use_action_correction=args.use_action_correction,
            )

            # Load the specific task + variation
            sciworld_env.load(
                task_name, var_idx,
                simplificationStr=simplification_str,
                generateGoldPath=True,
            )
            # Use max_steps from parquet data (30 for all tasks in test set)
            env_wrapper.max_steps = max_steps
            env_wrapper.sub_task_name = task_name

            # Reset environment — only call reset ONCE (matching original flow)
            obs, info = env_wrapper.reset()
            mission = info["taskDesc"].split("Task Description:\n")[-1]

            # Run episode
            episode_reward, final_score, success, steps, valid_actions = run_episode(
                client=client,
                model_name=model_name,
                env_wrapper=env_wrapper,
                mission=mission,
                initial_obs=obs,
                max_history=args.max_history,
                temperature=args.temperature,
                use_success_rate=args.use_success_rate,
                max_tokens=args.max_tokens,
            )

            result = {
                "task_id": task_id,
                "task_name": task_name,
                "variation_idx": var_idx,
                "episode_reward": episode_reward,
                "final_score": final_score,
                "success": success,
                "steps": steps,
                "valid_actions": valid_actions,
            }
            all_results.append(result)

            logger.info(
                f"    reward={episode_reward:.2f}, score={final_score:.1f}, "
                f"success={success}, steps={steps}, valid={valid_actions}/{steps}"
            )

            # Save incrementally
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        except Exception as e:
            logger.error(f"    ERROR in {task_name} var {var_idx}: {e}")
            result = {
                "task_id": task_id,
                "task_name": task_name,
                "variation_idx": var_idx,
                "episode_reward": 0.0,
                "final_score": 0.0,
                "success": False,
                "steps": 0,
                "valid_actions": 0,
                "error": str(e),
            }
            all_results.append(result)
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")
        finally:
            try:
                env_wrapper.close()
            except Exception:
                pass

    # ============================================================
    # Final summary
    # ============================================================
    if all_results:
        # Load all results including resumed ones
        if args.resume and results_file.exists():
            all_results = []
            with open(results_file) as f:
                for line in f:
                    all_results.append(json.loads(line))

        # Per-task summary
        task_summaries = defaultdict(list)
        for r in all_results:
            task_summaries[r["task_id"]].append(r)

        summary = {}
        logger.info("\n" + "=" * 70)
        logger.info("FINAL RESULTS")
        logger.info("=" * 70)
        logger.info(f"{'Task ID':<8} {'Task Name':<45} {'Avg Reward':>10} {'Avg Score':>10} {'Success':>8} {'N':>4}")
        logger.info("-" * 70)

        task_avg_rewards = []
        task_avg_scores = []
        task_success_rates = []

        for tid in sorted(task_summaries.keys()):
            results = task_summaries[tid]
            name = id2taskname.get(str(tid), "unknown")
            avg_reward = np.mean([r["episode_reward"] for r in results])
            avg_score = np.mean([r["final_score"] for r in results])
            sr = np.mean([float(r["success"]) for r in results])
            n = len(results)

            task_avg_rewards.append(avg_reward)
            task_avg_scores.append(avg_score)
            task_success_rates.append(sr)

            logger.info(f"{tid:<8} {name:<45} {avg_reward:>10.2f} {avg_score:>10.1f} {sr:>8.3f} {n:>4}")

            summary[tid] = {
                "task_name": name,
                "avg_reward": avg_reward,
                "avg_score": avg_score,
                "success_rate": sr,
                "n_episodes": n,
            }

        logger.info("-" * 70)
        overall_avg_reward = np.mean(task_avg_rewards)
        overall_avg_score = np.mean(task_avg_scores)
        overall_sr = np.mean(task_success_rates)
        logger.info(
            f"{'OVERALL':<8} {'(task-level average)':<45} "
            f"{overall_avg_reward:>10.2f} {overall_avg_score:>10.1f} {overall_sr:>8.3f}"
        )
        logger.info("=" * 70)

        # Save summary
        summary_data = {
            "model": args.model,
            "temperature": args.temperature,
            "max_history": args.max_history,
            "mode": "naive",
            "prompt_type": "single",
            "use_success_rate": args.use_success_rate,
            "overall_avg_reward": overall_avg_reward,
            "overall_avg_score": overall_avg_score,
            "overall_normalized_return": overall_avg_score - 100,
            "overall_success_rate": overall_sr,
            "total_episodes": len(all_results),
            "per_task": summary,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"\nResults saved to {output_dir}")
        # Paper metric interpretation: "average return" of -61.3 likely means
        # avg_score - 100 (ScienceWorld score ranges 0-100, shifted to -100 to 0)
        # OR it could be the raw cumulative step reward when use_success_rate=False
        normalized_return = overall_avg_score - 100
        logger.info(f"Target metric (paper): average return = -61.3")
        logger.info(f"Our result:")
        logger.info(f"  Avg score (0-100):        {overall_avg_score:.2f}")
        logger.info(f"  Avg return (score - 100): {normalized_return:.2f}")
        logger.info(f"  Avg cumulative reward:    {overall_avg_reward:.2f}")
        logger.info(f"  Avg success rate:         {overall_sr:.3f}")


if __name__ == "__main__":
    main()
