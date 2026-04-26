#!/usr/bin/env python3
"""Small ALFWorld action-parser diagnostic for qwen3 runs."""

import argparse
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path

import ray
from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AMADEUS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
WORKSPACE_ROOT = os.path.dirname(AMADEUS_ROOT)
sys.path.insert(0, WORKSPACE_ROOT)

load_dotenv(os.path.join(AMADEUS_ROOT, ".env"))
load_dotenv(os.path.join(AMADEUS_ROOT, "experiments", ".env"))
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE")

VERL_AGENT_ROOT = os.path.join(WORKSPACE_ROOT, "verl-agent")
sys.path.insert(0, VERL_AGENT_ROOT)

import types
for pkg_name in ["agent_system", "agent_system.environments"]:
    if pkg_name not in sys.modules:
        stub = types.ModuleType(pkg_name)
        stub.__path__ = [os.path.join(VERL_AGENT_ROOT, pkg_name.replace(".", "/"))]
        stub.__package__ = pkg_name
        sys.modules[pkg_name] = stub

from agent_system.environments.env_package.alfworld.envs import build_alfworld_envs
from run_alfworld_offline import ACTION_PROMPT_TEMPLATE, EXAMPLE_DEMONSTRATIONS, TASKS, extract_action, format_admissible_actions, is_valid_action
from methods.amadeus import AmadeusMemory

logger = logging.getLogger("Qwen3ActionDiagnostic")


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "diagnose.log", encoding="utf-8")],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def loose_extract_action(raw: str):
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("action:"):
            action = line[len("action:"):].strip()
            if action:
                return action, True
    m = re.search(r"Action\s*:\s*([^\n]+)", raw, flags=re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip(), True
    return "", False


def detect_task(initial_obs: str, gamefile: str):
    marker = "Your task is to: "
    idx = initial_obs.find(marker)
    task_desc = initial_obs[idx + len(marker):].strip() if idx != -1 else ""
    task_type = "unknown"
    for t in TASKS:
        if t in (gamefile or ""):
            task_type = t
            break
    return task_type, task_desc


def unwrap_info(info):
    info = dict(info)
    for k in list(info.keys()):
        if isinstance(info[k], (list, tuple)) and info[k]:
            info[k] = info[k][0]
    return info


def call_model(client, model_name, messages, max_tokens, disable_thinking):
    kwargs = dict(model=model_name, messages=messages, temperature=0.0, max_tokens=max_tokens)
    if disable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}
    return client.chat.completions.create(**kwargs).choices[0].message.content.strip()


def run_episode(client, model_name, envs, env_idx, initial_obs, gamefile, experience_context, max_steps, history_length, parser_mode, disable_thinking, out_f):
    task_type, task_desc = detect_task(initial_obs, gamefile)
    current_obs = initial_obs
    current_admissible = list(envs.get_admissible_commands[env_idx] or [])
    recent_history = []
    active_experiences = experience_context
    max_progress = 0.0
    success = False
    steps = []

    for step in range(max_steps):
        admissible = current_admissible
        admissible_str = format_admissible_actions(admissible)
        experience_section = f"\n==================================================\nRELEVANT EXPERIENCE\n==================================================\n{active_experiences}\n" if active_experiences else "\n"
        prompt = ACTION_PROMPT_TEMPLATE.format(
            example_demonstrations=EXAMPLE_DEMONSTRATIONS,
            experience_section=experience_section,
            task_description=task_desc,
            recent_history="\n".join(recent_history[-history_length:]) if recent_history else "(none)",
            current_observation=current_obs,
            admissible_actions=admissible_str,
        )

        action_str = None
        think_count = 0
        messages = [{"role": "user", "content": prompt}]
        raw_attempts = []
        parse_attempts = []

        while action_str is None and think_count < 3:
            raw = call_model(client, model_name, messages, 512, disable_thinking)
            first_line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
            strict_has = first_line.lower().startswith("action:")
            strict_action, strict_found = extract_action(raw)
            loose_action, loose_found = loose_extract_action(raw)
            raw_attempts.append(raw)
            parse_attempts.append({
                "think_count": think_count,
                "first_line": first_line,
                "strict_first_line_action": strict_has,
                "strict_extract_action": strict_action,
                "strict_extract_found": strict_found,
                "loose_action": loose_action,
                "loose_found": loose_found,
            })

            if parser_mode == "loose" and loose_found:
                action_str = loose_action
            elif parser_mode == "strict" and strict_has:
                action_str = strict_action
            elif first_line.lower().startswith("think-prune:") and active_experiences:
                think_count += 1
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Now respond with Think or Action."})
            else:
                think_count += 1
                think_text = raw[len("think:"):].strip() if raw.lower().startswith("think:") else raw
                recent_history.append(f"Think: {think_text}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Now execute an action. Respond with:\nAction: <exact action from admissible actions list>"})

        fallback = False
        if action_str is None:
            action_str = "look"
            fallback = True

        action_to_take = action_str
        retry_records = []
        retry_count = 0
        while not is_valid_action(action_to_take, admissible) and retry_count < 2:
            retry_count += 1
            retry_prompt = (
                "Your previous action was not in the admissible actions list and was NOT executed.\n"
                "Choose exactly one action by copying it verbatim from this admissible actions list.\n"
                f"Admissible actions: {format_admissible_actions(admissible)}\n"
                "Respond with only:\nAction: <exact admissible action>"
            )
            retry_raw = call_model(
                client,
                model_name,
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"Action: {action_to_take}"},
                    {"role": "user", "content": retry_prompt},
                ],
                128,
                disable_thinking,
            )
            retry_action, retry_found = loose_extract_action(retry_raw)
            retry_records.append({"raw": retry_raw, "found": retry_found, "action": retry_action})
            action_to_take = retry_action if retry_found else retry_raw.strip()

        if not is_valid_action(action_to_take, admissible):
            action_to_take = "look"
            fallback = True

        obs, scores, dones, info = ray.get(envs.workers[env_idx].step.remote(action_to_take))
        obs = obs[0] if isinstance(obs, (list, tuple)) else obs
        done = dones[0] if isinstance(dones, (list, tuple)) else dones
        info = unwrap_info(info)
        new_adm = info.get("admissible_commands")
        if new_adm is not None:
            current_admissible = list(new_adm) if isinstance(new_adm, (list, tuple)) else [new_adm]
        gcsr = info.get("extra.goal_condition_success_rate")
        if gcsr is not None:
            try:
                max_progress = max(max_progress, float(gcsr))
            except (TypeError, ValueError):
                pass
        if done:
            success = bool(info.get("won", False))
            if success:
                max_progress = 1.0

        record = {
            "env_idx": env_idx,
            "task_type": task_type,
            "task_description": task_desc,
            "step": step,
            "parser_mode": parser_mode,
            "disable_thinking": disable_thinking,
            "fallback_to_look": fallback,
            "raw_attempts": raw_attempts,
            "parse_attempts": parse_attempts,
            "chosen_action_raw": action_str,
            "chosen_action_exec": action_to_take,
            "retry_records": retry_records,
            "action_in_admissible": action_to_take in admissible,
            "admissible_count": len(admissible),
            "admissible_sample": admissible[:30],
            "done": bool(done),
            "won": bool(info.get("won", False)),
            "progress": max_progress,
            "observation": str(obs)[:1000],
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        logger.info(
            "env=%s step=%s parser=%s fallback=%s action=%r exec=%r in_adm=%s done=%s won=%s progress=%.2f first=%r",
            env_idx, step, parser_mode, fallback, action_str, action_to_take, action_to_take in admissible,
            bool(done), bool(info.get("won", False)), max_progress,
            parse_attempts[0]["first_line"] if parse_attempts else "",
        )

        steps.append(record)
        recent_history.append(f"Action: {action_to_take}\nObservation: {obs}")
        current_obs = obs
        if done:
            break

    return {"success": success, "progress": max_progress, "num_steps": len(steps), "task_type": task_type, "task_description": task_desc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--source_graph_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--env_num", type=int, default=134)
    parser.add_argument("--sample_ids", default="0,1,3,5")
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--history_length", type=int, default=10)
    parser.add_argument("--parser_mode", choices=["strict", "loose"], default="strict")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--embedding_model", default=str(Path(AMADEUS_ROOT) / "models" / "all-MiniLM-L6-v2"))
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(SCRIPT_DIR) / "logs" / f"diagnose_qwen3_actions_{args.parser_mode}_{timestamp}"
    setup_logging(output_dir)

    graph_src = Path(args.source_graph_dir) / "memory_graph.json"
    graph_dst = output_dir / "memory_graph.json"
    graph_dst.write_text(graph_src.read_text(), encoding="utf-8")

    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    logger.info("output=%s model=%s api_base=%s parser=%s disable_thinking=%s", output_dir, args.model_name, api_base, args.parser_mode, args.disable_thinking)

    memory = AmadeusMemory(output_dir=str(output_dir), embedding_model=args.embedding_model, model_name=args.model_name, api_base=api_base, api_key=api_key)
    client = OpenAI(base_url=api_base, api_key=api_key)

    if not os.environ.get("ALFWORLD_DATA"):
        os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")

    if not ray.is_initialized():
        alfworld_pythonpath = os.path.join(VERL_AGENT_ROOT, "agent_system", "environments", "env_package", "alfworld")
        ray.init(runtime_env={"env_vars": {"PYTHONPATH": f"{alfworld_pythonpath}:{VERL_AGENT_ROOT}", "ALFWORLD_DATA": os.environ["ALFWORLD_DATA"]}})

    alf_config_path = os.path.join(VERL_AGENT_ROOT, "agent_system/environments/env_package/alfworld/configs/config_tw.yaml")
    envs = build_alfworld_envs(
        alf_config_path,
        seed=42,
        env_num=args.env_num,
        group_n=1,
        is_train=False,
        env_kwargs={"eval_dataset": "eval_in_distribution"},
        resources_per_worker={"num_cpus": 0.05, "num_gpus": 0.0},
    )
    text_obs_list, _, infos = envs.reset()

    sample_ids = [int(x) for x in args.sample_ids.split(",") if x.strip()]
    summaries = []
    with open(output_dir / "steps.jsonl", "w", encoding="utf-8") as out_f:
        for env_idx in sample_ids:
            gamefile = infos[env_idx].get("extra.gamefile", "") or ""
            if isinstance(gamefile, list):
                gamefile = gamefile[0] if gamefile else ""
            task_type, task_desc = detect_task(text_obs_list[env_idx], gamefile)
            experience_context = memory.search(task_desc)
            logger.info("START env=%s task=%s desc=%s exp_chars=%s", env_idx, task_type, task_desc[:100], len(experience_context or ""))
            summary = run_episode(
                client, args.model_name, envs, env_idx, text_obs_list[env_idx], gamefile, experience_context,
                args.max_steps, args.history_length, args.parser_mode, args.disable_thinking, out_f,
            )
            summaries.append({"env_idx": env_idx, **summary})
            logger.info("END env=%s success=%s progress=%.2f steps=%s", env_idx, summary["success"], summary["progress"], summary["num_steps"])

    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    envs.close()
    ray.shutdown()
    logger.info("done: %s", output_dir)


if __name__ == "__main__":
    main()
