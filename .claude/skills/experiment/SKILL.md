---
name: experiment
description: Run and babysit Amadeus experiments, especially LoCoMo/ALFWorld, with project-specific defaults for Python envs, model paths, logs, APIs, and restart behavior.
---

# Experiment Skill

Use this skill whenever the user asks to start, rerun, monitor, resume, or debug LoCoMo/ALFWorld experiments in this repository.

## Defaults

- Repository root: `/data/hzy/Amadeus/amadeus-collab`.
- Python environment: use `/data/hzy/miniconda3/envs/amadeus1/bin/python` unless the user explicitly specifies another environment.
- If the user does not specify otherwise, use the `amadeus1` environment.
- Embedding model: `/data/hzy/Amadeus/amadeus-collab/models/all-MiniLM-L6-v2`.
- Main local model base directory: `/data/hzy/models/`.
  - The user will provide the main model name/path.
  - If they provide a model name rather than an absolute path, resolve it under `/data/hzy/models/` when launching local/vLLM-backed experiments.
- For LoCoMo judge/evaluation:
  - Load API credentials/config from `/data/hzy/Amadeus/amadeus-collab/.env`.
  - Judge model name: `qwen2.5-32b-instruct` (lowercase `b`; verify against `.env` API before launch).
  - Do not hardcode judge API keys in commands; rely on `.env` unless the user explicitly overrides.
- If the user does not specify a method, use `amadeus`.

## Log naming rule

Experiment run/log names must be distinctive and include:

1. node name or machine/GPU identifier when known,
2. a short description of the code/method change being tested,
3. timestamp.

Suggested pattern:

```text
{node_or_gpu}_{change_or_method}_{YYYYMMDD_HHMMSS}
```

Examples:

```text
gpu0_amadeus_pathfix_20260426_113012
nodeA_locomo_builder_retry_20260426_113012
```

## Launch behavior

Before starting an experiment:

1. Confirm the target task (`locomo`, `alfworld`, or both), model, GPU/port if provided, and method.
2. If not provided, default `method=amadeus` and Python env `amadeus1`.
3. Use the repository-local embedding model path above.
4. Prefer project scripts when they already express the intended experiment, but override env vars/arguments to enforce these defaults.
5. For LoCoMo, ensure judge API/model comes from `.env` and `qwen2.5-32B-instruct`.
6. Start long-running experiments with `Bash(run_in_background=true)` so they can be monitored.

## Monitoring and auto-repair loop

After starting an experiment, monitoring is mandatory. Do not ask the user whether to monitor it.

1. Immediately record the background task ID, output file path, run/log directory, command configuration, model/API ports, sample IDs, and expected next phase.
2. Schedule a one-shot check for 2 minutes after launch. This is required for every experiment launch.
3. At every check, inspect:
   - the background task status/output file,
   - the main console log and experiment log,
   - recent `ERROR`, `Traceback`, `Killed`, `OOM`, `model_not_found`, `_parse_json_content`, timeout, and completion markers,
   - whether expected result files such as `summary.json` have appeared.
4. If the task is still running normally, schedule the next one-shot check for 30 minutes later before ending the response. Never leave a running experiment without a future check scheduled.
5. If a `<task-notification>` reports `killed`, `failed`, or `completed`, immediately inspect the task output and logs in the next assistant turn. Do not respond with “No response requested” for experiment task notifications.
6. If logs show a path/environment/setup error, stop the experiment when needed, fix the path/setup issue, and restart it.
7. Only auto-fix infrastructure/path/setup issues, for example:
   - missing or wrong file paths,
   - wrong working directory,
   - wrong Python path/import path,
   - missing `ALFWORLD_DATA`, `VERL_AGENT_ROOT`, dataset, embedding path, output directory,
   - script using old migrated-server path,
   - wrong served model name or wrong local API port,
   - LoCoMo judge accidentally using the local model API instead of `.env`.
8. Do not rewrite core algorithm/model implementation bugs as part of this skill unless the user explicitly asks.
9. If the experiment is interrupted or crashes due to fixable path/setup issues, repair and resume/restart automatically.
10. Prefer resume/checkpoint options when available; otherwise restart using the same configuration.
11. If the root cause is not clear, report that uncertainty explicitly, include the evidence checked, and choose the least destructive recovery.
12. Keep the process fully automatic; do not ask the user for confirmation during the monitor/repair/restart cycle unless a destructive action or missing external asset requires user input.

## Missing external assets

If an experiment cannot continue because an external asset is genuinely absent, report the exact missing asset and path. Do not invent replacements.

Known useful fallback locations from this migration:

- LoCoMo data: `/data/hzy/Amadeus/amadeus/dataset/LoCoMo/locomo10.json`.
- ALFWorld data: `/data/hzy/Amadeus/amadeus/dataset/ALFWorld/json_2.1.1`.
- ALFWorld verl-agent: `/data/hzy/Amadeus/amadeus/experiments/verl-agent`.

## Safety

- Do not delete experiment outputs unless the user explicitly requests cleanup.
- Do not force-push, commit, or modify shared services unless explicitly asked.
- Stopping and restarting the experiment process launched by this skill is allowed as part of automatic path/setup repair.
