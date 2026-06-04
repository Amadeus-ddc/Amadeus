# Amadeus

An evolving memory system for LLM agents. Amadeus converts experience streams into a structured **MemoryGraph**, then uses adversarial self-play to discover memory gaps and improve graph-building strategies over time.

## How It Works

```text
Experience Stream (dialogue / trajectory)
  │
  ▼
┌──────────┐   schema    ┌─────────────┐
│  Buffer   │──emergence──▶  SchemaState │
│ Manager   │◀────────────│  (node/edge  │
└────┬──────┘             │   types +    │
     │                    │   rules)     │
     ▼                    └─────────────┘
┌──────────┐
│  Builder  │── schema-aware ──▶ MemoryGraph (NetworkX)
└──────────┘                         │
                                     ▼
                         ┌───────────────────────┐
                         │   Self-Play Optimizer  │
                         │  Questioner → Answerer │
                         │      → Judge           │
                         └───────────────────────┘
                                     │
                              strategy updates
                              fed back to Builder
```

1. **Buffer Manager** segments raw input into coherent chunks using LLM-based topic-shift detection.
2. **Schema Emergence** (optional) inspects early buffers and proposes graph node types, edge types, and construction rules — no hand-written ontology needed.
3. **Builder** converts each buffer into graph operations (ADD / UPDATE / DELETE / WAIT) guided by the current schema.
4. **Self-Play Optimizer** generates questions about the graph (Questioner), answers them (Answerer), judges correctness, and feeds strategy updates back to the Builder and Answerer.

## Repository Layout

```text
code/
├── agents/
│   ├── base.py          # Shared LLM-calling utilities
│   ├── builder.py       # Converts text buffers into graph operations
│   ├── answerer.py      # Retrieves and answers from the graph
│   └── questioner.py    # Generates probing questions
├── core/
│   ├── graph.py         # MemoryGraph: NetworkX-backed storage + hybrid search
│   ├── buffer.py        # Buffer accumulation and flushing
│   └── schema.py        # SchemaState: emergent node/edge types and rules
└── engine/
    └── optimizer.py     # Adversarial self-play loop

experiments/
├── LoCoMo/run_locomo.py                 # LoCoMo memory QA benchmark
├── ALFWorld/
│   ├── run_alfworld_streaming.py        # ALFWorld streaming evaluation
│   ├── run_alfworld_offline.py          # ALFWorld offline (cold-start) evaluation
│   └── methods/amadeus.py               # ALFWorld ↔ Amadeus adapter
└── verl-agent/                          # Bundled ALFWorld env wrapper (Apache 2.0, from GiGPO/verl-agent)
```

## Setup

**Requirements:** Python 3.10+, PyTorch, an OpenAI-compatible API endpoint.

### Install

```bash
pip install -e .
# Or without editable install:
pip install -r requirements.txt
```

### Configure

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_BASE_URL` | OpenAI-compatible API endpoint (e.g. local vLLM) |
| `OPENAI_API_KEY` | API key for the endpoint |
| `JUDGE_API_BASE` | *(Optional)* Separate judge endpoint for LoCoMo |
| `ALFWORLD_DATA` | Path to ALFWorld `json_2.1.1/` data directory |
| `VERL_AGENT_ROOT` | ALFWorld env wrapper (bundled at `experiments/verl-agent/`) |

### Local Assets (not tracked by git)

```text
dataset/LoCoMo/locomo10.json        # LoCoMo dataset
dataset/ALFWorld/json_2.1.1/         # ALFWorld game data
models/all-MiniLM-L6-v2/            # Sentence embedding model
```

The ALFWorld environment wrapper (`verl-agent/`) is bundled in the repo at `experiments/verl-agent/` — no extra setup needed.

## Start a Local Model Server

Example with vLLM:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/model \
  --served-model-name qwen2.5-7b \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

Then set:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=token-abc123
```

## Run LoCoMo

Quick smoke test (single conversation, 1 chunk, 1 question):

```bash
python experiments/LoCoMo/run_locomo.py \
  --sample_id conv-26 \
  --max_chunks 1 \
  --max_questions 1 \
  --no_selfplay \
  --model_name qwen2.5-7b \
  --embedding_model models/all-MiniLM-L6-v2 \
  --output_base_dir experiments/LoCoMo/logs \
  --run_name smoke_locomo
```

Full run:

```bash
python experiments/LoCoMo/run_locomo.py \
  --sample_id all \
  --model_name qwen2.5-7b \
  --embedding_model models/all-MiniLM-L6-v2 \
  --output_base_dir experiments/LoCoMo/logs \
  --run_name full_run
```

If using a separate judge model, set `JUDGE_API_BASE` / `JUDGE_API_KEY` or pass `--judge_api_base` / `--judge_api_key`.

## Run ALFWorld

### Streaming Mode

```bash
export ALFWORLD_DATA=dataset/ALFWorld
export VERL_AGENT_ROOT=experiments/verl-agent

python experiments/ALFWorld/run_alfworld_streaming.py \
  --method amadeus \
  --model_name qwen2.5-7b \
  --embedding_model models/all-MiniLM-L6-v2 \
  --env_num 1 \
  --max_tasks 5 \
  --output_dir experiments/ALFWorld/logs/streaming_run
```

### Offline Mode

Builds memory from pre-collected trajectories, then evaluates with fixed memory:

```bash
export ALFWORLD_TRAJ_FILE=dataset/ALFWorld/alfworld_format_traj.json

python experiments/ALFWorld/run_alfworld_offline.py \
  --method amadeus \
  --max_traj 5 \
  --env_num 1 \
  --model_name qwen2.5-7b \
  --embedding_model models/all-MiniLM-L6-v2 \
  --output_dir experiments/ALFWorld/logs/offline_run
```

## Key Options

### Memory System

| Flag | Description |
|------|-------------|
| `--embedding_model` | Local sentence embedding model path |
| `--selfplay_mode {adaptive,fixed}` | Optimizer self-play mode |
| `--selfplay_rounds` | Fixed self-play rounds |
| `--no_selfplay` | Disable self-play (Builder only) |
| `--use_cot` | CoT-style optimizer experience extraction |

### Schema Emergence (ALFWorld)

| Flag | Description |
|------|-------------|
| `--schema_exploration_buffer_limit` | Number of early episodes used for schema emergence (default: 3) |
| `--no_schema_emergence` | Disable schema emergence |
| `--enable_schema_replay` | Replay prior buffers after schema changes (expensive, off by default) |

### LoCoMo

| Flag | Description |
|------|-------------|
| `--sample_id` | `all`, one sample ID, or comma-separated IDs |
| `--max_workers` | Parallel sample workers |
| `--max_chunks` / `--max_questions` | Limit chunks/questions per sample for quick runs |
| `--ablation_mode` | Ablation presets for controlled experiments |

### ALFWorld

| Flag | Description |
|------|-------------|
| `--env_num` | Number of parallel ALFWorld environments |
| `--max_tasks` | Cap on evaluated tasks (streaming mode) |
| `--max_steps` | Max steps per episode |
| `--resume` | Resume from existing `results.jsonl` |
| `--eval_in_domain` / `--eval_out_domain` | Choose evaluation split |

## Output

Logs and artifacts are written to:

```text
experiments/LoCoMo/logs/<run_name>/
experiments/ALFWorld/logs/<output_dir>/
```

These directories are git-ignored. Do not commit generated logs, datasets, model weights, or `.env` files.

## License

[MIT](LICENSE)
