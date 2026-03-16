#!/usr/bin/env python3
"""
CLI entry point for Evo-Memory MATH-500 streaming evaluation.

Usage:
    python -m evo_memory.run_evo_eval \
        --agent none \
        --model_name qwen2.5-32b-instruct \
        --api_base http://localhost:8006/v1 \
        --data_path /data/hzy/Amadeus/amadeus/dataset/MATH/HuggingFaceH4___math-500 \
        --output_dir logs/results_evo

The framework is agent-agnostic.  To add a new memory model:
  1. Subclass evo_memory.base_agent.MemoryAgent
  2. Register it in AGENT_REGISTRY below
"""

import argparse
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
AMADEUS_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
if AMADEUS_ROOT not in sys.path:
    sys.path.insert(0, AMADEUS_ROOT)

# Offline mode
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.join(AMADEUS_ROOT, "data", "hf_cache"))

from evo_memory.evaluator import MATHStreamEvaluator


# ---------------------------------------------------------------------------
# Agent registry — extend this dict to plug in other memory models
# ---------------------------------------------------------------------------

def build_no_memory_agent(args):
    from evo_memory.no_memory_agent import NoMemoryAgent
    return NoMemoryAgent(
        model_name=args.model_name,
        api_base=args.api_base,
        api_key=args.api_key,
    )


def build_history_agent(args):
    from evo_memory.history_agent import HistoryAgent
    return HistoryAgent(
        model_name=args.model_name,
        api_base=args.api_base,
        api_key=args.api_key,
    )


def build_exprag_agent(args):
    from evo_memory.exprag_agent import ExpRAGAgent
    return ExpRAGAgent(
        model_name=args.model_name,
        api_base=args.api_base,
        api_key=args.api_key,
        embedder_model_path=args.embedder_path,
        top_k=args.top_k,
    )


AGENT_REGISTRY = {
    "none": build_no_memory_agent,
    "history": build_history_agent,
    "exprag": build_exprag_agent,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evo-Memory streaming evaluation on MATH-500"
    )

    # Agent
    parser.add_argument(
        "--agent", type=str, default="none",
        choices=list(AGENT_REGISTRY.keys()),
        help="Which memory agent to evaluate",
    )

    # Model / API
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--api_base", type=str, default="http://localhost:8006/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--embedder_path", type=str, default=None,
                        help="Path to sentence-transformer model for embeddings")

    # Data
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to MATH-500 HF dataset directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)

    # Retrieval
    parser.add_argument("--top_k", type=int, default=4,
                        help="Number of memory items to retrieve per question")

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Defaults
    if args.data_path is None:
        args.data_path = os.path.join(
            AMADEUS_ROOT, "dataset", "MATH", "HuggingFaceH4___math-500"
        )

    if args.output_dir is None:
        args.output_dir = os.path.join(EXPERIMENT_DIR, "logs", "results_evo")

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Build agent
    agent = AGENT_REGISTRY[args.agent](args)

    # Build evaluator
    evaluator = MATHStreamEvaluator(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        max_examples=args.max_examples,
    )

    # Run
    summary = evaluator.run(agent, run_tag=args.run_tag)

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
