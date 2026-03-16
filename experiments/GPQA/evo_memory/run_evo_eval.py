#!/usr/bin/env python3
"""
CLI entry point for Evo-Memory GPQA streaming evaluation.

Usage:
    python run_evo_eval.py \
        --agent amadeus \
        --model_name qwen2.5-7b-instruct \
        --api_base http://localhost:8006/v1 \
        --data_path repo/dataset/gpqa_diamond.csv \
        --output_dir results_evo

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
AMADEUS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if AMADEUS_ROOT not in sys.path:
    sys.path.insert(0, AMADEUS_ROOT)

from evo_memory.evaluator import GPQAStreamEvaluator


# ---------------------------------------------------------------------------
# Agent registry — extend this dict to plug in other memory models
# ---------------------------------------------------------------------------

def build_amadeus_agent(args):
    from evo_memory.amadeus_agent import AmadeusMemoryAgent
    return AmadeusMemoryAgent(
        model_name=args.model_name,
        api_base=args.api_base,
        api_key=args.api_key,
        graph_storage_path=os.path.join(args.output_dir, "memory_graph.json"),
        embedder_model_path=args.embedder_path,
        top_k_retrieve=args.top_k,
        enable_self_play=args.self_play,
        self_play_mode=args.self_play_mode,
        self_play_questions=args.self_play_questions,
        use_cot=args.use_cot,
        buffer_size=args.buffer_size,
    )


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


def build_remem_agent(args):
    from evo_memory.remem_agent import ReMemAgent
    return ReMemAgent(
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
    "remem": build_remem_agent,
    "amadeus": build_amadeus_agent,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evo-Memory streaming evaluation on GPQA"
    )

    # Agent
    parser.add_argument(
        "--agent", type=str, default="amadeus",
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
                        help="Path to GPQA CSV. Default: repo/dataset/gpqa_diamond.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)

    # Retrieval
    parser.add_argument("--top_k", type=int, default=4,
                        help="Number of memory items to retrieve per question")

    # Self-play (Questioner vs Answerer adversarial optimization)
    parser.add_argument("--self_play", action="store_true", default=True,
                        help="Enable Questioner-Answerer-Optimizer self-play loop (default: on)")
    parser.add_argument("--no_self_play", dest="self_play", action="store_false",
                        help="Disable self-play (Builder-only memory, no adversarial optimization)")
    parser.add_argument("--self_play_mode", type=str, default="adaptive_buffer_fixed_sp",
                        choices=["none", "adaptive_buffer_fixed_sp",
                                 "fixed_buffer_adaptive_sp", "fixed_buffer_fixed_sp_cot"],
                        help="Self-play ablation mode (same as run_locomo.py)")
    parser.add_argument("--self_play_questions", type=int, default=3,
                        help="Number of self-play questions per feedback round (fixed_sp modes)")
    parser.add_argument("--use_cot", action="store_true",
                        help="Use Chain-of-Thought evaluation in Optimizer (slower but finer-grained)")
    parser.add_argument("--buffer_size", type=int, default=3,
                        help="Number of experiences to accumulate before flushing to Builder (default: 3)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Defaults
    if args.data_path is None:
        args.data_path = os.path.join(SCRIPT_DIR, "repo", "dataset", "gpqa_diamond.csv")

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "results_evo")

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
    evaluator = GPQAStreamEvaluator(
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
