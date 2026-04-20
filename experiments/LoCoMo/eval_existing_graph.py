import os
import sys
import json
import re
import argparse
import logging
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, '.env'))

if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE")

from amadeus_collab.core.graph import MemoryGraph
from amadeus_collab.agents.answerer import AnswererAgent
from experiments.LoCoMo.run_locomo import (
    HuggingFaceEmbedder,
    load_data_for_experiment,
    evaluate_with_llm,
    setup_logging,
)

logger = logging.getLogger("EvalExistingGraph")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--judge_model_name", type=str, default="qwen2.5-32b-instruct")
    parser.add_argument("--embedding_model", type=str, default=os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2"))
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--judge_api_base", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "experiment.log"))

    default_judge_base = os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    default_judge_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    judge_api_base = args.judge_api_base if args.judge_api_base is not None else default_judge_base
    judge_api_key = args.judge_api_key if args.judge_api_key is not None else default_judge_key

    logger.info(f"Using existing graph: {args.graph_path}")
    logger.info(f"Using local embedding model: {args.embedding_model}")
    embedder = HuggingFaceEmbedder(model_path=args.embedding_model)
    graph = MemoryGraph(args.graph_path, embedder=embedder)
    answerer = AnswererAgent(graph, model_name=args.model_name, api_base=args.api_base, api_key=args.api_key)

    experiment_data = load_data_for_experiment(args.data_file, args.sample_id)
    sample_id, _, questions = experiment_data[0]

    sample_qa_results = []
    sample_scores = []
    category_scores = {}
    category_counts = {}
    usage = {
        "api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "judge_api_calls": 0,
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_total_tokens": 0,
    }

    for q_item in questions:
        q = q_item.get('question', 'Unknown')
        category = q_item.get('category', 'Unknown')
        if category == 5:
            continue
        gt = str(q_item.get('answer', q_item.get('adversarial_answer', 'N/A')))
        try:
            pred = answerer.answer(q)
        except Exception as e:
            logger.error(f"[{sample_id}] Answerer failed: {e}")
            pred = "Error"

        is_correct, score, reason, judge_usage = evaluate_with_llm(
            q, gt, pred,
            model_name=args.judge_model_name,
            api_base=judge_api_base,
            api_key=judge_api_key,
        )
        sample_scores.append(score)
        usage["judge_api_calls"] += judge_usage.get("api_calls", 0)
        usage["judge_prompt_tokens"] += judge_usage.get("prompt_tokens", 0)
        usage["judge_completion_tokens"] += judge_usage.get("completion_tokens", 0)
        usage["judge_total_tokens"] += judge_usage.get("total_tokens", 0)

        category_scores.setdefault(category, []).append(score)
        category_counts[category] = category_counts.get(category, 0) + 1

        logger.info(
            f"\n[{sample_id}] Q: {q}\nCategory: {category}\nGT: {gt}\nPred: {pred}\n"
            f"Result: {'✅' if is_correct else '❌'} (Score: {score:.2f})\nReason: {reason}"
        )
        sample_qa_results.append({
            "question": q,
            "category": category,
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_correct,
            "score": score,
            "reason": reason,
        })

    agent_stats = getattr(answerer, "usage_stats", None) or {}
    usage["api_calls"] += agent_stats.get("api_calls", 0)
    usage["prompt_tokens"] += agent_stats.get("prompt_tokens", 0)
    usage["completion_tokens"] += agent_stats.get("completion_tokens", 0)
    usage["total_tokens"] += agent_stats.get("total_tokens", 0)

    sample_avg_score = float(np.mean(sample_scores)) if sample_scores else 0.0
    graph_stats = {
        "nodes": graph.graph.number_of_nodes(),
        "edges": graph.graph.number_of_edges(),
    }
    aggregate_metrics = {
        "overall": {
            "mean": sample_avg_score,
            "std": float(np.std(sample_scores)) if sample_scores else 0.0,
            "count": len(sample_scores),
        }
    }
    for cat, scores in category_scores.items():
        aggregate_metrics[f"category_{cat}"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "count": len(scores),
        }

    sample_result = {
        "sample_id": sample_id,
        "avg_score": sample_avg_score,
        "qa_results": sample_qa_results,
        "usage": usage,
        "graph_stats": graph_stats,
        "mode": "eval_existing_graph",
    }
    with open(os.path.join(args.output_dir, f"sample_{sample_id}.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_result, f, ensure_ascii=False, indent=2)

    summary = {
        "sample_id": sample_id,
        "aggregate_metrics": aggregate_metrics,
        "usage": usage,
        "graph_totals": graph_stats,
        "mode": "eval_existing_graph",
    }
    with open(os.path.join(args.output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\n🏆 Sample {sample_id} Score (Avg Score): {sample_avg_score * 100:.1f}%")
    logger.info(
        f"[{sample_id}] API Usage: {usage['api_calls']} calls, {usage['prompt_tokens']} prompt tokens, "
        f"{usage['completion_tokens']} completion tokens, {usage['total_tokens']} total tokens"
    )
    logger.info(
        f"[{sample_id}] Judge Usage: {usage['judge_api_calls']} calls, {usage['judge_prompt_tokens']} prompt tokens, "
        f"{usage['judge_completion_tokens']} completion tokens, {usage['judge_total_tokens']} total tokens"
    )
    logger.info(f"[{sample_id}] Graph Totals: {graph_stats['nodes']} nodes, {graph_stats['edges']} edges")
    logger.info(f"Overall Avg Score: {sample_avg_score * 100:.1f}%")
    for key, value in aggregate_metrics.items():
        if key.startswith('category_'):
            cat = key.replace('category_', '')
            logger.info(f"Category {cat}: {value['mean'] * 100:.1f}% (n={value['count']})")


if __name__ == "__main__":
    main()
