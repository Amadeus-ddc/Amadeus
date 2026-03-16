#!/usr/bin/env python3
"""
MATH-500 streaming evaluation — method-agnostic harness.

Follows Memory-R's evaluation protocol:
  - Dataset: HuggingFaceH4/MATH-500 (500 problems)
  - Streaming: problems processed sequentially; memory evolves after each
  - Answer matching: <answer> tags, \\boxed{}, normalization, SymPy symbolic comparison
  - Metric: Accuracy (exact match after normalization)

The memory module is PLUGGABLE — controlled by --method flag.
To add a new method, see methods/base.py for the interface.

Usage:
  # 1. Start vLLM server
  CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
      --model /data/hzy/models/Qwen2.5-7B-Instruct --port 8000 \
      --max-model-len 8192 --gpu-memory-utilization 0.9 --trust-remote-code

  # 2. Run with Amadeus memory
  python run_math500.py --method amadeus

  # 3. Run baseline (no memory)
  python run_math500.py --method none

  # 4. Add your own method:
  #    (1) Create methods/your_method.py implementing MemoryModule
  #    (2) Register in methods/__init__.py
  #    (3) python run_math500.py --method your_method
"""

import sys
import os
import json
import re
import logging
import argparse
import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from openai import OpenAI

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

logger = logging.getLogger("MATH500")

SCRIPT_DIR = Path(__file__).resolve().parent


# ── Answer Extraction / Matching (from Memory-R) ────────────────
def extract_answer_from_response(response: str) -> str:
    """Extract answer from model response — try <answer> then \\boxed{}."""
    if "<answer>" in response:
        answer = response.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    if "\\boxed{" in response:
        idx = response.rfind("\\boxed{")
        if idx != -1:
            brace_count = 0
            start = idx + len("\\boxed{")
            for i in range(start, len(response)):
                if response[i] == "{":
                    brace_count += 1
                elif response[i] == "}":
                    if brace_count == 0:
                        return response[start:i].strip()
                    brace_count -= 1
    return response.strip()


def extract_gold_answer(solution: str) -> str:
    """Extract gold answer from MATH-500 solution field (\\boxed{...})."""
    if "\\boxed{" in solution:
        idx = solution.rfind("\\boxed{")
        if idx != -1:
            brace_count = 0
            start = idx + len("\\boxed{")
            for i in range(start, len(solution)):
                if solution[i] == "{":
                    brace_count += 1
                elif solution[i] == "}":
                    if brace_count == 0:
                        return solution[start:i].strip()
                    brace_count -= 1
    return solution.strip()


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    if answer.endswith("."):
        answer = answer[:-1]
    answer = answer.replace("$", "")
    if answer.startswith("\\(") and answer.endswith("\\)"):
        answer = answer[2:-2]
    answer = answer.replace("\\(", "").replace("\\)", "")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("^\\circ", "").replace("°", "")
    if answer.startswith("\\text{") and answer.endswith("}"):
        answer = answer[6:-1]
    answer = answer.replace(" ", "")
    answer = answer.replace("\\dfrac", "\\frac")
    return answer


def try_numeric_compare(pred: str, gold: str) -> bool:
    try:
        from sympy import sympify, simplify
        pred_val = sympify(pred)
        gold_val = sympify(gold)
        if pred_val is not None and gold_val is not None:
            diff = simplify(pred_val - gold_val)
            return diff == 0
    except Exception:
        pass
    try:
        p = float(pred.replace(",", ""))
        g = float(gold.replace(",", ""))
        return abs(p - g) < 1e-5
    except Exception:
        pass
    return False


def answers_match(pred_answer: str, gold_answer: str) -> bool:
    pred_norm = normalize_answer(pred_answer)
    gold_norm = normalize_answer(gold_answer)
    if pred_norm == gold_norm:
        return True
    if try_numeric_compare(pred_norm, gold_norm):
        return True
    return False


def extract_reasoning(response: str) -> str:
    if "<reasoning>" in response:
        reasoning = response.split("<reasoning>")[-1]
        reasoning = reasoning.split("</reasoning>")[0]
        return reasoning.strip()
    return response.strip()


# ── Prompt Templates ────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem step by step.

Respond in the following format:
<reasoning>
...your step-by-step reasoning...
</reasoning>
<answer>
...your final answer (just the value, e.g., 42 or \\frac{1}{2})...
</answer>"""

SYSTEM_PROMPT_WITH_MEMORY = """You are an expert mathematician with access to a knowledge base of previously solved problems.

Below are some relevant prior solutions that may help you:
__MEMORY_CONTEXT__

Now solve the new problem step by step.

Respond in the following format:
<reasoning>
...your step-by-step reasoning...
</reasoning>
<answer>
...your final answer (just the value, e.g., 42 or \\frac{1}{2})...
</answer>"""


def solve_problem(client: OpenAI, model_name: str, problem: str,
                  memory_context: str, temperature: float = 0.0,
                  max_tokens: int = 4096) -> str:
    if memory_context:
        system = SYSTEM_PROMPT_WITH_MEMORY.replace("__MEMORY_CONTEXT__", memory_context)
    else:
        system = SYSTEM_PROMPT

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MATH-500 streaming eval — method-agnostic harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available methods:
  none      No memory (baseline)
  amadeus   Amadeus MemoryGraph (Builder + self-play)

To add a new method:
  1. Create methods/your_method.py implementing MemoryModule (see methods/base.py)
  2. Register it in methods/__init__.py METHODS dict
  3. Run: python run_math500.py --method your_method
        """,
    )

    # Harness arguments
    parser.add_argument("--method", type=str, default="none", choices=list(METHODS.keys()),
                        help="Memory method to use (default: none)")
    parser.add_argument("--model_name", type=str, default="/data/hzy/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="token-abc123")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--max_problems", type=int, default=-1,
                        help="Max problems to evaluate (-1 = all 500)")

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

    os.environ["OPENAI_BASE_URL"] = args.api_base
    os.environ["OPENAI_API_KEY"] = args.api_key

    # Load dataset
    logger.info("Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = list(dataset)
    if args.max_problems > 0:
        problems = problems[:args.max_problems]
    logger.info(f"Loaded {len(problems)} problems")

    # Init OpenAI client
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # Init memory method
    method_kwargs = vars(args).copy()
    method_kwargs["output_dir"] = str(output_dir)
    memory = load_method(args.method, **method_kwargs)
    logger.info(f"Memory module loaded: {memory.__class__.__name__}")

    # Resume support
    results_file = output_dir / "results.jsonl"
    completed = set()
    if args.resume and results_file.exists():
        with open(results_file) as f:
            for line in f:
                entry = json.loads(line)
                completed.add(entry["id"])
        logger.info(f"Resuming: {len(completed)} problems already done")

    # ── Streaming Evaluation Loop ──
    correct_count = 0
    total_count = 0

    for idx, item in enumerate(problems):
        if idx in completed:
            continue

        problem = item["problem"]
        gold_solution = item["solution"]
        gold_answer = extract_gold_answer(gold_solution)

        logger.info(f"[{idx+1}/{len(problems)}] Problem #{idx}")

        # Step 1: SEARCH — retrieve relevant prior solutions
        memory_context = memory.search(problem)
        if memory_context:
            logger.info(f"  Retrieved {len(memory_context)} chars of memory context")

        # Step 2: PREDICT — solve problem
        try:
            response = solve_problem(
                client, args.model_name, problem,
                memory_context, args.temperature, args.max_tokens,
            )
        except Exception as e:
            logger.error(f"  LLM call failed: {e}")
            response = ""

        # Step 3: Extract & match
        pred_answer = extract_answer_from_response(response)
        match = answers_match(pred_answer, gold_answer)

        if match:
            correct_count += 1
        total_count += 1

        icon = "+" if match else "x"
        logger.info(
            f"  [{icon}] pred={pred_answer[:60]}, gold={gold_answer[:60]}, "
            f"running_acc={correct_count}/{total_count} "
            f"({correct_count/total_count*100:.1f}%)"
        )

        # Step 4: EVOLVE — store experience
        reasoning = extract_reasoning(response)
        memory.evolve(problem, reasoning, pred_answer, match, idx)

        # Save result incrementally
        result = {
            "id": idx,
            "problem": problem,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "match": match,
            "response": response,
            "method": args.method,
            "memory_used": bool(memory_context),
        }
        with open(results_file, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Summary ──
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"MATH-500 Evaluation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Method:   {args.method}")
    logger.info(f"Model:    {args.model_name}")
    logger.info(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_count})")
    logger.info(f"{'='*60}")

    summary = {
        "method": args.method,
        "model": args.model_name,
        "temperature": args.temperature,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count,
        "timestamp": timestamp,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
