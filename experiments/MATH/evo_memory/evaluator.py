"""
Evo-Memory streaming evaluator for MATH-500.

Implements the evaluation protocol from:
  "Evo-Memory: Benchmarking LLM Agent Test-time Learning
   with Self-Evolving Memory" (Wei et al., 2025)

Core loop for each question t in a fixed sequential order:
  1. Search   — agent retrieves relevant memories
  2. Synthesis — agent generates answer  y_hat_t
  3. Evaluate — compare with ground truth -> f_t (correct / wrong)
  4. Evolve   — agent receives f_t and updates its memory

Feedback is a *correctness signal only* — the correct answer is NOT revealed.
"""

import csv
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import List, Optional

from datasets import load_dataset
from tqdm import tqdm

from .base_agent import MemoryAgent

logger = logging.getLogger("EvoMemory.Evaluator")


# ---------------------------------------------------------------------------
# Answer extraction / matching (from run_math500.py, proven correct)
# ---------------------------------------------------------------------------

def extract_answer_from_response(response: str) -> str:
    """Extract answer from model response — try \\boxed{}, then 'The correct answer is'."""
    if not response:
        return ""
    # Priority 1: \\boxed{} — most reliable for math
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
    # Priority 2: "The correct answer is ..." — grab everything after, possibly multiline
    m = re.search(
        r"(?:correct answer is|Final Answer[:\s]*(?:The correct answer is)?)\s*[:\s]*([\s\S]+?)(?:\n\n|\Z)",
        response, re.IGNORECASE,
    )
    if m:
        ans = m.group(1).strip().rstrip(".")
        # Strip surrounding $, \[, \] for display math
        ans = re.sub(r"^\\\[", "", ans)
        ans = re.sub(r"\\\]$", "", ans)
        ans = ans.strip().strip("$").strip()
        if ans:
            return ans
    # Priority 3: last line fallback
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return response.strip()


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


def extract_gold_answer(item: dict) -> str:
    """Extract gold answer from a MATH-500 dataset item.

    The dataset has an 'answer' column; fallback to \\boxed{} in 'solution'.
    """
    if "answer" in item and item["answer"]:
        return item["answer"].strip()
    solution = item.get("solution", "")
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


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MATHStreamEvaluator:
    """
    Evo-Memory style streaming evaluator for MATH-500.

    Usage:
        evaluator = MATHStreamEvaluator(data_path, output_dir)
        evaluator.run(agent)
    """

    def __init__(
        self,
        data_path: str,
        output_dir: str = "results",
        seed: int = 0,
        max_examples: Optional[int] = None,
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.seed = seed
        self.max_examples = max_examples

        # Load dataset from local path
        self.dataset = load_dataset(data_path, split="test")
        self.examples = list(self.dataset)
        if max_examples:
            self.examples = self.examples[:max_examples]

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def run(self, agent: MemoryAgent, run_tag: str = None) -> dict:
        """
        Run the full streaming evaluation.

        Args:
            agent:   A MemoryAgent instance.
            run_tag: Optional tag for the output files. Defaults to agent.name + timestamp.

        Returns:
            Summary dict with accuracy, per-question results, etc.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tag = run_tag or f"{agent.name}_{timestamp}"

        csv_path = os.path.join(self.output_dir, f"evo_{tag}.csv")
        log_path = os.path.join(self.output_dir, f"evo_{tag}.log")
        summary_path = os.path.join(self.output_dir, "evo_results_summary.jsonl")

        # Setup file logging for this run
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(file_handler)

        logger.info(f"=== Evo-Memory MATH-500 Evaluation ===")
        logger.info(f"Agent:    {agent.name}")
        logger.info(f"Data:     {self.data_path}")
        logger.info(f"Examples: {len(self.examples)}")
        logger.info(f"Seed:     {self.seed}")

        # Reset agent memory for a clean run
        agent.reset()

        correct = 0
        total = len(self.examples)
        per_question: List[dict] = []
        cumulative_acc: List[float] = []

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "qid", "question", "gold_answer", "pred_answer",
                "is_correct", "cumulative_accuracy",
            ])

            for qid, item in enumerate(tqdm(self.examples, desc=f"Eval {agent.name}")):
                question = item["problem"]
                gold_answer = extract_gold_answer(item)

                # --- SEARCH + SYNTHESIS: agent answers ---
                t0 = time.time()
                pred_answer = agent.answer(question)
                elapsed = time.time() - t0

                # Also try extracting from the raw response if agent cached it
                raw_response = getattr(agent, "_last_raw_response", pred_answer)
                extracted = extract_answer_from_response(raw_response)
                if extracted:
                    pred_answer = extracted

                # --- EVALUATE ---
                is_correct = answers_match(pred_answer, gold_answer)
                if is_correct:
                    correct += 1

                cum_acc = correct / (qid + 1)
                cumulative_acc.append(cum_acc)

                # --- EVOLVE: agent receives feedback ---
                agent.receive_feedback(
                    question=question,
                    model_answer=pred_answer,
                    is_correct=is_correct,
                )

                # Log
                status = "CORRECT" if is_correct else "WRONG"
                logger.info(
                    f"Q{qid}: {status}  pred={pred_answer[:60]}  gold={gold_answer[:60]}  "
                    f"cum_acc={cum_acc:.3f}  time={elapsed:.1f}s"
                )

                writer.writerow([
                    qid, question[:200],
                    gold_answer,
                    pred_answer,
                    is_correct, f"{cum_acc:.4f}",
                ])

                per_question.append({
                    "qid": qid,
                    "predicted": pred_answer,
                    "gold": gold_answer,
                    "is_correct": is_correct,
                    "cumulative_accuracy": cum_acc,
                    "time_s": round(elapsed, 2),
                })

        accuracy = correct / total if total > 0 else 0.0

        summary = {
            "timestamp": timestamp,
            "agent": agent.name,
            "data_file": self.data_path,
            "seed": self.seed,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "csv_file": csv_path,
        }

        # Print results
        result_msg = (
            f"\n{'=' * 55}\n"
            f"  Evo-Memory MATH-500 Results: {agent.name}\n"
            f"  Accuracy: {accuracy:.4f}  ({correct}/{total})\n"
            f"  CSV: {csv_path}\n"
            f"{'=' * 55}"
        )
        print(result_msg)
        logger.info(result_msg)

        # Save summary line
        with open(summary_path, "a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        # Remove file handler
        logging.getLogger().removeHandler(file_handler)

        return summary
