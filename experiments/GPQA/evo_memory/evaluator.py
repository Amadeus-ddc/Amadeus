"""
Evo-Memory streaming evaluator for GPQA.

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
import random
import time
from collections import namedtuple
from datetime import datetime
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from .base_agent import MemoryAgent

logger = logging.getLogger("EvoMemory.Evaluator")

# Re-use GPQA data structures
LETTER = ["A", "B", "C", "D"]
LETTER_TO_IDX = {l: i for i, l in enumerate(LETTER)}

Example = namedtuple(
    "Example", ["question", "choice1", "choice2", "choice3", "choice4", "correct_index"]
)


# ---------------------------------------------------------------------------
# Data loading (identical logic to original GPQA repo)
# ---------------------------------------------------------------------------

def load_examples(path: str, seed: int = 0) -> List[Example]:
    df = pd.read_csv(path)
    random.seed(seed)

    def make(row):
        answers = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
            row["Correct Answer"],
        ]
        random.shuffle(answers)
        correct_idx = answers.index(row["Correct Answer"])
        return Example(row.Question, answers[0], answers[1], answers[2], answers[3], correct_idx)

    return [make(row) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class GPQAStreamEvaluator:
    """
    Evo-Memory style streaming evaluator for GPQA.

    Usage:
        evaluator = GPQAStreamEvaluator(data_path, output_dir)
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

        self.examples = load_examples(data_path, seed=seed)
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

        logger.info(f"=== Evo-Memory GPQA Evaluation ===")
        logger.info(f"Agent:    {agent.name}")
        logger.info(f"Data:     {self.data_path}")
        logger.info(f"Examples: {len(self.examples)}")
        logger.info(f"Seed:     {self.seed}")

        # Reset agent memory for a clean run
        agent.reset()

        correct = 0
        refusals = 0
        total = len(self.examples)
        per_question: List[dict] = []
        cumulative_acc: List[float] = []

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "qid", "question", "correct_answer", "model_answer",
                "is_correct", "cumulative_accuracy",
            ])

            for qid, ex in enumerate(tqdm(self.examples, desc=f"Eval {agent.name}")):
                choices = [ex.choice1, ex.choice2, ex.choice3, ex.choice4]
                correct_letter = LETTER[ex.correct_index]

                # --- SEARCH + SYNTHESIS: agent answers ---
                t0 = time.time()
                predicted_letter = agent.answer(ex.question, choices)
                elapsed = time.time() - t0

                # Validate
                if predicted_letter not in LETTER_TO_IDX:
                    logger.warning(f"Q{qid}: invalid answer '{predicted_letter}', treating as refusal")
                    refusals += 1
                    predicted_letter = "A"  # fallback

                is_correct = (predicted_letter == correct_letter)
                if is_correct:
                    correct += 1

                cum_acc = correct / (qid + 1)
                cumulative_acc.append(cum_acc)

                # --- EVOLVE: agent receives feedback ---
                agent.receive_feedback(
                    question=ex.question,
                    choices=choices,
                    model_answer=predicted_letter,
                    is_correct=is_correct,
                )

                # Log
                status = "CORRECT" if is_correct else "WRONG"
                logger.info(
                    f"Q{qid}: {status}  pred={predicted_letter} gt={correct_letter}  "
                    f"cum_acc={cum_acc:.3f}  time={elapsed:.1f}s"
                )

                writer.writerow([
                    qid, ex.question,
                    f"({correct_letter}) {choices[ex.correct_index]}",
                    f"({predicted_letter}) {choices[LETTER_TO_IDX[predicted_letter]]}",
                    is_correct, f"{cum_acc:.4f}",
                ])

                per_question.append({
                    "qid": qid,
                    "predicted": predicted_letter,
                    "correct": correct_letter,
                    "is_correct": is_correct,
                    "cumulative_accuracy": cum_acc,
                    "time_s": round(elapsed, 2),
                })

        accuracy = correct / total if total > 0 else 0.0

        # Final flush for any remaining buffered state
        if hasattr(agent, 'finalize'):
            agent.finalize()

        summary = {
            "timestamp": timestamp,
            "agent": agent.name,
            "data_file": self.data_path,
            "seed": self.seed,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "refusals": refusals,
            "csv_file": csv_path,
        }

        # Print results
        result_msg = (
            f"\n{'=' * 55}\n"
            f"  Evo-Memory GPQA Results: {agent.name}\n"
            f"  Accuracy: {accuracy:.4f}  ({correct}/{total})\n"
            f"  Refusals: {refusals}\n"
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
