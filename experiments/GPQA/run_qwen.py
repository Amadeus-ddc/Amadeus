"""
GPQA evaluation script for local models (Qwen2.5-7B-Instruct etc.)
Uses the same prompt construction and answer parsing as the original GPQA repo,
but replaces API calls with local vLLM inference.

No modifications to the original repo code are needed.
"""

import sys
import os
import csv
import re
import json
import random
import logging
import argparse
from datetime import datetime
from collections import namedtuple

import pandas as pd
from tqdm import tqdm

# ============================================================
# Reuse the original repo's data loading and prompt construction
# by importing directly or reimplementing the pure functions.
# We reimplement to avoid touching the original code's imports
# (which pull in openai/anthropic at module level).
# ============================================================

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# ---- Data Loading (identical to repo's utils.py) ----

def load_examples(path: str, seed: int = 0):
    question_df = pd.read_csv(path)
    random.seed(seed)

    def shuffle_choices_and_create_example(row):
        list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'],
                        row['Incorrect Answer 3'], row['Correct Answer']]
        random.shuffle(list_choices)
        return Example(row.Question, list_choices[0], list_choices[1],
                       list_choices[2], list_choices[3],
                       list_choices.index(row['Correct Answer']))

    return [shuffle_choices_and_create_example(row) for _, row in question_df.iterrows()]


# ---- Prompt Construction (identical to repo's utils.py) ----

def base_prompt(example: Example) -> str:
    prompt = f"What is the correct answer to this question: {example.question}"
    prompt += f"\n\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
    return prompt


def zero_shot_prompt(example: Example) -> str:
    prompt = base_prompt(example)
    prompt += f'\n\nFormat your response as follows: "The correct answer is (insert answer here)"'
    return prompt


def chain_of_thought_prompt(example: Example, cot_examples_path: str) -> str:
    prompt = "Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.\n"
    with open(cot_examples_path, 'r') as f:
        json_data = json.load(f)
    for q in json_data["questions"]:
        prompt += f'Question: {q["question"]}\nChoices:\n'
        for choice, value in q["choices"].items():
            prompt += f'({choice}) {value}\n'
        prompt += f"Let's think step by step: \n{q['explanation']}\n"
        prompt += f'The correct answer is ({q["correct_answer"]})\n'
    prompt += f"Question: {example.question}"
    prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
    prompt += '\nGive step by step reasoning before you answer, and when you\'re ready to answer, please use the format "The correct answer is (insert answer here)":\n'
    return prompt


def five_shot_prompt(example: Example, cot_examples_path: str) -> str:
    prompt = "Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly.\n"
    with open(cot_examples_path, 'r') as f:
        json_data = json.load(f)
    for q in json_data["questions"]:
        prompt += f'Question: {q["question"]}\nChoices:\n'
        for choice, value in q["choices"].items():
            prompt += f'({choice}) {value}\n'
        prompt += f'The correct answer is ({q["correct_answer"]})\n'
    prompt += f"Question: {example.question}"
    prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
    prompt += '\nWhen you\'re ready to answer, please use the format "The correct answer is (insert answer here).'
    return prompt


def zero_shot_cot_prompt_step1(example: Example) -> str:
    """First step: generate reasoning."""
    prompt = base_prompt(example)
    prompt += "\nLet's think step by step: "
    return prompt


SKIP_QUESTION_ID = 69  # Original repo skips this question (choices too long)


def zero_shot_cot_prompt_step2(step1_prompt: str, reasoning: str) -> str:
    """Second step: extract answer after reasoning."""
    return step1_prompt + f"{reasoning}\n\nBased on the above, what is the single, most likely answer choice? Answer in the format \"The correct answer is (insert answer here)\"."


def create_prompts(examples, prompt_type, cot_examples_path):
    if prompt_type == 'zero_shot':
        return [zero_shot_prompt(ex) for ex in examples]
    elif prompt_type == 'chain_of_thought':
        return [chain_of_thought_prompt(ex, cot_examples_path) for ex in examples]
    elif prompt_type == '5_shot':
        return [five_shot_prompt(ex, cot_examples_path) for ex in examples]
    elif prompt_type == 'zero_shot_chain_of_thought':
        # This requires two-step inference, handled separately
        return [zero_shot_cot_prompt_step1(ex) for ex in examples]
    else:
        raise ValueError(f"Prompt type {prompt_type} not supported for local models.")


# ---- Answer Parsing (identical to repo's run_baseline.py) ----

def parse_sampled_answer(answer: str):
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)',
                r'answer \((.)\)', r'\((.)\)']
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match and match.group(1) in LETTER_TO_INDEX:
            return match.group(1)
    return None


# ---- vLLM Inference ----

def build_chat_messages(prompt: str, system_prompt: str = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def run_evaluation(args):
    from vllm import LLM, SamplingParams

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(args.output_dir, f"{args.prompt_type}_{args.model_name_short}_{timestamp}.log")
    csv_filename = os.path.join(args.output_dir, f"{args.prompt_type}_{args.model_name_short}_{timestamp}.csv")

    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info(f"Model: {args.model_path}")
    logging.info(f"Prompt type: {args.prompt_type}")
    logging.info(f"Data: {args.data_filename}")
    logging.info(f"Seed: {args.seed}")

    # Load data
    examples = load_examples(args.data_filename, seed=args.seed)
    if args.max_examples:
        examples = examples[:args.max_examples]
    logging.info(f"Loaded {len(examples)} examples")

    # Load model
    logging.info("Loading model...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95 if args.temperature > 0 else 1.0,
    )

    system_prompt = "You are a very intelligent assistant, who follows instructions directly."

    # Build prompts
    cot_examples_path = os.path.join(args.repo_path, "prompts", "chain_of_thought_examples.json")

    is_two_step = (args.prompt_type == 'zero_shot_chain_of_thought')

    if is_two_step:
        step1_prompts = create_prompts(examples, args.prompt_type, cot_examples_path)
        # Mark question 69 as placeholder (original repo skips it too)
        for i in range(len(step1_prompts)):
            if i == SKIP_QUESTION_ID and i < len(step1_prompts):
                step1_prompts[i] = "Choices too long"
    else:
        raw_prompts = create_prompts(examples, args.prompt_type, cot_examples_path)

    # Apply chat template
    def apply_chat_template(prompt_text):
        messages = build_chat_messages(prompt_text, system_prompt)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if is_two_step:
        # Step 1: generate reasoning
        logging.info("Zero-shot CoT Step 1: generating reasoning...")
        formatted_step1 = [apply_chat_template(p) for p in step1_prompts]
        step1_outputs = llm.generate(formatted_step1, sampling_params)
        step1_responses = [out.outputs[0].text for out in step1_outputs]

        # Step 2: extract answer
        step2_prompts = [zero_shot_cot_prompt_step2(p, r)
                         for p, r in zip(step1_prompts, step1_responses)]
        formatted_step2 = [apply_chat_template(p) for p in step2_prompts]
        outputs = llm.generate(formatted_step2, sampling_params)
        # Combine reasoning + answer for logging
        full_responses = [r1 + "\n" + out.outputs[0].text
                          for r1, out in zip(step1_responses, outputs)]
    else:
        # Single-step inference (batch)
        formatted_prompts = [apply_chat_template(p) for p in raw_prompts]
        logging.info(f"Running batch inference on {len(formatted_prompts)} prompts...")
        outputs = llm.generate(formatted_prompts, sampling_params)
        full_responses = [out.outputs[0].text for out in outputs]

    # Evaluate
    correct = 0
    refusals = 0
    total = len(examples)

    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Question id', 'Question', 'Correct answer', 'Model answer',
                            'Correct', 'Model response'])

        for qid, (example, response) in enumerate(zip(examples, full_responses)):
            # Skip question 69 (choices too long) - matches original repo behavior
            if qid == SKIP_QUESTION_ID:
                csvwriter.writerow([qid, example.question,
                                    example[example.correct_index + 1],
                                    "Couldn't find an answer choice!", False, ""])
                continue

            sampled_answer = parse_sampled_answer(response)

            if sampled_answer is None:
                refusals += 1
                csvwriter.writerow([qid, example.question,
                                    example[example.correct_index + 1],
                                    "Couldn't find an answer choice!", False, response])
                if args.verbose:
                    logging.info(f"Q{qid}: REFUSAL (no answer parsed)")
                continue

            is_correct = LETTER_TO_INDEX[sampled_answer] == example.correct_index
            if is_correct:
                correct += 1

            csvwriter.writerow([qid, example.question,
                                example[example.correct_index + 1],
                                example[LETTER_TO_INDEX[sampled_answer] + 1],
                                is_correct, response])

            if args.verbose:
                logging.info(f"Q{qid}: {'CORRECT' if is_correct else 'WRONG'} "
                             f"(predicted={sampled_answer}, "
                             f"correct_idx={example.correct_index})")

    accuracy = correct / total
    refusal_rate = refusals / total

    result_msg = (f"\n{'='*50}\n"
                  f"Results: {args.model_name_short} / {args.prompt_type}\n"
                  f"Accuracy: {accuracy:.4f} ({correct}/{total})\n"
                  f"Refusal rate: {refusal_rate:.4f} ({refusals}/{total})\n"
                  f"CSV: {csv_filename}\n"
                  f"{'='*50}")
    print(result_msg)
    logging.info(result_msg)

    # Save summary
    summary_file = os.path.join(args.output_dir, "results_summary.jsonl")
    with open(summary_file, 'a') as f:
        summary = {
            "timestamp": timestamp,
            "model": args.model_path,
            "model_short": args.model_name_short,
            "prompt_type": args.prompt_type,
            "data_file": args.data_filename,
            "seed": args.seed,
            "temperature": args.temperature,
            "accuracy": accuracy,
            "refusal_rate": refusal_rate,
            "correct": correct,
            "total": total,
            "refusals": refusals,
            "csv_file": csv_filename,
        }
        f.write(json.dumps(summary) + "\n")


def main():
    parser = argparse.ArgumentParser(description="GPQA evaluation with local models (vLLM)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local model or HuggingFace model ID")
    parser.add_argument("--model_name_short", type=str, default=None,
                        help="Short name for logging (default: derived from model_path)")
    parser.add_argument("--data_filename", type=str, required=True,
                        help="Path to GPQA CSV data file")
    parser.add_argument("--repo_path", type=str, required=True,
                        help="Path to the cloned GPQA repo (for prompt templates)")
    parser.add_argument("--prompt_type", type=str, default="zero_shot",
                        choices=["zero_shot", "chain_of_thought", "5_shot",
                                 "zero_shot_chain_of_thought"],
                        help="Prompt type for evaluation")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Max number of examples to evaluate")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for answer choice shuffling")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Max tokens for generation (original repo uses 1000)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max model context length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed per-question results")

    args = parser.parse_args()

    if args.model_name_short is None:
        args.model_name_short = os.path.basename(args.model_path.rstrip('/'))

    run_evaluation(args)


if __name__ == "__main__":
    main()
