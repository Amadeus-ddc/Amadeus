import sys
import os
import json
import re
import logging
import argparse
import datetime
import shutil
import time
import random
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from openai import OpenAI

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE")

from amadeus.core.graph import MemoryGraph
from amadeus.core.buffer import TimeWindowBuffer
from amadeus.agents.builder import BuilderAgent
from amadeus.agents.answerer import AnswererAgent
from amadeus.agents.questioner import QuestionerAgent
from amadeus.engine.optimizer import AdversarialOptimizer

class HuggingFaceEmbedder:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def embed(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return embeddings[0].cpu().numpy()
        except Exception as e:
            print(f"Embedding failed: {e}")
            return None

def setup_logging(log_path=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode='a', encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

def get_sample_logger(sample_id, log_path):
    logger = logging.getLogger(f"Sample_{sample_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Clear existing handlers to avoid duplication if re-used
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

logger = logging.getLogger("Experiment")

def parse_sample_entry(target_data):
    """
    解析单个样本数据，返回 chunks 和 qa_pairs
    """
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    chunks = []
    
    # 1. 处理 Conversation Sessions (对话)
    if 'conversation' in target_data:
        conv = target_data['conversation']
        session_keys = [k for k in conv.keys() if k.startswith('session_') and isinstance(conv[k], list)]
        session_keys.sort(key=natural_sort_key)
        
        for sk in session_keys:
            # 1. Try to get explicit date_time
            date_key = f"{sk}_date_time"
            time_context = "Unknown Date"
            
            if date_key in conv and conv[date_key]:
                time_context = conv[date_key]
            else:
                # Fallback to summary if date is missing
                summary_key = f"{sk}_summary"
                if summary_key in conv and isinstance(conv[summary_key], str):
                    time_context = conv[summary_key][:150]
            
            # 构造 Chunk: 上下文 + 对话内容
            # 加上 Speaker 前缀解决实体混淆
            chunk_text = f"--- Session Context: {time_context} ---\n"
            for turn in conv[sk]:
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                chunk_text += f"{speaker}: {text}\n"
            
            chunks.append(chunk_text)
    
    # 2. 处理 Source Documents (D1, D2...) - 通常是日记或新闻
    exclude_keys = {'qa', 'sample_id', 'id', 'category', 'dataset', 'conversation'}
    other_keys = [k for k in target_data.keys() if k not in exclude_keys and isinstance(target_data[k], str)]
    other_keys.sort(key=natural_sort_key)
    
    for k in other_keys:
        # 将 Key (如 "D1: 2023-05-01") 作为时间上下文
        chunk_text = f"--- Source/Date Context: {k} ---\n{target_data[k]}"
        chunks.append(chunk_text)

    qa_pairs = target_data.get('qa', [])
    return chunks, qa_pairs

def load_data_for_experiment(path: str, target_id: str = None):
    """
    加载实验数据。如果指定 target_id, 只返回该样本: 否则返回所有样本。
    支持逗号分隔的多个ID，例如 "conv-26,conv-27"
    返回: List of (sample_id, chunks, qa_pairs)
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ 数据文件未找到: {path}")
        sys.exit(1)
    
    results = []
    found = False
    
    # 解析 target_id 列表
    target_ids = set()
    if target_id and target_id != "all":
        target_ids = {tid.strip() for tid in target_id.split(',')}

    for entry in dataset:
        sid = entry.get('sample_id')
        
        # 过滤逻辑
        if target_ids and sid not in target_ids:
            continue
            
        found = True
        chunks, qa_pairs = parse_sample_entry(entry)
        results.append((sid, chunks, qa_pairs))
        
    if not found and target_id:
        logger.error(f"❌ 未在数据集中找到 ID: {target_id}")
        sys.exit(1)
        
    logger.info(f"✅ 成功加载 {len(results)} 个样本")
    return results

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

def evaluate_with_llm(question, ground_truth, prediction, model_name="qwen2.5-32b-instruct", api_base=None, api_key=None, logger=None):
    client = OpenAI(base_url=api_base, api_key=api_key)
    
    max_retries = 3
    content = None
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": ACCURACY_PROMPT.format(
                        question=question, gold_answer=ground_truth, generated_answer=prediction
                    ),
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
    except Exception as e:
        if logger:
            logger.error(f"Evaluation LLM Call Failed: {e}")
        return False

    if not content:
        return False
    
    try:
        
        def extract_json(text):
            text = text.strip()
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = text
            return json_str

        json_str = extract_json(content)
        result = json.loads(json_str)
        label = result.get("label", "WRONG")
        
        # Try to capture reasoning if the model provides it in the JSON (though prompt is ambiguous)
        reason = result.get("reason", result.get("reasoning", result.get("explanation", "")))
        
        is_correct = (label == "CORRECT")
        score = 1.0 if is_correct else 0.0
        
        return is_correct, score, reason
    except Exception as e:
        (logger or logging.getLogger("Experiment")).error(f"LLM Judge failed: {e}")
        return False, 0.0, str(e)

def process_sample(sample_data, args, embedder, judge_api_base, judge_api_key, run_output_dir):
    sample_id, chunks, questions = sample_data
    
    # Create sample directory
    sample_dir = os.path.join(run_output_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Setup Sample Logger
    sample_logger = get_sample_logger(sample_id, os.path.join(sample_dir, "execution.log"))
    sample_logger.info(f"\n{'='*40}\n🚀 Running Sample: {sample_id}\n{'='*40}")
    
    # Graph Path inside sample directory
    graph_path = os.path.join(sample_dir, "graph.json")
    if os.path.exists(graph_path): os.remove(graph_path)
        
    graph = MemoryGraph(graph_path, embedder=embedder)
    buffer = TimeWindowBuffer(trigger_threshold=1) # 每一个Session都是完整上下文，直接触发
    
    # Pass logger to agents
    builder = BuilderAgent(graph, model_name=args.model_name, logger=sample_logger)
    answerer = AnswererAgent(graph, model_name=args.model_name, logger=sample_logger)
    questioner = QuestionerAgent(model_name=args.model_name, logger=sample_logger)
    optimizer = AdversarialOptimizer(questioner, builder, answerer, model_name=args.model_name, logger=sample_logger)

    sample_logger.info(f"[{sample_id}] 🧠 Phase 1: Building Memory ({len(chunks)} contextual sessions)...")
    
    # Adaptive Buffer Logic
    current_buffer = ""
    
    # Ablation Settings
    ablation_mode = args.ablation_mode
    fixed_buffer_size = args.fixed_buffer_size
    fixed_sp_count = args.fixed_sp_count
    
    # Determine Optimizer Mode
    optimizer_mode = "adaptive"
    optimizer_fixed_count = None
    use_cot = False
    enable_optimizer = True
    
    if ablation_mode == "no_self_play":
        enable_optimizer = False
    elif ablation_mode == "adaptive_buffer_fixed_sp":
        optimizer_mode = "fixed"
        optimizer_fixed_count = fixed_sp_count
    elif ablation_mode == "fixed_buffer_adaptive_sp":
        optimizer_mode = "adaptive"
    elif ablation_mode == "fixed_buffer_fixed_sp_cot":
        optimizer_mode = "fixed"
        optimizer_fixed_count = fixed_sp_count
        use_cot = True
    
    # Re-implementing loop to match original logic structure but with ablation support
    current_buffer = ""
    chunks_since_flush = 0
    
    for i, chunk in enumerate(chunks):
        if current_buffer == "":
            current_buffer += chunk
            chunks_since_flush = 1
        else:
            # Check flush condition
            should_flush = False
            if ablation_mode == "fixed_buffer_adaptive_sp" or ablation_mode == "fixed_buffer_fixed_sp_cot":
                if chunks_since_flush >= fixed_buffer_size:
                    should_flush = True
            else:
                should_flush = builder.check_flush_condition(current_buffer, chunk)
            
            if should_flush:
                sample_logger.info(f"[{sample_id}] 🔄 Flush Triggered at chunk {i}. Processing Buffer...")
                kept_items, action_log = builder.process_buffer(current_buffer)
                
                if enable_optimizer and graph.graph.number_of_nodes() > 0:
                    try:
                        optimizer.step(current_buffer, action_log, mode=optimizer_mode, fixed_loops=optimizer_fixed_count, use_cot=use_cot)
                    except Exception as e:
                        sample_logger.warning(f"[{sample_id}] Optimizer step failed (skipping): {e}")
                
                current_buffer = chunk
                chunks_since_flush = 1
            else:
                current_buffer += "\n" + chunk
                chunks_since_flush += 1
    
    # 3. Final Flush for remaining content
    if current_buffer:
        sample_logger.info(f"[{sample_id}] 🔄 Final Flush...")
        kept_items, action_log = builder.process_buffer(current_buffer)
        if enable_optimizer and graph.graph.number_of_nodes() > 0:
            try:
                optimizer.step(current_buffer, action_log, mode=optimizer_mode, fixed_loops=optimizer_fixed_count, use_cot=use_cot)
            except Exception as e:
                sample_logger.warning(f"[{sample_id}] Optimizer step failed (skipping): {e}")

    sample_logger.info(f"[{sample_id}] 📊 Graph Ready. Nodes: {graph.graph.number_of_nodes()}, Edges: {graph.graph.number_of_edges()}")

    # Save graph to output dir (Already saved to sample_dir/graph.json by MemoryGraph, but let's ensure)
    # MemoryGraph saves to self.storage_file which is graph_path = sample_dir/graph.json
    # So we don't need to copy it anymore, but let's keep a log
    sample_logger.info(f"[{sample_id}] 💾 Graph saved to: {graph_path}")

    sample_logger.info(f"[{sample_id}] 🔍 Phase 2: Evaluation...")
    
    sample_qa_results = []
    sample_scores = []
    
    local_category_scores = {}
    local_category_counts = {}
    local_total_questions = 0

    for q_item in questions:
        q = q_item.get('question', 'Unknown')
        category = q_item.get('category', 'Unknown')
        
        # Skip Category 5 to match baseline
        if category == 5:
            continue

        # 优先使用 'answer'，如果没有则尝试 'adversarial_answer'，最后才是 'N/A'
        gt = str(q_item.get('answer', q_item.get('adversarial_answer', 'N/A')))
        
        try:
            pred = answerer.answer(q)
        except Exception as e:
            sample_logger.error(f"[{sample_id}] Answerer failed: {e}")
            pred = "Error"
        
        is_correct, score, reason = evaluate_with_llm(
            q, gt, pred, 
            model_name=args.judge_model_name,
            api_base=judge_api_base,
            api_key=judge_api_key,
            logger=sample_logger
        )
        
        # Update stats
        sample_scores.append(score)
        local_total_questions += 1
        
        # Update category stats
        if category not in local_category_scores:
            local_category_scores[category] = []
            local_category_counts[category] = 0
        local_category_scores[category].append(score)
        local_category_counts[category] += 1
        
        icon = "✅" if is_correct else "❌"
        
        sample_logger.info(f"\n[{sample_id}] Q: {q}\nCategory: {category}\nGT: {gt}\nPred: {pred}\nResult: {icon} (Score: {score:.2f})\nReason: {reason}")
        
        sample_qa_results.append({
            "question": q,
            "category": category,
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": is_correct,
            "score": score,
            "reason": reason
        })

    sample_avg_score = np.mean(sample_scores) if sample_scores else 0.0
    sample_logger.info(f"\n🏆 Sample {sample_id} Score (Avg Score): {sample_avg_score * 100:.1f}%")
    
    # Save sample result
    sample_result = {
        "sample_id": sample_id,
        "avg_score": float(sample_avg_score),
        "qa_results": sample_qa_results
    }
    
    with open(os.path.join(sample_dir, "result.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_result, f, ensure_ascii=False, indent=2)
        
    return {
        "sample_result": sample_result,
        "category_scores": local_category_scores,
        "category_counts": local_category_counts,
        "total_questions": local_total_questions
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(base_dir, 'data', 'locomo10.json')
    default_log = os.path.join(base_dir, 'log', 'locomo2.log')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default=default_data)
    parser.add_argument("--sample_id", type=str, default="conv-26")
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--judge_model_name", type=str, default="qwen2.5-32b-instruct")
    parser.add_argument("--log_path", type=str, default=default_log)
    parser.add_argument("--judge_api_base", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    # Default to the local path found in the workspace
    parser.add_argument("--embedding_model", type=str, default="/home/ubuntu/hzy/crl/Amadeus/amadeus/models/all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default="results_locomo")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of parallel workers")
    
    # Ablation Arguments
    parser.add_argument("--ablation_mode", type=str, default="none", 
                        choices=["none", "fixed_buffer_adaptive_sp", "adaptive_buffer_fixed_sp", "fixed_buffer_fixed_sp_cot", "no_self_play"],
                        help="Ablation study mode")
    parser.add_argument("--fixed_buffer_size", type=int, default=3, help="Number of chunks for fixed buffer size")
    parser.add_argument("--fixed_sp_count", type=int, default=3, help="Number of questions for fixed self-play")
    
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Update log path to be inside the output directory if default
    if args.log_path == default_log:
        args.log_path = os.path.join(run_output_dir, "experiment.log")

    setup_logging(args.log_path)

    # Capture default env vars for Judge before they might be overwritten by CLI args for the main model
    # Priority for Judge: CLI args > JUDGE_ env vars > OPENAI_ env vars (from .env)
    default_judge_base = os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    default_judge_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if args.api_base: os.environ["OPENAI_BASE_URL"] = args.api_base
    if args.api_key: os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Resolve final Judge config
    judge_api_base = args.judge_api_base if args.judge_api_base is not None else default_judge_base
    judge_api_key = args.judge_api_key if args.judge_api_key is not None else default_judge_key
    
    logger.info(f"Using local embedding model: {args.embedding_model}")
    embedder = HuggingFaceEmbedder(model_path=args.embedding_model)

    DATA_FILE = args.data_file
    TARGET_ID = args.sample_id
    
    # 加载数据 (支持单个或全部)
    experiment_data = load_data_for_experiment(DATA_FILE, TARGET_ID if TARGET_ID != "all" else None)
    
    total_samples = len(experiment_data)
    
    # Statistics containers
    all_sample_results = []
    category_scores = {}
    category_counts = {}
    total_questions = 0
    
    logger.info(f"🚀 Starting parallel execution with {args.max_workers} workers for {total_samples} samples.")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_sample, item, args, embedder, judge_api_base, judge_api_key, run_output_dir)
            for item in experiment_data
        ]
        
        for future in as_completed(futures):
            try:
                res = future.result()
                
                # Aggregate results
                all_sample_results.append(res["sample_result"])
                total_questions += res["total_questions"]
                
                for cat, scores in res["category_scores"].items():
                    if cat not in category_scores:
                        category_scores[cat] = []
                        category_counts[cat] = 0
                    category_scores[cat].extend(scores)
                    category_counts[cat] += res["category_counts"][cat]
                    
            except Exception as e:
                logger.error(f"❌ A sample failed to process: {e}", exc_info=True)

    # Calculate aggregate metrics
    aggregate_results = {"overall": {}}
    all_scores = [q["score"] for s in all_sample_results for q in s["qa_results"]]
    
    if all_scores:
        aggregate_results["overall"] = {
            "mean": float(np.mean(all_scores)),
            "std": float(np.std(all_scores)),
            "count": len(all_scores)
        }
        
    for cat, scores in category_scores.items():
        aggregate_results[f"category_{cat}"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "count": len(scores)
        }

    # Final Summary
    final_summary = {
        "timestamp": timestamp,
        "config": {
            "model_name": args.model_name,
            "judge_model_name": args.judge_model_name,
            "data_file": args.data_file,
        },
        "total_samples": total_samples,
        "total_questions": total_questions,
        "category_distribution": category_counts,
        "aggregate_metrics": aggregate_results
    }

    with open(os.path.join(run_output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*40}\n🌟 FINAL EXPERIMENT RESULT\nTotal Samples: {total_samples}\nTotal Questions: {total_questions}")
    logger.info(f"Overall Avg Score: {aggregate_results['overall'].get('mean', 0) * 100:.1f}%")
    
    logger.info("\nCategory Breakdown:")
    for cat, metrics in aggregate_results.items():
        if cat.startswith("category_"):
            cat_name = cat.replace("category_", "")
            logger.info(f"  Category {cat_name}: {metrics['mean'] * 100:.1f}% (n={metrics['count']})")
            
    logger.info(f"\nResults saved to: {run_output_dir}\n{'='*40}")

if __name__ == "__main__":
    main()