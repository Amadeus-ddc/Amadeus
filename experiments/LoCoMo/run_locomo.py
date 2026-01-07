import sys
import os
import json
import re
import logging
import argparse
import datetime
import shutil
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from openai import OpenAI

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(BASE_DIR))
load_dotenv(os.path.join(BASE_DIR, '.env'))

if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE")

from amadeus.code.core.graph import MemoryGraph
from amadeus.code.core.buffer import TimeWindowBuffer
from amadeus.code.agents.builder import BuilderAgent
from amadeus.code.agents.answerer import AnswererAgent
from amadeus.code.agents.questioner import QuestionerAgent
from amadeus.code.engine.optimizer import AdversarialOptimizer

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

def evaluate_with_llm(question, ground_truth, prediction, model_name="qwen2.5-32b-instruct", api_base=None, api_key=None):
    client = OpenAI(base_url=api_base, api_key=api_key)
    
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
            temperature=0.0,
        )
        content = response.choices[0].message.content
        
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
        logger.error(f"LLM Judge failed: {e}")
        return False, 0.0, str(e)

def process_sample(sample_data, args, embedder, judge_api_base, judge_api_key, run_output_dir):
    sample_id, chunks, questions = sample_data
    logger.info(f"\n{'='*40}\n🚀 Running Sample: {sample_id}\n{'='*40}")
    
    # Define Sub-directories
    # Structure: experiments/LoCoMo/logs/run_xxx/{sample_id}/graphs/
    #            experiments/LoCoMo/logs/run_xxx/{sample_id}/results/
    sample_base_dir = os.path.join(run_output_dir, sample_id)
    graphs_dir = os.path.join(sample_base_dir, "graphs")
    results_dir = os.path.join(sample_base_dir, "results")
    
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Use the graphs_dir for the working graph file directly
    graph_path = os.path.join(graphs_dir, f"graph_{sample_id}.json")
    if os.path.exists(graph_path): os.remove(graph_path)
        
    graph = MemoryGraph(graph_path, embedder=embedder)
    buffer = TimeWindowBuffer(trigger_threshold=1) # 每一个Session都是完整上下文，直接触发
    builder = BuilderAgent(graph, model_name=args.model_name)
    answerer = AnswererAgent(graph, model_name=args.model_name)
    questioner = QuestionerAgent(model_name=args.model_name)
    optimizer = AdversarialOptimizer(questioner, builder, answerer, model_name=args.model_name)

    logger.info(f"[{sample_id}] 🧠 Phase 1: Building Memory ({len(chunks)} contextual sessions)...")
    
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
    
    if ablation_mode == "adaptive_buffer_fixed_sp":
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
                logger.info(f"[{sample_id}] 🔄 Flush Triggered at chunk {i}. Processing Buffer...")
                kept_items, action_log = builder.process_buffer(current_buffer)
                
                if graph.graph.number_of_nodes() > 0:
                    try:
                        optimizer.step(current_buffer, action_log, mode=optimizer_mode, fixed_loops=optimizer_fixed_count, use_cot=use_cot)
                    except Exception as e:
                        logger.warning(f"[{sample_id}] Optimizer step failed (skipping): {e}")
                
                current_buffer = chunk
                chunks_since_flush = 1
            else:
                current_buffer += "\n" + chunk
                chunks_since_flush += 1
    
    # 3. Final Flush for remaining content
    if current_buffer:
        logger.info(f"[{sample_id}] 🔄 Final Flush...")
        kept_items, action_log = builder.process_buffer(current_buffer)
        if graph.graph.number_of_nodes() > 0:
            try:
                optimizer.step(current_buffer, action_log, mode=optimizer_mode, fixed_loops=optimizer_fixed_count, use_cot=use_cot)
            except Exception as e:
                logger.warning(f"[{sample_id}] Optimizer step failed (skipping): {e}")

    logger.info(f"[{sample_id}] 📊 Graph Ready. Nodes: {graph.graph.number_of_nodes()}, Edges: {graph.graph.number_of_edges()}")

    # Graph is already at the correct location (graphs_dir)
    logger.info(f"[{sample_id}] 💾 Graph saved to: {graph_path}")

    logger.info(f"[{sample_id}] 🔍 Phase 2: Evaluation...")
    
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
            logger.error(f"[{sample_id}] Answerer failed: {e}")
            pred = "Error"
        
        is_correct, score, reason = evaluate_with_llm(
            q, gt, pred, 
            model_name=args.judge_model_name,
            api_base=judge_api_base,
            api_key=judge_api_key
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
        
        logger.info(f"\n[{sample_id}] Q: {q}\nCategory: {category}\nGT: {gt}\nPred: {pred}\nResult: {icon} (Score: {score:.2f})\nReason: {reason}")
        
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
    logger.info(f"\n🏆 Sample {sample_id} Score (Avg Score): {sample_avg_score * 100:.1f}%")
    
    # Save sample result to results_dir
    sample_result = {
        "sample_id": sample_id,
        "avg_score": float(sample_avg_score),
        "qa_results": sample_qa_results
    }
    
    # results_dir was defined at start of function
    # Because we don't have access to results_dir variable here in this tool call block context if I assume stateless replacement
    # But wait, python scope rules. results_dir IS available if defined in the same function scope.
    # I need to make sure I access the variable 'results_dir' which I defined in previous step.
    
    # RE-DERIVING path just to be safe in case of scope confusion in user mind (but code is contiguous)
    # Actually I edited the top of the function. results_dir is in scope.
    
    result_file_path = os.path.join(run_output_dir, sample_id, "results", f"sample_{sample_id}.json")
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_result, f, ensure_ascii=False, indent=2)
        
    return {
        "sample_result": sample_result,
        "category_scores": local_category_scores,
        "category_counts": local_category_counts,
        "total_questions": local_total_questions
    }

def main():
    # Define default log base dir: experiments/LoCoMo/logs
    # BASE_DIR is .../amadeus/
    # We want .../amadeus/experiments/LoCoMo/logs
    DEFAULT_LOG_BASE = os.path.join(BASE_DIR, "experiments", "LoCoMo", "logs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default=os.path.join(BASE_DIR, "data", "locomo10.json"))
    parser.add_argument("--sample_id", type=str, default="all", help="Target sample ID or 'all' for entire dataset")
    parser.add_argument("--model_name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--judge_model_name", type=str, default="qwen2.5-32b-instruct")
    # Change default to simple filename, will be placed in run dir
    parser.add_argument("--log_path", type=str, default="experiment.log") 
    parser.add_argument("--judge_api_base", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    # Default to the local path found in the workspace
    parser.add_argument("--embedding_model", type=str, default=os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2"))
    # Base output dir is now the logs dir
    parser.add_argument("--output_base_dir", type=str, default=DEFAULT_LOG_BASE, help="Base Directory for experiment logs and outputs")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers")
    
    # Ablation Arguments
    parser.add_argument("--ablation_mode", type=str, default="none", 
                        choices=["none", "fixed_buffer_adaptive_sp", "adaptive_buffer_fixed_sp", "fixed_buffer_fixed_sp_cot"],
                        help="Ablation study mode")
    parser.add_argument("--fixed_buffer_size", type=int, default=3, help="Number of chunks for fixed buffer size")
    parser.add_argument("--fixed_sp_count", type=int, default=3, help="Number of questions for fixed self-play")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for the run directory")
    
    args = parser.parse_args()

    # Create run directory with timestamp: experiments/LoCoMo/logs/run_2026...
    if args.run_name:
        run_name = args.run_name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Still needed for summary
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    run_dir = os.path.join(args.output_base_dir, run_name)
    run_output_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    # Set up logging to experiments/LoCoMo/logs/run_2026.../experiment.log
    if not os.path.isabs(args.log_path):
        current_log_path = os.path.join(run_dir, args.log_path)
    else:
        current_log_path = args.log_path

    setup_logging(current_log_path)
    
    logger.info(f"📂 Experiment Run Directory: {run_dir}")
    logger.info(f"📝 Global Log File: {current_log_path}")

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