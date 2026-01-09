import os
import sys
import json
import logging
from transformers import AutoTokenizer

# 设置日志
def setup_logging(log_path):
    # 清除旧的 handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger("TokenAnalyzer")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", help="Specific run directories to analyze")
    args = parser.parse_args()

    # 1. 路径自动识别
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_base_dir = os.path.join(script_dir, "logs")

    if args.run_dirs:
        run_paths = [os.path.abspath(d) for d in args.run_dirs]
    else:
        if not os.path.exists(logs_base_dir):
            print(f"错误: 找不到日志目录 {logs_base_dir}")
            return
        runs = [d for d in os.listdir(logs_base_dir) if os.path.isdir(os.path.join(logs_base_dir, d))]
        runs = [d for d in runs if d.startswith("run_") or d.startswith("results_")]
        if not runs:
            print("错误: 在 logs 目录下找不到任何实验运行文件夹。")
            return
        runs.sort(key=lambda x: os.path.getmtime(os.path.join(logs_base_dir, x)), reverse=True)
        run_paths = [os.path.join(logs_base_dir, runs[0])]

    for run_dir in run_paths:
        if not os.path.exists(run_dir):
            print(f"跳过不存在的目录: {run_dir}")
            continue
            
        latest_run = os.path.basename(run_dir)
        logger = setup_logging(os.path.join(run_dir, "token_analysis.log"))
        logger.info(f"正在分析目录: {latest_run}")

        # 2. 初始化 Tokenizer
        # 实验中默认使用的是 Qwen2.5 家族模型，这里优先尝试加载 Qwen2.5 的分词器
        model_name = "Qwen/Qwen2.5-7B-Instruct" 
        logger.info(f"正在加载分词器: {model_name}")
        try:
            # 尝试本地加载或从 cache 加载
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"无法在线或从缓存加载 Qwen 分词器: {e}")
            logger.info("尝试搜索本地模型目录...")
            # 尝试寻找 workspace 下的 models 目录
            base_dir = os.path.dirname(os.path.dirname(script_dir))
            local_model_path = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                logger.info(f"成功加载本地分词器: {local_model_path}")
            except:
                logger.error("无法加载任何分词器，将回退到简单的单词计数（可能不准确）。")
                tokenizer = None

        # 3. 处理记忆图文件
        # 兼容两种结构：
        # A: run_dir/conv-XX/graphs/graph_conv-XX.json
        # B: run_dir/graph_conv-XX.json (并行运行时的扁平结构)
        
        graph_files = []
        
        # 检查扁平结构
        flat_graphs = [f for f in os.listdir(run_dir) if f.startswith("graph_conv-") and f.endswith(".json")]
        if flat_graphs:
            logger.info(f"检测到扁平结构，找到 {len(flat_graphs)} 个记忆图文件")
            for f in flat_graphs:
                conv_id = f.replace("graph_", "").replace(".json", "")
                graph_files.append((conv_id, os.path.join(run_dir, f)))
        else:
            # 检查子目录结构
            conv_subdirs = [d for d in os.listdir(run_dir) if d.startswith("conv-") and os.path.isdir(os.path.join(run_dir, d))]
            if conv_subdirs:
                logger.info(f"检测到子目录结构，找到 {len(conv_subdirs)} 个样本目录")
                for conv_id in conv_subdirs:
                    graph_path = os.path.join(run_dir, conv_id, "graphs", f"graph_{conv_id}.json")
                    if os.path.exists(graph_path):
                        graph_files.append((conv_id, graph_path))
        
        graph_files.sort() # 排序以保证输出一致性
        
        if not graph_files:
            logger.error("在运行目录下未找到任何有效的记忆图文件或 conv-XX 文件夹。")
            continue

        logger.info(f"开始计算 {len(graph_files)} 个样本的 Token 数...")
        
        stats_per_conv = {}
        total_tokens_sum = 0
        count = 0
        
        for conv_id, graph_path in graph_files:
            try:
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)