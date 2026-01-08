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

        # 3. 处理每个 conv 子目录
        conv_subdirs = [d for d in os.listdir(run_dir) if d.startswith("conv-") and os.path.isdir(os.path.join(run_dir, d))]
        conv_subdirs.sort() # 排序以保证输出一致性
        
        if not conv_subdirs:
            logger.error("在运行目录下未找到任何 conv-XX 文件夹。")
            continue

        logger.info(f"找到 {len(conv_subdirs)} 个样本目录，开始计算...")
        
        stats_per_conv = {}
        total_tokens_sum = 0
        count = 0
        
        for conv_id in conv_subdirs:
            # 记忆图通常储存在 {sample_id}/graphs/graph_{sample_id}.json
            graph_path = os.path.join(run_dir, conv_id, "graphs", f"graph_{conv_id}.json")
            
            if not os.path.exists(graph_path):
                logger.warning(f"跳过 {conv_id}: 找不到记忆图文件 {graph_path}")
                continue
                
            try:
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                # 使用与 MemoryGraph.get_full_state 兼容的文本表示（但不截断）
                # 排除 embedding 字段，只统计文本信息
                
                # 统计节点
                node_texts = []
                for node in graph_data.get("nodes", []):
                    node_id = node.get("id", "Unknown") # NetworkX 默认 id
                    desc = node.get("description", "")
                    node_texts.append(f"{node_id}: {desc}")
                
                # 统计边
                edge_texts = []
                # 支持 "links" 或 "edges" 键名
                edges = graph_data.get("links", graph_data.get("edges", []))
                for edge in edges:
                    src = edge.get("source", "Unknown")
                    tgt = edge.get("target", "Unknown")
                    rel = edge.get("relation", "")
                    ts = edge.get("timestamp")
                    ts_str = f" [Time: {ts}]" if ts else ""
                    edge_texts.append(f"{src} --{rel}{ts_str}--> {tgt}")
                
                # 构建用于 token 统计的完整文本
                full_text = "Nodes:\n" + "\n".join(node_texts) + "\nEdges:\n" + "\n".join(edge_texts)
                
                if tokenizer:
                    tokens = tokenizer.encode(full_text)
                    token_count = len(tokens)
                else:
                    # 简单回退：按空格切分单词
                    token_count = len(full_text.split())
                
                stats_per_conv[conv_id] = token_count
                total_tokens_sum += token_count
                count += 1
                logger.info(f"Sample {conv_id}: {token_count} tokens")
                
            except Exception as e:
                logger.error(f"处理 {conv_id} 时出错: {e}")

        # 4. 计算平均值并输出结果
        average_tokens = total_tokens_sum / count if count > 0 else 0
        
        output_result = {
            "run_directory": latest_run,
            "tokenizer_used": str(tokenizer.name_or_path) if tokenizer else "Simple Whitespace Split",
            "sample_count": count,
            "tokens_per_sample": stats_per_conv,
            "average_tokens": round(average_tokens, 2)
        }
        
        # 输出到 run 目录下的 json 文件
        output_json_path = os.path.join(run_dir, "memory_token_stats.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"\n{'='*30}")
        logger.info(f"分析完成: {latest_run}")
        logger.info(f"处理样本数: {count}")
        logger.info(f"平均 Token 数: {average_tokens:.2f}")
        logger.info(f"结果已保存至: {output_json_path}")
        logger.info(f"{'='*30}")

if __name__ == "__main__":
    main()
