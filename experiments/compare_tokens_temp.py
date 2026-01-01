import json
import os
import sys
import tiktoken

# Paths
graph_path = "/home/ubuntu/hzy/crl/Amadeus/amadeus/experiments/results_locomo/run_20260101_202508/graph_conv-26.json"
data_path = "/home/ubuntu/hzy/crl/Amadeus/amadeus/experiments/data/locomo10.json"
sample_id = "conv-26"

def count_tokens(text):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def clean_graph(graph_data):
    # Remove embeddings from nodes
    if "nodes" in graph_data:
        for node in graph_data["nodes"]:
            if "embedding" in node:
                del node["embedding"]
    return json.dumps(graph_data, ensure_ascii=False, indent=2)

def get_original_text(data_path, target_id):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for entry in data:
        if entry.get('sample_id') == target_id:
            # Reconstruct text logic from run_locomo.py
            chunks = []
            if 'conversation' in entry:
                conv = entry['conversation']
                # Sort keys
                import re
                def natural_sort_key(s):
                    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
                
                session_keys = [k for k in conv.keys() if k.startswith('session_') and isinstance(conv[k], list)]
                session_keys.sort(key=natural_sort_key)
                
                for sk in session_keys:
                    date_key = f"{sk}_date_time"
                    time_context = conv.get(date_key, "Unknown Date")
                    if not time_context:
                         summary_key = f"{sk}_summary"
                         if summary_key in conv:
                             time_context = conv[summary_key][:150]

                    chunk_text = f"--- Session Context: {time_context} ---\n"
                    for turn in conv[sk]:
                        speaker = turn.get('speaker', 'Unknown')
                        text = turn.get('text', '')
                        chunk_text += f"{speaker}: {text}\n"
                    chunks.append(chunk_text)
            
            # Source docs
            exclude_keys = {'qa', 'sample_id', 'id', 'category', 'dataset', 'conversation'}
            other_keys = [k for k in entry.keys() if k not in exclude_keys and isinstance(entry[k], str)]
            other_keys.sort(key=natural_sort_key)
            for k in other_keys:
                chunk_text = f"--- Source/Date Context: {k} ---\n{entry[k]}"
                chunks.append(chunk_text)
                
            return "\n".join(chunks)
    return None

def main():
    # 1. Process Graph
    if not os.path.exists(graph_path):
        print(f"Graph file not found: {graph_path}")
        return

    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    cleaned_graph_str = clean_graph(graph_data)
    graph_tokens = count_tokens(cleaned_graph_str)
    
    # 2. Process Original Text
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    original_text = get_original_text(data_path, sample_id)
    if original_text is None:
        print(f"Sample {sample_id} not found in data.")
        return
        
    original_tokens = count_tokens(original_text)
    
    # 3. Report
    print(f"--- Comparison for {sample_id} ---")
    print(f"Original Text Tokens: {original_tokens}")
    print(f"Graph JSON Tokens (no embeddings): {graph_tokens}")
    if original_tokens > 0:
        ratio = graph_tokens / original_tokens
        print(f"Compression Ratio: {ratio:.4f} ({ratio*100:.2f}%)")

if __name__ == "__main__":
    main()
