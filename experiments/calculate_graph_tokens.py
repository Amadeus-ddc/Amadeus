import json
import os
import glob
import numpy as np
from transformers import AutoTokenizer

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def remove_embeddings(data):
    """Recursively remove 'embedding' keys from dictionary or list."""
    if isinstance(data, dict):
        # Create a list of keys to avoid runtime error during iteration
        keys = list(data.keys())
        for key in keys:
            if key == "embedding":
                del data[key]
            else:
                remove_embeddings(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_embeddings(item)
    return data

def main():
    # Use the local model path found in the workspace
    model_path = "/home/ubuntu/hzy/crl/Amadeus/amadeus/models/all-MiniLM-L6-v2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {model_path}: {e}")
        # Fallback to a simple whitespace split approximation if tokenizer fails
        print("Falling back to simple whitespace splitting (approximate).")
        tokenizer = None

    target_dir = '/home/ubuntu/hzy/crl/Amadeus/amadeus/experiments/results_locomo/run_20260105_182633'
    pattern = os.path.join(target_dir, 'graph_conv-*.json')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No graph files found in {target_dir}")
        return

    print(f"Found {len(files)} graph files.")
    
    token_counts = []
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Remove embeddings to get a cleaner token count
                data = remove_embeddings(data)
                
                text_content = json.dumps(data, ensure_ascii=False)
                
                if tokenizer:
                    count = count_tokens(text_content, tokenizer)
                else:
                    count = len(text_content.split()) # Very rough approximation
                
                token_counts.append(count)
                print(f"{file_name}: {count} tokens")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    if token_counts:
        avg_tokens = np.mean(token_counts)
        print(f"\nAverage Token Count: {avg_tokens:.2f}")
        print(f"Min Token Count: {np.min(token_counts)}")
        print(f"Max Token Count: {np.max(token_counts)}")
    else:
        print("No valid token counts obtained.")

if __name__ == "__main__":
    main()
