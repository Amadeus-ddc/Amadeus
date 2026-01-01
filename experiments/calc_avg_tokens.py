import json
import tiktoken
import os
import numpy as np

def count_tokens(text):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def main():
    data_path = 'amadeus/experiments/data/locomo10.json'
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples.")
    
    token_counts = []
    
    for i, sample in enumerate(data):
        sample_id = sample.get('sample_id', f'sample_{i}')
        
        # Extract Conversation Text Only
        conv_text = ""
        if 'conversation' in sample:
            conv = sample['conversation']
            # Sort keys to ensure deterministic order (though not strictly necessary for sum)
            import re
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            
            session_keys = [k for k in conv.keys() if k.startswith('session_') and isinstance(conv[k], list)]
            session_keys.sort(key=natural_sort_key)
            
            for sk in session_keys:
                # Add session header context if available (usually date/time)
                date_key = f"{sk}_date_time"
                if date_key in conv and conv[date_key]:
                    conv_text += f"--- {conv[date_key]} ---\n"
                
                for turn in conv[sk]:
                    speaker = turn.get('speaker', 'Unknown')
                    text = turn.get('text', '')
                    conv_text += f"{speaker}: {text}\n"
        
        # 2. Add Source Documents (D1, D2...) - Logic from run_locomo.py
        exclude_keys = {'qa', 'sample_id', 'id', 'category', 'dataset', 'conversation'}
        other_keys = [k for k in sample.keys() if k not in exclude_keys and isinstance(sample[k], str)]
        other_keys.sort(key=natural_sort_key)
        
        for k in other_keys:
            conv_text += f"\n--- Source/Date Context: {k} ---\n{sample[k]}\n"

        count = count_tokens(conv_text)
        token_counts.append(count)
        print(f"{sample_id}: {count} tokens")

    avg_tokens = np.mean(token_counts)
    print(f"\n{'='*30}")
    print(f"Average Conversation Tokens: {avg_tokens:.2f}")
    print(f"{'='*30}")

if __name__ == "__main__":
    main()
