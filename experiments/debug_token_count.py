import json
import tiktoken

def count_tokens(text):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

with open('amadeus/experiments/data/locomo10.json', 'r') as f:
    data = json.load(f)

sample = data[0] # conv-26
print(f"Sample ID: {sample['sample_id']}")

total_tokens = 0

# 1. Conversation
conv_text = ""
if 'conversation' in sample:
    conv = sample['conversation']
    for k, v in conv.items():
        if k.startswith('session_') and isinstance(v, list):
            for turn in v:
                conv_text += f"{turn.get('speaker')}: {turn.get('text')}\n"
        elif isinstance(v, str):
             # session summaries inside conversation dict?
             pass

c_tokens = count_tokens(conv_text)
print(f"Conversation Tokens: {c_tokens}")
total_tokens += c_tokens

# 2. Other fields
for key in ['event_summary', 'observation', 'session_summary']:
    if key in sample:
        content = sample[key]
        if isinstance(content, str):
            t = count_tokens(content)
            print(f"Field '{key}' Tokens: {t}")
            total_tokens += t
        elif isinstance(content, list):
            text = "\n".join([str(x) for x in content])
            t = count_tokens(text)
            print(f"Field '{key}' (List) Tokens: {t}")
            total_tokens += t
        elif isinstance(content, dict):
            text = json.dumps(content)
            t = count_tokens(text)
            print(f"Field '{key}' (Dict) Tokens: {t}")
            total_tokens += t

print(f"Total Calculated Tokens: {total_tokens}")
