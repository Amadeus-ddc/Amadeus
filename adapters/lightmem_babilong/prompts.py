"""
Prompts for LightMemory on BABILong benchmark.
"""

import sys
import os
from pathlib import Path

# Add paths
BABILONG_PATH = os.environ.get("BABILONG_PATH", "/data/hzy/Amadeus/amadeus/experiments/babilong")
if BABILONG_PATH not in sys.path:
    sys.path.insert(0, BABILONG_PATH)

try:
    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
except ImportError:
    DEFAULT_PROMPTS = {}
    DEFAULT_TEMPLATE = "{instruction}\n\n{examples}\n\n{post_prompt}\n\n<context>\n{context}\n</context>\n\nQuestion: {question}"
    get_formatted_input = None

# Prompt for extracting facts from BABILong context
BABILONG_METADATA_PROMPT = """
You are a Fact Extractor for the BABILong benchmark.

Your task is to extract **all facts** from the provided context, where each fact is a simple statement about locations, possessions, or states.

**Important Instructions:**
1. Process each message in order (lowest source_id → highest).
2. Extract ALL factual information, including:
   - Location facts: "X is in Y", "X went to Y", "X moved to Y"
   - Possession facts: "X has Y", "X got Y", "X dropped Y"
   - State facts: "X is Y", "X are Z"
3. Preserve all specific details (names, locations, objects).
4. Each fact should be a clear, standalone statement.
5. Do NOT skip any information, no matter how trivial it seems.

**Output format:**
Always return a JSON object with key `"data"`, which is a list of items:
{
  "data": [
    {"source_id": 0, "fact": "Charlie went to the hallway."},
    {"source_id": 1, "fact": "Judith came back to the kitchen."},
    ...
  ]
}

**Example:**
Input messages:
- [0] Charlie went to the hallway.
- [1] Judith came back to the kitchen.
- [2] Charlie travelled to balcony.
- [3] Where is Charlie?

Output:
{
  "data": [
    {"source_id": 0, "fact": "Charlie went to the hallway."},
    {"source_id": 1, "fact": "Judith came back to the kitchen."},
    {"source_id": 2, "fact": "Charlie travelled to balcony."}
  ]
}

Remember: Extract ONLY facts, not questions. Be exhaustive and include all specific details.
"""


def build_answer_prompt(task: str, context: str, question: str) -> str:
    """
    Build answer prompt using amadeus's DEFAULT_PROMPTS.

    Args:
        task: Task name (e.g., 'qa1')
        context: Retrieved facts from LightMemory
        question: Original question

    Returns:
        Formatted prompt for answer generation
    """
    if task not in DEFAULT_PROMPTS:
        # Fallback to generic prompt
        return f"""Based on the following facts, answer the question:

Facts:
{context}

Question: {question}

Answer:"""

    # Use amadeus's prompt structure
    prompt_config = DEFAULT_PROMPTS[task]
    instruction = prompt_config.get('instruction', '')
    examples = prompt_config.get('examples', '')
    post_prompt = prompt_config.get('post_prompt', '')

    # Build prompt using amadeus's template
    if get_formatted_input:
        try:
            prompt = get_formatted_input(
                context=context,
                question=question,
                examples=examples,
                instruction=instruction,
                post_prompt=post_prompt,
                template=DEFAULT_TEMPLATE
            )
            return prompt
        except Exception:
            pass

    # Fallback: manual construction
    prompt = f"""{instruction}

{examples}

{post_prompt}

<context>
{context}
</context>

Question: {question}

Answer:"""
    return prompt
