import json
import logging
import re
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from openai import OpenAI
from amadeus.core.graph import MemoryGraph

logger = logging.getLogger("Amadeus.Builder")

class ActionType(str, Enum):
    ADD = "ADD"       # 新增/合并信息
    UPDATE = "UPDATE" # 修正信息
    DELETE = "DELETE" # 删除错误信息
    WAIT = "WAIT"     # 暂存模糊信息

class MemoryOperation(BaseModel):
    action: ActionType = Field(..., description="Cognitive primitive.")
    subject: str = Field(..., description="Primary entity name.")
    object: Optional[str] = Field(None, description="Target entity. If Present -> Edge Op. If Null -> Node Op.")
    content: Optional[str] = Field(None, description="Node description / Edge relation / Raw text for WAIT.")
    timestamp: Optional[str] = Field(None, description="Absolute date (YYYY-MM-DD) PREFERRED. If calculation fails, use relative time (e.g. '10 years ago').")
    reason: str = Field(..., description="Reason for this operation (Conflict/New Fact/Ambiguity).")

class BuilderOutput(BaseModel):
    chain_of_thought: str = Field(..., description="Step-by-step reasoning about Buffer vs Graph.")
    operations: List[MemoryOperation] = Field(..., description="Sequence of atomic operations.")

class BuilderAgent:
    def __init__(self, graph: MemoryGraph, model_name: str = "gpt-4-turbo"):
        self.graph = graph
        self.client = OpenAI()
        self.model_name = model_name

    def _get_system_prompt(self) -> str:
        return """You are 'The Builder', the state manager of the Amadeus Memory System.
Your goal is to maintain a **High-Fidelity Knowledge Graph** by synchronizing the **Short-term Buffer** (New Reality) with the **Long-term Graph** (Past Memory).

**CORE PHILOSOPHY:**
1. **Episodic Primacy**: The Buffer is the "Now". If it conflicts with the Graph ("Past"), the Buffer wins.
2. **Minimalism**: Do NOT store chit-chat ("Hello", "How are you"), redundant facts, or temporary states. Use **IGNORE** (by outputting no operation).
3. **Ambiguity Aversion**: If you are unsure who "he" is, or what "it" refers to, **WAIT**.

**SUBJECT RESOLUTION (WHO IS IT ABOUT?)(CRITICAL)**
The buffer format is: `Speaker: Text`.
- If Caroline says: "I went running", Subject = **Caroline**.
- If Caroline says: "Melanie, you serve amazing pottery", Subject = **Melanie**.
- If Caroline says: "My mom visited", Subject = **Caroline's Mom** (create new entity).
- **DO NOT** blindly assign the Speaker as the Subject. Analyze who the sentence describes!

**TEMPORAL NORMALIZATION RULE (CRITICAL):**
The Buffer will start with a context line like "--- Session Context: [Date/Summary] ---".
You MUST use this context to resolve relative time expressions into ABSOLUTE DATES.
- Input: "Context: 2023-07-15... Input: I went hiking last Friday."
- Action: Calculate the date (e.g., 2023-07-07) and store: ADD(Caroline, Hiking, "Went hiking"). Set "timestamp": "2023-07-07".
- **Fallback**: If you CANNOT calculate the absolute date (e.g., context is missing year), you MAY store the relative expression (e.g., "10 years ago", "in childhood") in the "timestamp" field.

**COGNITIVE PRIMITIVES:**

1. **ADD(subject, object?, content, timestamp?)**
   - **Trigger**: A NEW, verifiable fact that does not exist in the Graph.
   - **Node**: `object`=null. Create specific entities (e.g., "Caroline" not "User").
   - **Edge**: `object`=Target. Create relationships.
   - *Example*: "I just adopted a cat." -> ADD("Caroline", "Cat", "ADOPTED", "2023-01-01")

2. **UPDATE(subject, object?, content, timestamp?)**
   - **Trigger**: The entity exists, but the state has changed or become more detailed.
   - **Type A (Refinement)**: Old: "Likes pizza". New: "Loves pepperoni". -> UPDATE description.
   - **Type B (Overwriting)**: Old: "Lives in London". New: "Moved to Paris". -> UPDATE edge/attribute to the NEW truth.

3. **DELETE(subject, object?)**
   - **Trigger**: Explicit contradiction or obsolescence.
   - **Rule**: If a relationship is physically impossible to co-exist (e.g., "Single" vs "Married"), DELETE the old one first.

4. **WAIT(subject, content)**
   - **Trigger**: Unresolved pronouns ("He said..."), vague future plans, or incomplete stories.
   - **Action**: Keep the RAW text in `content`. It will roll over to the next turn.

**OUTPUT SCHEMA (JSON):**
{
  "chain_of_thought": "Step 1: Identify absolute date from Context. Step 2: Identify entities. Step 3: Compare with Graph. Step 4: Resolve conflicts.",
  "operations": [
    {
      "action": "ADD" | "UPDATE" | "DELETE" | "WAIT",
      "subject": "EntityName",
      "object": "TargetName" or null,
      "content": "Description/Relation/RawText",
      "timestamp": "YYYY-MM-DD" | "10 years ago" | "last summer",
      "reason": "Cite the specific diff between Buffer and Graph."
    }
  ]
}
"""

    def process_buffer(self, buffer_content: str) -> List[str]:
        context = self.graph.get_full_state()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"=== CURRENT GRAPH ===\n{context}\n\n=== NEW BUFFER ===\n{buffer_content}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            raw_content = response.choices[0].message.content
            if not raw_content:
                logger.error("Empty response from Builder LLM")
                return []
                
            clean_json = re.sub(r'^```json\s*|\s*```$', '', raw_content.strip(), flags=re.MULTILINE)
            data = json.loads(clean_json)
            
            cot = data.get("chain_of_thought", "No reasoning provided.")
            logger.info(f"🤔 Builder CoT: {cot}")

            ops_data = data.get("operations", [])
            if not isinstance(ops_data, list):
                return []

            kept_items = []
            for i, op_dict in enumerate(ops_data):
                try:
                    if not isinstance(op_dict, dict): continue
                    
                    # 标准化 Key
                    normalized_op = {k.lower(): v for k, v in op_dict.items()}
                    
                    # 修复 null 字符串
                    for key in ["object", "content", "timestamp"]:
                        if key in normalized_op and isinstance(normalized_op[key], str):
                            if normalized_op[key].lower() in ["null", "none", "undefined"]:
                                normalized_op[key] = None

                    # 修复缺失 subject
                    if "subject" not in normalized_op:
                        for alt_key in ["entity", "source", "node", "from"]:
                            if alt_key in normalized_op:
                                normalized_op["subject"] = normalized_op.pop(alt_key)
                                break
                    
                    op = MemoryOperation(**normalized_op)
                    
                    # ADD / UPDATE
                    if op.action in [ActionType.ADD, ActionType.UPDATE]:
                        if op.object: 
                            rel = op.content if op.content else "related to"
                            self.graph.add_edge(op.subject, op.object, rel, timestamp=op.timestamp)
                            logger.info(f"🔗 LINK: {op.subject} --{rel}--> {op.object} (Time: {op.timestamp})")
                        else:
                            self.graph.add_node(op.subject, "Entity", op.content or "")
                            logger.info(f"➕ NODE: {op.subject}")

                    # DELETE
                    elif op.action == ActionType.DELETE:
                        if op.object:
                            self.graph.delete_edge(op.subject, op.object)
                        else:
                            self.graph.delete_node(op.subject)

                    # WAIT
                    elif op.action == ActionType.WAIT:
                        if op.content:
                            kept_items.append(op.content)
                            logger.info(f"⏳ WAIT: {op.content[:30]}...")

                except Exception as e:
                    logger.warning(f"Op {i} skipped: {e}")

            self.graph.save()
            return kept_items

        except Exception as e:
            logger.error(f"Builder Failed: {e}")
            return []