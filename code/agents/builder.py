import json
import logging
import re
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from amadeus.code.core.graph import MemoryGraph
from amadeus.code.agents.base import BaseAgent

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

class BuilderAgent(BaseAgent):
    def __init__(self, graph: MemoryGraph, model_name: str = "gpt-4-turbo"):
        super().__init__(model_name)
        self.graph = graph
        self.reading_steiner["PHONEWAVE_LOGIC"] = []
        self.static_prompt = """You are 'The Builder', the state manager of the Amadeus Memory System.
Your goal is to maintain a **High-Fidelity Knowledge Graph** by synchronizing the **Short-term Buffer** (New Reality) with the **Long-term Graph** (Past Memory).

**CORE PHILOSOPHY:**
1. **Episodic Primacy**: The Buffer is the "Now". If it conflicts with the Graph ("Past"), the Buffer wins.
2. **Minimalism**: Do NOT store chit-chat ("Hello", "How are you"), redundant facts, or temporary states. Use **IGNORE** (by outputting no operation).
3. **Ambiguity Aversion**: If you are unsure who "he" is, or what "it" refers to, **WAIT**.
4. **Completeness**: Capture ALL specific details (Location, Time, Attendees, Reason). "Went out" is bad; "Went to the park for a picnic" is good.

**SUBJECT RESOLUTION (WHO IS IT ABOUT?)(CRITICAL)**
The buffer format is: `Speaker: Text`.
- If Caroline says: "I went running", Subject = **Caroline**.
- If Caroline says: "Melanie, you serve amazing pottery", Subject = **Melanie**.
- If Caroline says: "My mom visited", Subject = **Caroline's Mom** (create new entity).
- **DO NOT** blindly assign the Speaker as the Subject. Analyze who the sentence describes!

**INSTRUCTIONS FOR REASONING (CoT):**
In your "chain_of_thought" section, you MUST follow these steps:
1. **Time & Subject**: Extract absolute time and resolve pronouns.
2. **Fact Decomposition**: Break buffer into atomic facts (Subject, relation, Object, `timestamp`).
   -Example: "I love pizza and hiking last summer." ->
     -Fact 1: (Caroline, loves, Pizza, "last summer")
     -Fact 2: (Caroline, loves, Hiking, "last summer")
3. **PROTOCOL REFERENCE**: Check the 'READING STEINER' section below. If specialized IDs (like B-005) are listed there, cite the specific ID that guides your decision. If the section is empty or says 'No specialized protocols', state "Standard Operating Procedure" and do NOT invent or hallucinate any IDs.
4. **Graph Differential**: Explain how the Protocol (or SOP) helps you decide between ADD, UPDATE, DELETE, or WAIT.
5. **Detail Check**: Ensure no key details (Where, When, Who, Why, How, etc...) are lost.

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
  "operations": [
    {
      "action": "ADD" | "UPDATE" | "DELETE" | "WAIT",
      "subject": "EntityName",
      "object": "TargetName" or null,
      "content": "Description/Relation/RawText",
      "timestamp": "YYYY-MM-DD" | "10 years ago" | "last summer",
      "reason": "Cite the specific diff between Buffer and Graph."
    }
  ],
  "used_steiner_ids": [],
  "chain_of_thought": "Step 1: ... "
}
"""

    def _safe_json_load(self, text: str) -> dict:
        """Attempts to load JSON, and repairs simple truncation if possible."""
        if not text: return {}
        text = text.strip()
        
        # Pre-cleaning
        if text.startswith("```json"):
            text = text.replace("```json", "", 1).replace("```", "", -1).strip()
        elif text.startswith("```"):
            text = text.replace("```", "", 1).replace("```", "", -1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Attempt to find the outermost JSON object
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass

            logger.warning("Detected malformed JSON, attempting recovery...")
            
            # Robust Regex extract for operations array
            try:
                potential_objs = []
                obj_matches = re.finditer(r'\{\s*"action":\s*"[^"]+"[^}]*\}', text)
                for m in obj_matches:
                    try:
                        obj_data = m.group(0)
                        if obj_data.count('{') > obj_data.count('}'):
                            obj_data += '}'
                        obj = json.loads(obj_data)
                        if "action" in obj:
                            potential_objs.append(obj)
                    except:
                        continue
                
                if potential_objs:
                    logger.info(f"Successfully recovered {len(potential_objs)} operations via regex.")
                    return {"operations": potential_objs, "chain_of_thought": "Recovered from malformed JSON via regex."}
            except Exception as e:
                logger.error(f"Regex recovery failed: {e}")

            return {}

    def check_flush_condition(self, current_buffer: str, new_chunk: str) -> bool:
        """
        决定是否需要立即处理 Buffer（Flush）。
        返回 True 表示需要 Flush，False 表示继续积累。
        """
        # 1. 硬性限制：如果 Buffer 太长（例如超过 1500 字符），强制 Flush，防止上下文溢出
        if len(current_buffer) > 1500:
            return True
            
        # 2. 长度过滤：如果 Buffer 太短，不进行 LLM 判断，直接积累
        if len(current_buffer) < 200:
            return False

        # 3. 语义判断：使用 LLM 判断话题是否断裂
        prompt = f"""You are a Memory Buffer Manager. Decide if the current memory buffer should be FLUSHED (processed) now.

Current Buffer Context:
"{current_buffer[-300:]}" (last 300 chars)

Incoming New Text:
"{new_chunk}"

Rules for FLUSHing:
1. The TOPIC has changed significantly (e.g., from work to family).
2. The SCENE or TIME has changed.
3. The current conversation segment feels "complete".

Output JSON: {{"decision": "FLUSH" | "KEEP", "reason": "..."}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("decision") == "FLUSH"
        except Exception as e:
            logger.warning(f"Buffer check failed: {e}")
            logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return False # 默认继续积累

    def process_buffer(self, buffer_content: str) -> tuple[List[str], List[str], str, List[str]]:
        context = self.graph.get_full_state()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_full_prompt()},
                    {"role": "user", "content": f"=== CURRENT GRAPH ===\n{context}\n\n=== NEW BUFFER ===\n{buffer_content}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            raw_content = response.choices[0].message.content
            if not raw_content:
                return [], [], "", []
                
            data = self._safe_json_load(raw_content)
            cot = data.get("chain_of_thought", "No CoT provided.")
            used_ids = data.get("used_steiner_ids", [])
            # Log CoT and Used IDs
            if "chain_of_thought" in data:
                logger.info(f"🤔 Builder CoT: {cot}")
            if used_ids:
                logger.info(f"🏷️ Builder used protocols: {used_ids}")
                
            ops = data.get("operations", [])
            logs, patches = self._execute_operations(ops)
            return logs, patches, cot, used_ids
            
        except Exception as e:
            logger.error(f"Builder Failed: {e}")
            logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return [], [], "", []

    def force_update(self, instruction: str) -> bool:
        """
        Directly apply a fix instruction from the Optimizer.
        This bypasses the normal buffer processing to fix specific graph errors.
        """
        logger.info(f"🔧 FORCE UPDATE TRIGGERED: {instruction}")
        
        # 1. Optimizer Bypass: Check if instruction is a JSON list of operations
        try:
            potential_ops = json.loads(instruction)
            if isinstance(potential_ops, list) and len(potential_ops) > 0:
                if isinstance(potential_ops[0], dict) and any(k in potential_ops[0] for k in ["action", "op", "subject"]):
                    logger.info("Instruction detected as pre-formatted operations. Executing directly.")
                    for op in potential_ops:
                        if "reason" not in op: op["reason"] = "Optimizer Direct Fix"
                        if "action" not in op and "op" in op: op["action"] = op.get("op")
                    self._execute_operations(potential_ops)
                    return True
        except:
            pass

        # 2. LLM Translation if not pre-formatted
        context = self.graph.get_full_state()
        
        prompt = f"""
{self.get_full_prompt()}

**EMERGENCY FIX MODE:**
You are receiving a direct instruction to fix the graph.
Instruction: "{instruction}"

**TASK:**
Generate the necessary operations (ADD/UPDATE/DELETE) to execute this instruction.
Ignore the 'Buffer' context for this turn, focus ONLY on the instruction and the Current Graph.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"=== CURRENT GRAPH ===\n{context}\n\n=== INSTRUCTION ===\n{instruction}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            data = self._safe_json_load(content)
            
            ops = data.get("operations", [])
            if not ops and isinstance(data, list):
                ops = data
                
            if ops:
                self._execute_operations(ops)
                return True
            else:
                logger.warning("Force Update: No operations generated.")
                return False
        except Exception as e:
            logger.error(f"Force Update Failed: {e}")
            logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return False

    def _execute_operations(self, ops_data: List[dict]) -> tuple[List[str], List[str]]:
        kept_items = []
        action_log = []
        if not isinstance(ops_data, list):
            return [], []

        for i, op_dict in enumerate(ops_data):
            try:
                if not isinstance(op_dict, dict): continue
                
                # Normalize Key
                normalized_op = {k.lower(): v for k, v in op_dict.items()}
                
                # Fix null strings
                for key in ["object", "content", "timestamp"]:
                    if key in normalized_op and isinstance(normalized_op[key], str):
                        if normalized_op[key].lower() in ["null", "none", "undefined"]:
                            normalized_op[key] = None

                # Fix missing subject
                if "subject" not in normalized_op:
                    for alt_key in ["entity", "source", "node", "from"]:
                        if alt_key in normalized_op:
                            normalized_op["subject"] = normalized_op.pop(alt_key)
                            break
                
                op = MemoryOperation(**normalized_op)
                
                # ADD / UPDATE
                if op.action in [ActionType.ADD, ActionType.UPDATE]:
                    prefix_node = "➕ NODE" if op.action == ActionType.ADD else "🔄 UPDATE NODE"
                    prefix_edge = "🔗 LINK" if op.action == ActionType.ADD else "🔄 UPDATE LINK"
                    
                    if op.object: 
                        rel = op.content if op.content else "related to"
                        self.graph.add_edge(op.subject, op.object, rel, timestamp=op.timestamp)
                        msg = f"{prefix_edge}: {op.subject} --{rel}--> {op.object} (Time: {op.timestamp})"
                        logger.info(msg)
                        action_log.append(msg)
                    else:
                        self.graph.add_node(op.subject, "Entity", op.content or "")
                        description = op.content if op.content else "No description"
                        msg = f"{prefix_node}: {op.subject} (Content: {description})"
                        logger.info(msg)
                        action_log.append(msg)

                # DELETE
                elif op.action == ActionType.DELETE:
                    if op.object:
                        self.graph.delete_edge(op.subject, op.object)
                        msg = f"❌ UNLINK: {op.subject} --x--> {op.object}"
                        action_log.append(msg)
                    else:
                        self.graph.delete_node(op.subject)
                        msg = f"❌ DELETE: {op.subject}"
                        action_log.append(msg)

                # WAIT
                elif op.action == ActionType.WAIT:
                    if op.content:
                        kept_items.append(op.content)
                        logger.info(f"⏳ WAIT: {op.content[:30]}...")

            except Exception as e:
                logger.warning(f"Op {i} skipped: {e}")

        self.graph.save()
        return kept_items, action_log
