import json
import logging
import re
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from amadeus_collab.core.graph import MemoryGraph
from amadeus_collab.core.schema import (
    EmergenceOutput,
    EdgeTypeProposal,
    NodeTypeProposal,
    RuleProposal,
    SchemaProposalBundle,
    SchemaState,
)
from amadeus_collab.agents.base import BaseAgent

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
    node_type: Optional[str] = Field(None, description="Type for node operations.")
    edge_type: Optional[str] = Field(None, description="Type for edge operations.")

class BuilderOutput(BaseModel):
    chain_of_thought: str = Field(..., description="Step-by-step reasoning about Buffer vs Graph.")
    operations: List[MemoryOperation] = Field(..., description="Sequence of atomic operations.")
    schema_proposals: Optional[dict] = Field(default=None, description="Optional schema proposals for node/edge types and rules.")

class BuilderAgent(BaseAgent):
    def __init__(self, graph: MemoryGraph, model_name: str = "gpt-4-turbo"):
        super().__init__(model_name)
        self.graph = graph
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

**TEMPORAL NORMALIZATION RULE (CRITICAL):**
The Buffer will start with a context line like "--- Session Context: [Date/Summary] ---".
You MUST use this context to resolve relative time expressions into ABSOLUTE DATES.
- Input: "Context: 2023-07-15... Input: I went hiking last Friday."
- Action: Calculate the date (e.g., 2023-07-07) and store: ADD(Caroline, Hiking, "Went hiking"). Set "timestamp": "2023-07-07".
- **Fallback**: If you CANNOT calculate the absolute date (e.g., context is missing year), you MAY store the relative expression (e.g., "10 years ago", "in childhood") in the "timestamp" field.

**SCHEMA-AWARE BUILDING:**
- You will be shown emerged node types, edge types, and builder rules.
- Keep the original reasoning process, but additionally prefer reusing existing node types and edge types.
- Only propose a new type when existing types are clearly insufficient.
- New types must be reusable abstract categories, never sample-specific names, possessive labels, or one-off event titles.
- ExperienceNode is only an example of a reusable high-level experience. Do not create it unless the buffer supports a truly reusable experience.

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

**THINKING PROCESS (CHAIN OF THOUGHT):**
1. **Time & Subject**: Extract absolute time and resolve pronouns.
2. **Fact Decomposition**: Break buffer into atomic facts (Subject-Predicate-Object).
3. **Graph Differential**: For each atomic fact, check if it exists in Graph.
   - Missing? -> ADD.
   - Changing? -> UPDATE.
   - Contradiction? -> DELETE.
   - Ambiguity? -> WAIT.
4. **Detail Check**: Ensure no key details (Where, When, Who, Why) are lost.

**OUTPUT SCHEMA (JSON):**
{
  "chain_of_thought": "Step 1: Date is 2023-08-01. Entities are Caroline and Mom. Step 2: New fact 'Mom visited'. Step 3: Graph has no 'Mom', so ADD Node Mom, ADD Edge visited. Step 4: Include 'dinner' detail in content.",
  "operations": [
    {
      "action": "ADD" | "UPDATE" | "DELETE" | "WAIT",
      "subject": "EntityName",
      "object": "TargetName" or null,
      "content": "Description/Relation/RawText",
      "timestamp": "YYYY-MM-DD" | "10 years ago" | "last summer",
      "reason": "Cite the specific diff between Buffer and Graph.",
      "node_type": "OptionalNodeTypeName",
      "edge_type": "OptionalEdgeTypeName"
    }
  ],
  "schema_proposals": {
    "node_types": [
      {
        "name": "AbstractNodeType",
        "description": "...",
        "when_to_create": "...",
        "usage_scene": "...",
        "examples": ["..."]
      }
    ],
    "edge_types": [
      {
        "name": "AbstractEdgeType",
        "description": "...",
        "when_to_create": "...",
        "usage_scene": "...",
        "examples": ["..."]
      }
    ],
    "rules": [
      {
        "name": "RuleName",
        "rule_text": "...",
        "examples": ["..."]
      }
    ]
  }
}
"""

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
            self._record_usage(response)
            result = json.loads(response.choices[0].message.content)
            return result.get("decision") == "FLUSH"
        except Exception as e:
            logger.warning(f"Buffer check failed: {e}")
            logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return False # 默认继续积累

    def _build_schema_context(self, schema_state: Optional[SchemaState]) -> str:
        if not schema_state:
            return "No schema artifact provided. Use broad defaults and only propose new abstract types when necessary."
        return schema_state.to_prompt_context()

    def _parse_schema_proposals(self, payload: Optional[dict]) -> SchemaProposalBundle:
        if not isinstance(payload, dict):
            return SchemaProposalBundle()
        try:
            return SchemaProposalBundle(
                node_types=[NodeTypeProposal(**item) for item in payload.get("node_types", []) if isinstance(item, dict)],
                edge_types=[EdgeTypeProposal(**item) for item in payload.get("edge_types", []) if isinstance(item, dict)],
                rules=[RuleProposal(**item) for item in payload.get("rules", []) if isinstance(item, dict)],
            )
        except Exception as e:
            logger.warning(f"Schema proposal parse failed: {e}")
            return SchemaProposalBundle()

    def process_buffer(
        self,
        buffer_content: str,
        schema_state: Optional[SchemaState] = None,
        buffer_index: Optional[int] = None,
        replay_mode: bool = False,
    ) -> tuple[List[str], List[str], SchemaProposalBundle]:
        context = self.graph.get_full_state()
        schema_context = self._build_schema_context(schema_state)
        buffer_meta = f"Buffer Index: {buffer_index}" if buffer_index is not None else "Buffer Index: Unknown"
        replay_meta = "Replay Mode: ON" if replay_mode else "Replay Mode: OFF"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_full_prompt()},
                    {
                        "role": "user",
                        "content": (
                            f"=== CURRENT GRAPH ===\n{context}\n\n"
                            f"=== CURRENT SCHEMA ===\n{schema_context}\n\n"
                            f"=== BUILD META ===\n{buffer_meta}\n{replay_meta}\n\n"
                            f"=== NEW BUFFER ===\n{buffer_content}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            self._record_usage(response)

            raw_content = response.choices[0].message.content
            if not raw_content:
                return [], [], SchemaProposalBundle()

            data = json.loads(raw_content)

            if "chain_of_thought" in data:
                logger.info(f"🤔 Builder CoT: {data['chain_of_thought']}")

            ops = data.get("operations", [])
            kept_items, action_log = self._execute_operations(ops)
            schema_proposals = self._parse_schema_proposals(data.get("schema_proposals"))
            if schema_proposals.node_types or schema_proposals.edge_types or schema_proposals.rules:
                logger.info(
                    "🧩 Builder schema proposals | nodes=%s | edges=%s | rules=%s",
                    [p.name for p in schema_proposals.node_types],
                    [p.name for p in schema_proposals.edge_types],
                    [p.name for p in schema_proposals.rules],
                )
            logger.info(
                "🛠️ Builder execution summary | buffer_index=%s | replay=%s | ops=%d | waits=%d",
                buffer_index,
                replay_mode,
                len(action_log),
                len(kept_items),
            )
            return kept_items, action_log, schema_proposals

        except Exception as e:
            logger.error(f"Builder Failed: {e}")
            logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return [], [], SchemaProposalBundle()

    def force_update(self, instruction: str, schema_state: Optional[SchemaState] = None) -> bool:
        """
        Directly apply a fix instruction from the Optimizer.
        This bypasses the normal buffer processing to fix specific graph errors.
        """
        logger.info(f"🔧 FORCE UPDATE TRIGGERED: {instruction}")
        context = self.graph.get_full_state()
        schema_context = self._build_schema_context(schema_state)

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
                    {
                        "role": "user",
                        "content": f"=== CURRENT GRAPH ===\n{context}\n\n=== CURRENT SCHEMA ===\n{schema_context}\n\n=== INSTRUCTION ===\n{instruction}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            self._record_usage(response)
            content = response.choices[0].message.content
            data = json.loads(content)
            ops = data.get("operations", [])
            self._execute_operations(ops)
            return True
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

                # Map legacy/invalid actions to supported ones.
                action_val = normalized_op.get("action")
                if isinstance(action_val, str) and action_val.upper() == "LINK":
                    logger.warning("Received action LINK; mapping to ADD.")
                    normalized_op["action"] = "ADD"
                
                op = MemoryOperation(**normalized_op)
                
                # ADD / UPDATE
                if op.action in [ActionType.ADD, ActionType.UPDATE]:
                    prefix_node = "➕ NODE" if op.action == ActionType.ADD else "🔄 UPDATE NODE"
                    prefix_edge = "🔗 LINK" if op.action == ActionType.ADD else "🔄 UPDATE LINK"
                    
                    if op.object:
                        rel = op.content if op.content else "related to"
                        edge_type = op.edge_type or "RelationEdge"
                        self.graph.add_edge(op.subject, op.object, rel, timestamp=op.timestamp, edge_type=edge_type)
                        msg = f"{prefix_edge}: {op.subject} --{rel}<{edge_type}>--> {op.object} (Time: {op.timestamp})"
                        logger.info(msg)
                        action_log.append(msg)
                    else:
                        node_type = op.node_type or "Entity"
                        self.graph.add_node(op.subject, node_type, op.content or "")
                        description = op.content if op.content else "No description"
                        msg = f"{prefix_node}: {op.subject} <{node_type}> (Content: {description})"
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

    def emerge_schema(self, buffers: List[str], current_schema: Optional[SchemaState] = None) -> EmergenceOutput:
        schema_context = self._build_schema_context(current_schema)
        prompt = f"""
You are deriving a reusable schema for graph building from the first few buffers of a sample.

Requirements:
- Propose reusable node types, edge types, and builder rules.
- Keep node types and edge types separate.
- Prefer a small, high-coverage schema.
- Do not invent sample-specific names, possessive labels, or one-off event titles as types.
- ExperienceNode is only an example of a reusable high-level experience. Do not force it when the evidence only supports a single event.
- Decide whether the schema is stable enough to continue building, or whether more buffers are needed.
- Output JSON only.

Current schema:
{schema_context}

Buffers for emergence:
{chr(10).join(f'--- Buffer {i+1} ---{chr(10)}{buf}' for i, buf in enumerate(buffers))}

Return:
{{
  "analysis": "...",
  "stable": true | false,
  "reason": "...",
  "proposals": {{
    "node_types": [...],
    "edge_types": [...],
    "rules": [...]
  }}
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            self._record_usage(response)
            data = json.loads(response.choices[0].message.content)
            proposals = self._parse_schema_proposals(data.get("proposals"))
            output = EmergenceOutput(
                analysis=data.get("analysis", ""),
                stable=bool(data.get("stable", False)),
                reason=data.get("reason", ""),
                proposals=proposals,
            )
            logger.info(
                "🧠 Schema emergence | stable=%s | reason=%s | node_types=%s | edge_types=%s | rules=%s",
                output.stable,
                output.reason,
                [p.name for p in output.proposals.node_types],
                [p.name for p in output.proposals.edge_types],
                [p.name for p in output.proposals.rules],
            )
            if output.analysis:
                logger.info(f"🧠 Schema emergence analysis: {output.analysis}")
            return output
        except Exception as e:
            logger.error(f"Schema emergence failed: {e}")
            return EmergenceOutput()

    def review_schema_proposals(
        self,
        proposals: SchemaProposalBundle,
        current_schema: Optional[SchemaState] = None,
    ) -> List[dict]:
        schema_context = self._build_schema_context(current_schema)
        proposal_payload = proposals.model_dump(mode="json")
        prompt = f"""
Review schema proposals for graph building.

Rules:
- Compare against existing active node types and edge types.
- Reject or generalize sample-specific labels, possessive labels, and one-off event titles.
- Merge aliases into existing abstract categories when possible.
- Keep only reusable abstract categories.
- Rules may be kept unless they are duplicates or sample-specific.
- Output JSON only.

Current schema:
{schema_context}

Proposals:
{json.dumps(proposal_payload, ensure_ascii=False, indent=2)}

Return:
{{
  "results": [
    {{
      "kind": "node" | "edge" | "rule",
      "proposed_name": "...",
      "action": "keep" | "merge" | "generalize" | "reject",
      "canonical_name": "...",
      "reason": "..."
    }}
  ]
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            self._record_usage(response)
            data = json.loads(response.choices[0].message.content)
            results = data.get("results", [])
            if isinstance(results, list):
                cleaned = [item for item in results if isinstance(item, dict)]
                if cleaned:
                    logger.info(
                        "🔍 Schema review results | %s",
                        "; ".join(
                            f"{item.get('kind')}:{item.get('proposed_name')}->{item.get('action')}({item.get('canonical_name')})"
                            for item in cleaned
                        ),
                    )
                return cleaned
        except Exception as e:
            logger.error(f"Schema review failed: {e}")

        fallback = []
        for item in proposals.node_types:
            fallback.append(
                {
                    "kind": "node",
                    "proposed_name": item.name,
                    "action": "reject" if re.search(r"'s|\bof\b", item.name, re.IGNORECASE) else "keep",
                    "canonical_name": None,
                    "reason": "fallback review",
                }
            )
        for item in proposals.edge_types:
            fallback.append(
                {
                    "kind": "edge",
                    "proposed_name": item.name,
                    "action": "reject" if re.search(r"'s|\bof\b", item.name, re.IGNORECASE) else "keep",
                    "canonical_name": None,
                    "reason": "fallback review",
                }
            )
        for item in proposals.rules:
            fallback.append(
                {
                    "kind": "rule",
                    "proposed_name": item.name,
                    "action": "keep",
                    "canonical_name": item.name,
                    "reason": "fallback review",
                }
            )
        return fallback
