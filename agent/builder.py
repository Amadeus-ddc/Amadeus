import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from amadeus.core.graph import MemoryGraph

logger = logging.getLogger("Amadeus.Builder")

# --- 1. 核心 Schema 定义 ---

class Entity(BaseModel):
    name: str = Field(..., description="Unique name of the entity")
    type: str = Field(..., description="Category (Person, Location, Object, etc.)")
    description: str = Field(..., description="Factual attributes")

class Relation(BaseModel):
    source: str
    target: str
    relation: str

class WaitItem(BaseModel):
    """
    WAIT 算子的实体化。
    代表那些“很重要但目前还不清楚”的信息片段。
    """
    original_text: str = Field(..., description="The exact raw text snippet to keep in buffer")
    reason: str = Field(..., description="Why defer? e.g., 'Unresolved pronoun', 'Future plan unconfirmed'")

class ExtractionResult(BaseModel):
    """
    Builder 的完整决策输出：
    - commits: 确定的事实 (Entities + Relations) -> 存入 Graph
    - defers: 模糊的信息 (WaitItems) -> 保留 Buffer
    """
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    deferred_items: List[WaitItem] = Field(default_factory=list, description="Items to WAIT/KEEP in buffer")
    reasoning: str = Field(..., description="High-level strategy summary")

# --- 2. Builder Agent ---

class BuilderAgent:
    def __init__(self, graph: MemoryGraph, model_name: str = "gpt-4-turbo"):
        self.graph = graph
        self.client = OpenAI() 
        self.model_name = model_name

    def _get_system_prompt(self) -> str:
        return """You are 'The Builder' in the Amadeus Memory System.
Your goal is to COMPRESS the Short-term Buffer into the Long-term Memory Graph.

**Decision Logic (The Min-Max Game):**
1. **COMMIT (Add/Update):** - Use when information is explicit, factual, and high-confidence.
   - Example: "Alice is in the Kitchen" -> Entity(Alice), Entity(Kitchen), Relation(AT).
   
2. **WAIT (Defer):** - CRITICAL: Do NOT store ambiguous or incomplete info.
   - Use when pronouns are unclear ("He went there"), or the state is transient/uncertain.
   - Action: Add these snippets to `deferred_items`. They will STAY in the buffer for the next turn.
   
3. **IGNORE (Delete):**
   - Chit-chat, greetings, or redundant info. Just don't include them in the output.

**Constraint:**
- Don't hallucinate. If you are not sure, use WAIT.
- Output strictly in JSON format matching the schema.
"""

    def process_buffer(self, buffer_content: str) -> List[str]:
        """
        执行建图，并返回需要'保留'在 Buffer 中的文本列表。
        """
        logger.info("🧠 Builder analyzing buffer...")
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"Current Graph State:\n{self.graph.get_full_state()}\n\nNew Buffer Content:\n---\n{buffer_content}\n---"}
                ],
                response_format=ExtractionResult,
                temperature=0.0,
            )
            
            result: ExtractionResult = completion.choices[0].message.parsed
            
            # --- 1. 执行 COMMIT (Add/Update) ---
            commit_count = 0
            for entity in result.entities:
                self.graph.add_node(entity.name, entity.type, entity.description)
                commit_count += 1
            
            for rel in result.relations:
                self.graph.add_edge(rel.source, rel.target, rel.relation.upper())
                commit_count += 1
            
            logger.info(f"🔨 Committed {commit_count} facts. Reason: {result.reasoning}")

            # --- 2. 处理 WAIT ---
            kept_texts = []
            if result.deferred_items:
                logger.info(f"⏳ WAITING on {len(result.deferred_items)} items:")
                for item in result.deferred_items:
                    logger.info(f"   - '{item.original_text}' (Reason: {item.reason})")
                    kept_texts.append(item.original_text)
            
            # --- 3. 持久化 ---
            self.graph.save()
            
            # 返回需要保留的文本列表 (给外部 Buffer 模块用)
            return kept_texts

        except Exception as e:
            logger.error(f"Builder failed: {e}")
            return [] # 出错时保守策略：也许应该返回整个 buffer，但这里先返回空
