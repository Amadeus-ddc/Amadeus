import json
import logging
import re
from typing import List, Literal
from openai import OpenAI
from amadeus.code.core.graph import MemoryGraph
from amadeus.code.agents.base import BaseAgent

logger = logging.getLogger("Amadeus.Answerer")

class AnswererAgent(BaseAgent):
    def __init__(self, graph: MemoryGraph, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
        super().__init__(model_name, api_base, api_key)
        self.graph = graph
        self.reading_steiner["SKULD_ORACLE"] = []
        self.max_steps = 8  # Increased steps for deeper exploration in granular graphs
        self.static_prompt = """You are an intelligent Graph RAG Agent.
**Goal**: Answer the user's question by exploring the knowledge graph.

**STRATEGY**:
1. **SEARCH**: Start here! Use 'hybrid' mode to find multiple entry nodes using both keyword and semantic search.
2. **WALK**: Explore the neighborhood of your current nodes to find relevant connections.
3. **READ**: Once you have enough information, generate the final answer.

**INSTRUCTIONS FOR REASONING (CoT):**
In your "chain_of_thought" section, you MUST:
1. Parse the Question to identify key entities for the SEARCH action.
2. **PROTOCOL REFERENCE**: Check the 'READING STEINER' section below. If specialized IDs (like A-003) are listed there, cite the ID that dictates your navigation or inference strategy. If the section is empty, state "Standard Retrieval Logic" and do NOT invent IDs.
3. Explain why you choose to WALK to a specific neighbor or why you decide to stop and READ.
4. If multiple paths exist, explain how the Oracle helps you filter out noise or "harmful" memory paths.

**AVAILABLE TOOLS**:

1. **SEARCH**: Find entry nodes or jump to new nodes.
   - `query`: The search text.
   - `mode`: "hybrid" (RECOMMENDED: combines keyword and semantic), "keyword", or "semantic".
   - *Condition*: Use this if you are nowhere, lost, or need to find a specific entity.

2. **WALK**: Move to a connected node.
   - `node`: The target node name from "Visible Neighbors".
   - *Condition*: Use this to follow a path that might lead to the answer.

3. **READ**: Finish and answer.
   - `answer`: The final concise answer.
   - *Condition*: Use this when you have found the answer in "Current Node Content" **OR in the 'Visible Neighbors' (e.g., timestamps on edges)**.

**RESPONSE FORMAT (JSON ONLY)**:
{
  "chain_of_thought": "...",
  "used_steiner_ids": [],
  "tool": "SEARCH",
  "query": "...",
  "mode": "hybrid"
}
OR
{
  "chain_of_thought": "...",
  "used_steiner_ids": [],
  "tool": "READ",
  "answer": "..."
}
OR
{
  "chain_of_thought": "...",
  "used_steiner_ids": [],
  "tool": "WALK",
  "node": "..."
}
"""

    def _clean_answer(self, text: str) -> str:
        """
        [Sniper Filter] 强力清洗函数
        去除所有 LLM 习惯性的“废话前缀”，只保留核心事实。
        """
        if not text: return ""
        text = text.strip()
        
        # 1. 暴力移除废话前缀
        patterns = [
            r"^Based on (the|this|my) (memory|conversation|graph|context|information).*?(\.|,|:)",
            r"^According to .*?(\.|,|:)",
            r"^The graph indicates (that)?",
            r"^I found (that)?",
            r"^The answer is",
            r"^I can confirm (that)?",
            r"^It is mentioned (that)?",
        ]
        
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
            
        text = text.lstrip(" ,:.-")

        # 2. Yes/No 专用截断策略 (LoCoMo 特化)
        lower_text = text.lower()
        if lower_text.startswith("yes") or lower_text.startswith("no"):
            # 如果解释太长，强制截断，只保留第一句或前几个词
            if len(text.split()) > 10: 
                parts = text.split(',')
                if len(parts) > 1:
                    return parts[0].strip() + " " + " ".join(parts[1].split()[:5])
            
        return text

    def _keyword_search(self, query: str) -> List[str]:
        """Local implementation of keyword search to ensure availability."""
        hits = []
        stop_words = {"what", "which", "who", "where", "when", "how", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on", "at", "did", "does", "do"}
        keywords = [k.lower() for k in re.findall(r'\w+', query) if k.lower() not in stop_words]
        
        # Access the underlying networkx graph
        nx_graph = self.graph.graph
        
        for node, data in nx_graph.nodes(data=True):
            content = f"{node} {data.get('description', '')}".lower()
            score = 0
            for k in keywords:
                if k in node.lower(): score += 10
                elif k in content: score += 1
            if score > 0:
                hits.append((node, score))
        
        hits.sort(key=lambda x: x[1], reverse=True)
        return [h[0] for h in hits[:10]]

    def answer(self, question: str) -> tuple[str, str, List[str]]:
        logger.info(f"❓ Question: {question}")
        
        history = []
        current_nodes = []
        full_cot = []
        all_used_ids = set()
        
        for step in range(self.max_steps):
            # --- 1. Prepare Context ---
            history_str = "\n".join(history[-5:]) if history else "None"
            
            if not current_nodes:
                status_str = "Status: You are currently NOT at any node. You need to SEARCH to find entry points."
                view_str = ""
            else:
                # Get neighbors view
                neighbors_view = self.graph.primitive_get_neighbors(current_nodes)
                # Read current nodes content
                current_content = self.graph.primitive_read(current_nodes)
                status_str = f"Status: You are at nodes: {current_nodes}"
                view_str = f"**Current Node Content**:\n{current_content}\n\n**Visible Neighbors**:\n{neighbors_view}"
                
            prompt = f"""
{self.get_full_prompt()}

**User Question**: "{question}"

**Exploration History**:
{history_str}

**{status_str}**
{view_str}
"""
            # --- 2. LLM Decision ---
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Output valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                content = res.choices[0].message.content
                
                # Clean json string
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                decision = json.loads(content.strip())
                
                # Capture CoT and Used IDs
                step_cot = decision.get("chain_of_thought", "No CoT provided.")
                full_cot.append(f"Step {step+1}: {step_cot}")
                
                used_ids = decision.get("used_steiner_ids", [])
                if isinstance(used_ids, list):
                    for sid in used_ids:
                        all_used_ids.add(sid)
                
                tool = decision.get("tool")
                logger.info(f"Step {step+1}: {tool} - {decision}")
                if used_ids:
                    logger.info(f"🏷️ Answerer used protocols: {used_ids}")

                # --- 3. Execute Tool ---
                if tool == "SEARCH":
                    query = decision.get("query", question)
                    mode = decision.get("mode", "hybrid")
                    
                    results = []
                    
                    def do_keyword():
                        return self._keyword_search(query)
                        
                    def do_semantic():
                        if hasattr(self.graph, "semantic_search"):
                            return self.graph.semantic_search(query)
                        return []

                    if mode == "keyword":
                        results = do_keyword()
                    elif mode == "semantic":
                        results = do_semantic()
                    else: # hybrid
                        k_res = do_keyword()
                        s_res = do_semantic()
                        # Combine: Keyword matches are often more precise for names, Semantic for concepts.
                        # We'll prioritize keyword matches, then semantic.
                        seen = set(k_res)
                        results = k_res + [x for x in s_res if x not in seen]
                            
                    if results:
                        # Keep top 5 to allow broader exploration
                        current_nodes = results[:5]
                        history.append(f"SEARCH({mode}, '{query}') -> Found: {current_nodes}")
                    else:
                        history.append(f"SEARCH({mode}, '{query}') -> Found nothing.")
                
                elif tool == "WALK":
                    target = decision.get("node")
                    if self.graph.graph.has_node(target):
                        current_nodes = [target]
                        history.append(f"WALK -> Moved to {target}")
                    else:
                        history.append(f"WALK -> Failed. Node '{target}' does not exist.")

                elif tool == "READ":
                    ans = self._clean_answer(decision.get("answer"))
                    return ans, "\n".join(full_cot), list(all_used_ids)
                
            except Exception as e:
                logger.error(f"Step failed: {e}")
                history.append(f"Error: {str(e)}")
        
        # Fallback: Try to answer with whatever history we have
        logger.warning("Max steps reached. Attempting to answer from history.")
        try:
            fallback_prompt = f"""You have explored the graph but reached the step limit.
Based on the exploration history below, provide the best possible answer to the question.
If you found partial information, use it to infer the answer according to your common sense and intuition.
Only say "Unknown" if you have absolutely NO relevant information.

**Question**: "{question}"

**History**:
{history_str}

**Current Context**:
{view_str}

Return ONLY the answer text.
"""
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": fallback_prompt}],
                temperature=0.0
            )
            return self._clean_answer(res.choices[0].message.content), "\n".join(full_cot), list(all_used_ids)
        except Exception:
            return "Unknown", "\n".join(full_cot), list(all_used_ids)
