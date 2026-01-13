# import json
# import logging
# import re
# from typing import List, Literal
# from openai import OpenAI
# from amadeus.code.core.graph import MemoryGraph
# from amadeus.code.agents.base import BaseAgent

# logger = logging.getLogger("Amadeus.Answerer")

# class AnswererAgent(BaseAgent):
#     def __init__(self, graph: MemoryGraph, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
#         super().__init__(model_name, api_base, api_key)
#         self.graph = graph
#         self.max_steps = 8  # Increased steps for deeper exploration in granular graphs
#         self.static_prompt = """You are an intelligent Graph RAG Agent.
# **Goal**: Answer the user's question by exploring the knowledge graph.

# **STRATEGY**:
# 1. **SEARCH**: Start here! Use 'hybrid' mode to find multiple entry nodes using both keyword and semantic search.
# 2. **WALK**: Explore the neighborhood of your current nodes to find relevant connections.
# 3. **READ**: Once you have enough information, generate the final answer.

# **AVAILABLE TOOLS**:

# 1. **SEARCH**: Find entry nodes or jump to new nodes.
#    - `query`: The search text.
#    - `mode`: "hybrid" (RECOMMENDED: combines keyword and semantic), "keyword", or "semantic".
#    - *Condition*: Use this if you are nowhere, lost, or need to find a specific entity.

# 2. **WALK**: Move to a connected node.
#    - `node`: The target node name from "Visible Neighbors".
#    - *Condition*: Use this to follow a path that might lead to the answer.

# 3. **READ**: Finish and answer.
#    - `answer`: The final concise answer.
#    - *Condition*: Use this when you have found the answer in "Current Node Content" **OR in the 'Visible Neighbors' (e.g., timestamps on edges)**.

# **RESPONSE FORMAT (JSON ONLY)**:
# {
#   "tool": "SEARCH",
#   "query": "...",
#   "mode": "hybrid"
# }
# OR
# {
#   "tool": "WALK",
#   "node": "TargetNodeName"
# }
# OR
# {
#   "tool": "READ",
#   "answer": "Final Answer Here"
# }
# """

#     def _clean_answer(self, text: str) -> str:
#         """
#         [Sniper Filter] 强力清洗函数
#         去除所有 LLM 习惯性的“废话前缀”，只保留核心事实。
#         """
#         if not text: return ""
#         text = text.strip()
        
#         # 1. 暴力移除废话前缀
#         patterns = [
#             r"^Based on (the|this|my) (memory|conversation|graph|context|information).*?(\.|,|:)",
#             r"^According to .*?(\.|,|:)",
#             r"^The graph indicates (that)?",
#             r"^I found (that)?",
#             r"^The answer is",
#             r"^I can confirm (that)?",
#             r"^It is mentioned (that)?",
#         ]
        
#         for p in patterns:
#             text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
            
#         text = text.lstrip(" ,:.-")

#         # 2. Yes/No 专用截断策略 (LoCoMo 特化)
#         lower_text = text.lower()
#         if lower_text.startswith("yes") or lower_text.startswith("no"):
#             # 如果解释太长，强制截断，只保留第一句或前几个词
#             if len(text.split()) > 10: 
#                 parts = text.split(',')
#                 if len(parts) > 1:
#                     return parts[0].strip() + " " + " ".join(parts[1].split()[:5])
            
#         return text

#     def _keyword_search(self, query: str) -> List[str]:
#         """Local implementation of keyword search to ensure availability."""
#         hits = []
#         stop_words = {"what", "which", "who", "where", "when", "how", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on", "at", "did", "does", "do"}
#         keywords = [k.lower() for k in re.findall(r'\w+', query) if k.lower() not in stop_words]
        
#         # Access the underlying networkx graph
#         nx_graph = self.graph.graph
        
#         for node, data in nx_graph.nodes(data=True):
#             content = f"{node} {data.get('description', '')}".lower()
#             score = 0
#             for k in keywords:
#                 if k in node.lower(): score += 10
#                 elif k in content: score += 1
#             if score > 0:
#                 hits.append((node, score))
        
#         hits.sort(key=lambda x: x[1], reverse=True)
#         return [h[0] for h in hits[:10]]

#     def _format_candidate_descriptions(self, candidates: List[str]) -> str:
#         if not candidates:
#             return "None"
#         lines = []
#         for name in candidates:
#             desc = self.graph.graph.nodes[name].get("description", "")
#             lines.append(f"- [{name}]: {desc}")
#         return "\n".join(lines)

#     def _rank_candidates(self, question: str, candidates: List[str]) -> List[str]:
#         if not candidates:
#             return []
#         candidates_view = self._format_candidate_descriptions(candidates)
#         prompt = f"""You are ranking candidate graph nodes by relevance to the question.
# Return JSON only: {{"ranked_nodes": ["NodeA", "NodeB", "NodeC"]}}

# Rules:
# - Only return JSON.
# - Preserve node names exactly as provided.
# - Use descriptions to decide relevance.

# Question: "{question}"

# Candidates:
# {candidates_view}
# """
#         try:
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"},
#                 temperature=0.0
#             )
#             content = res.choices[0].message.content
#             if "```json" in content:
#                 content = content.split("```json")[1].split("```")[0]
#             elif "```" in content:
#                 content = content.split("```")[1].split("```")[0]
#             data = json.loads(content.strip())
#             ranked = data.get("ranked_nodes", [])
#             if not isinstance(ranked, list):
#                 return candidates
#             ranked_filtered = [n for n in ranked if n in candidates]
#             for n in candidates:
#                 if n not in ranked_filtered:
#                     ranked_filtered.append(n)
#             return ranked_filtered
#         except Exception as e:
#             logger.warning(f"Rank candidates failed: {e}")
#             return candidates

#     def _check_sufficiency(
#         self,
#         question: str,
#         current_content: str,
#         neighbors_view: str,
#         visited_nodes: str,
#         visited_edges: str,
#         search_candidates_view: str
#     ) -> tuple[bool, str]:
#         prompt = f"""Decide if the current graph context is sufficient to answer the user's question.
# Return JSON only: {{"sufficient": true|false, "reason": "short reason"}}.

# Question: "{question}"

# Current Node Content:
# {current_content}

# Visible Neighbors:
# {neighbors_view}

# Search Candidates (Descriptions):
# {search_candidates_view}

# Visited Nodes:
# {visited_nodes}

# Visited Edges:
# {visited_edges}
# """
#         try:
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"},
#                 temperature=0.0
#             )
#             content = res.choices[0].message.content
#             if "```json" in content:
#                 content = content.split("```json")[1].split("```")[0]
#             elif "```" in content:
#                 content = content.split("```")[1].split("```")[0]
#             data = json.loads(content.strip())
#             sufficient = bool(data.get("sufficient", False))
#             reason = str(data.get("reason", "")).strip()
#             return sufficient, reason
#         except Exception as e:
#             logger.warning(f"Sufficiency check failed: {e}")
#             return False, str(e)

#     def _answer_from_context(
#         self,
#         question: str,
#         history_str: str,
#         visited_context_str: str,
#         view_str: str,
#         search_candidates_view: str,
#         reason_line: str
#     ) -> str:
#         prompt = f"""You have explored the graph. {reason_line}
# Based on the exploration history below, provide the best possible answer to the question.
# If you found partial information, use it to infer the answer according to your common sense and intuition.


# **Question**: "{question}"

# **History**:
# {history_str}

# **Visited Context**:
# {visited_context_str}

# **Search Candidates (Descriptions)**:
# {search_candidates_view}

# **Current Context**:
# {view_str}

# Return ONLY the answer text.
# """
#         try:
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0
#             )
#             return self._clean_answer(res.choices[0].message.content)
#         except Exception:
#             return "Unknown"

#     def answer(self, question: str) -> str:
#         logger.info(f"❓ Question: {question}")
        
#         history = []
#         current_nodes = []
#         visited_node_order = []
#         visited_node_content = {}
#         visited_edge_order = []
#         visited_edge_set = set()
#         visited_context_str = "None"
#         search_candidates = []
#         search_candidate_set = set()
#         search_candidates_view = "None"
#         ranked_candidates = []
#         walk_queue = []
#         walked_candidates = set()
#         walked_count = 0
#         max_ranked_walks = 5
#         next_walk_target = None
#         force_read = False
        
#         for step in range(self.max_steps):
#             # --- 1. Prepare Context ---
#             history_str = "\n".join(history[-5:]) if history else "None"
#             performed_ranked_walk = False
#             if next_walk_target and not force_read:
#                 target = next_walk_target
#                 next_walk_target = None
#                 performed_ranked_walk = True
#                 if self.graph.graph.has_node(target):
#                     current_nodes = [target]
#                     walked_candidates.add(target)
#                     walked_count += 1
#                     history.append(f"WALK (ranked) -> Moved to {target}")
#                     logger.info(f"WALK (ranked) -> Moved to {target}")
#                 else:
#                     history.append(f"WALK (ranked) -> Failed. Node '{target}' does not exist.")
#                     logger.info(f"WALK (ranked) -> Failed. Node '{target}' does not exist.")
            
#             if not current_nodes:
#                 status_str = "Status: You are currently NOT at any node. You need to SEARCH to find entry points."
#                 current_content = ""
#                 neighbors_view = ""
#                 view_str = ""
#             else:
#                 # Get neighbors view
#                 neighbors_view = self.graph.primitive_get_neighbors(current_nodes)
#                 # Read current nodes content
#                 current_content = self.graph.primitive_read(current_nodes)
#                 # Cache visited nodes and edges for later answering.
#                 nx_graph = self.graph.graph
#                 for name in current_nodes:
#                     if name not in visited_node_content:
#                         visited_node_content[name] = self.graph.primitive_read([name])
#                         visited_node_order.append(name)
#                     for _, target, data in nx_graph.out_edges(name, data=True):
#                         rel = data.get('relation', 'related')
#                         edge_ts = data.get('timestamp')
#                         edge_ts_str = f" [Time: {edge_ts}]" if edge_ts else ""
#                         edge_line = f"[{name}] --[{rel}{edge_ts_str}]--> [{target}]"
#                         if edge_line not in visited_edge_set:
#                             visited_edge_set.add(edge_line)
#                             visited_edge_order.append(edge_line)
#                 status_str = f"Status: You are at nodes: {current_nodes}"
#                 view_str = f"**Current Node Content**:\n{current_content}\n\n**Visible Neighbors**:\n{neighbors_view}"

#             if visited_node_order:
#                 nodes_view = "\n".join(visited_node_content[n] for n in visited_node_order)
#             else:
#                 nodes_view = "None"
#             if visited_edge_order:
#                 edges_view = "\n".join(visited_edge_order)
#             else:
#                 edges_view = "None"
#             visited_context_str = f"**Visited Nodes**:\n{nodes_view}\n\n**Visited Edges**:\n{edges_view}"
                
#             prompt = f"""
# {self.get_full_prompt()}

# **User Question**: "{question}"

# **Exploration History**:
# {history_str}

# **Visited Context**:
# {visited_context_str}

# **Search Candidates (Descriptions)**:
# {search_candidates_view}

# **{status_str}**
# {view_str}
# """
#             if performed_ranked_walk:
#                 sufficient, reason = self._check_sufficiency(
#                     question,
#                     current_content,
#                     neighbors_view,
#                     nodes_view,
#                     edges_view,
#                     search_candidates_view
#                 )
#                 history.append(f"SUFFICIENT -> {sufficient} ({reason})")
#                 logger.info(f"SUFFICIENT -> {sufficient} ({reason})")
#                 if sufficient:
#                     walk_queue = []
#                 else:
#                     if walked_count < max_ranked_walks and walk_queue:
#                         next_walk_target = walk_queue.pop(0)
#                         continue
#                     force_read = True
#                     history.append("FORCE READ: ranked walk limit reached.")
#                     logger.info("FORCE READ: ranked walk limit reached.")
#                     return self._answer_from_context(
#                         question,
#                         history_str,
#                         visited_context_str,
#                         view_str,
#                         search_candidates_view,
#                         "Ranked walk limit reached."
#                     )
#             # --- 2. LLM Decision ---
#             try:
#                 res = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant. Output valid JSON only."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     response_format={"type": "json_object"},
#                     temperature=0.0
#                 )
#                 content = res.choices[0].message.content
                
#                 # Clean json string
#                 if "```json" in content:
#                     content = content.split("```json")[1].split("```")[0]
#                 elif "```" in content:
#                     content = content.split("```")[1].split("```")[0]
                
#                 decision = json.loads(content.strip())
#                 tool = decision.get("tool")
#                 logger.info(f"Step {step+1}: {tool} - {decision}")

#                 # --- 3. Execute Tool ---
#                 if tool == "SEARCH":
#                     query = decision.get("query", question)
#                     mode = decision.get("mode", "hybrid")
                    
#                     results = []
                    
#                     def do_keyword():
#                         return self._keyword_search(query)
                        
#                     def do_semantic():
#                         if hasattr(self.graph, "semantic_search"):
#                             return self.graph.semantic_search(query)
#                         return []

#                     if mode == "keyword":
#                         results = do_keyword()
#                     elif mode == "semantic":
#                         results = do_semantic()
#                     else: # hybrid
#                         k_res = do_keyword()
#                         s_res = do_semantic()
#                         # Combine: Keyword matches are often more precise for names, Semantic for concepts.
#                         # We'll prioritize keyword matches, then semantic.
#                         seen = set(k_res)
#                         results = k_res + [x for x in s_res if x not in seen]
                            
#                     if results:
#                         # Keep top 5 to allow broader exploration
#                         current_nodes = results[:5]
#                         history.append(f"SEARCH({mode}, '{query}') -> Found: {current_nodes}")
#                         for name in current_nodes:
#                             if name not in search_candidate_set:
#                                 search_candidate_set.add(name)
#                                 search_candidates.append(name)
#                         search_candidates_view = self._format_candidate_descriptions(search_candidates)
#                         ranked_candidates = self._rank_candidates(question, current_nodes)
#                         history.append(f"RANK -> {ranked_candidates}")
#                         logger.info(f"RANK -> {ranked_candidates}")
#                         walk_queue = [n for n in ranked_candidates if n not in walked_candidates]
#                         walked_count = 0
#                         if walk_queue:
#                             next_walk_target = walk_queue.pop(0)
#                     else:
#                         history.append(f"SEARCH({mode}, '{query}') -> Found nothing.")
                
#                 elif tool == "WALK":
#                     target = decision.get("node")
#                     if self.graph.graph.has_node(target):
#                         current_nodes = [target]
#                         history.append(f"WALK -> Moved to {target}")
#                     else:
#                         history.append(f"WALK -> Failed. Node '{target}' does not exist.")

#                 elif tool == "READ":
#                     ans = self._clean_answer(decision.get("answer"))
#                     return ans
                
#             except Exception as e:
#                 logger.error(f"Step failed: {e}")
#                 history.append(f"Error: {str(e)}")
        
#         # Fallback: Try to answer with whatever history we have
#         logger.warning("Max steps reached. Attempting to answer from history.")
#         return self._answer_from_context(
#             question,
#             history_str,
#             visited_context_str,
#             view_str,
#             search_candidates_view,
#             "Step limit reached."
#         )
# import json
# import logging
# import re
# from typing import List, Literal
# from openai import OpenAI
# from amadeus.code.core.graph import MemoryGraph
# from amadeus.code.agents.base import BaseAgent

# logger = logging.getLogger("Amadeus.Answerer")

# class AnswererAgent(BaseAgent):
#     def __init__(self, graph: MemoryGraph, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
#         super().__init__(model_name, api_base, api_key)
#         self.graph = graph
#         self.max_steps = 8  # Increased steps for deeper exploration in granular graphs
#         self.static_prompt = """You are an intelligent Graph RAG Agent.
# **Goal**: Answer the user's question by exploring the knowledge graph.

# **STRATEGY**:
# 1. **SEARCH**: Start here! Use 'hybrid' mode to find multiple entry nodes using both keyword and semantic search.
# 2. **WALK**: Explore the neighborhood of your current nodes to find relevant connections.
# 3. **READ**: Once you have enough information, generate the final answer.

# **AVAILABLE TOOLS**:

# 1. **SEARCH**: Find entry nodes or jump to new nodes.
#    - `query`: The search text.
#    - `mode`: "hybrid" (RECOMMENDED: combines keyword and semantic), "keyword", or "semantic".
#    - *Condition*: Use this if you are nowhere, lost, or need to find a specific entity.

# 2. **WALK**: Move to a connected node.
#    - `node`: The target node name from "Visible Neighbors".
#    - *Condition*: Use this to follow a path that might lead to the answer.

# 3. **READ**: Finish and answer.
#    - `answer`: The final concise answer.
#    - *Condition*: Use this when you have found the answer in "Current Node Content" **OR in the 'Visible Neighbors' (e.g., timestamps on edges)**.

# **RESPONSE FORMAT (JSON ONLY)**:
# {
#   "tool": "SEARCH",
#   "query": "...",
#   "mode": "hybrid"
# }
# OR
# {
#   "tool": "WALK",
#   "node": "TargetNodeName"
# }
# OR
# {
#   "tool": "READ",
#   "answer": "Final Answer Here"
# }
# """

#     def _clean_answer(self, text: str) -> str:
#         """
#         [Sniper Filter] 强力清洗函数
#         去除所有 LLM 习惯性的“废话前缀”，只保留核心事实。
#         """
#         if not text: return ""
#         text = text.strip()
        
#         # 1. 暴力移除废话前缀
#         patterns = [
#             r"^Based on (the|this|my) (memory|conversation|graph|context|information).*?(\.|,|:)",
#             r"^According to .*?(\.|,|:)",
#             r"^The graph indicates (that)?",
#             r"^I found (that)?",
#             r"^The answer is",
#             r"^I can confirm (that)?",
#             r"^It is mentioned (that)?",
#         ]
        
#         for p in patterns:
#             text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
            
#         text = text.lstrip(" ,:.-")

#         # 2. Yes/No 专用截断策略 (LoCoMo 特化)
#         lower_text = text.lower()
#         if lower_text.startswith("yes") or lower_text.startswith("no"):
#             # 如果解释太长，强制截断，只保留第一句或前几个词
#             if len(text.split()) > 10: 
#                 parts = text.split(',')
#                 if len(parts) > 1:
#                     return parts[0].strip() + " " + " ".join(parts[1].split()[:5])
            
#         return text

#     def _keyword_search(self, query: str) -> List[str]:
#         """Local implementation of keyword search to ensure availability."""
#         hits = []
#         stop_words = {"what", "which", "who", "where", "when", "how", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on", "at", "did", "does", "do"}
#         keywords = self._llm_extract_keywords(query)
#         if not keywords:
#             keywords = [k.lower() for k in re.findall(r'\w+', query) if k.lower() not in stop_words]
#         else:
#             keywords = [k.lower() for k in keywords if k.lower() not in stop_words]
        
#         # Access the underlying networkx graph
#         nx_graph = self.graph.graph
        
#         for node, data in nx_graph.nodes(data=True):
#             content = f"{node} {data.get('description', '')}".lower()
#             score = 0
#             for k in keywords:
#                 if k in node.lower(): score += 10
#                 elif k in content: score += 1
#             if score > 0:
#                 hits.append((node, score))
        
#         hits.sort(key=lambda x: x[1], reverse=True)
#         return [h[0] for h in hits[:10]]

#     def _llm_extract_keywords(self, query: str) -> List[str]:
#         """Extract keywords using LLM, with a safe fallback."""
#         prompt = (
#             "You are a keyword extractor. Given a question, extract 3-8 concise keywords or short phrases.\n"
#             "Requirements:\n"
#             "- Output JSON only in the form: {\"keywords\": [\"...\", \"...\"]}\n"
#             "- Keep each keyword 1-4 words, preserve proper nouns and domain terms.\n"
#             "- No explanations, no extra text.\n"
#             f"Question: {query}"
#         )
#         try:
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"},
#                 temperature=0.0
#             )
#             content = res.choices[0].message.content
#             data = json.loads(content)
#             keywords = data.get("keywords", [])
#             if isinstance(keywords, list):
#                 return [str(k).strip() for k in keywords if str(k).strip()]
#         except Exception:
#             return []
#         return []

#     def answer(self, question: str) -> str:
#         logger.info(f"❓ Question: {question}")
        
#         history = []
#         current_nodes = []
#         visited_node_order = []
#         visited_node_content = {}
#         visited_edge_order = []
#         visited_edge_set = set()
#         visited_context_str = "None"
        
#         for step in range(self.max_steps):
#             # --- 1. Prepare Context ---
#             history_str = "\n".join(history[-5:]) if history else "None"
            
#             if not current_nodes:
#                 status_str = "Status: You are currently NOT at any node. You need to SEARCH to find entry points."
#                 view_str = ""
#             else:
#                 # Get neighbors view
#                 neighbors_view = self.graph.primitive_get_neighbors(current_nodes)
#                 # Read current nodes content
#                 current_content = self.graph.primitive_read(current_nodes)
#                 # Cache visited nodes and edges for later answering.
#                 nx_graph = self.graph.graph
#                 for name in current_nodes:
#                     if name not in visited_node_content:
#                         visited_node_content[name] = self.graph.primitive_read([name])
#                         visited_node_order.append(name)
#                     for _, target, data in nx_graph.out_edges(name, data=True):
#                         rel = data.get('relation', 'related')
#                         edge_ts = data.get('timestamp')
#                         edge_ts_str = f" [Time: {edge_ts}]" if edge_ts else ""
#                         edge_line = f"[{name}] --[{rel}{edge_ts_str}]--> [{target}]"
#                         if edge_line not in visited_edge_set:
#                             visited_edge_set.add(edge_line)
#                             visited_edge_order.append(edge_line)
#                 status_str = f"Status: You are at nodes: {current_nodes}"
#                 view_str = f"**Current Node Content**:\n{current_content}\n\n**Visible Neighbors**:\n{neighbors_view}"

#             if visited_node_order:
#                 nodes_view = "\n".join(visited_node_content[n] for n in visited_node_order[-5:])
#             else:
#                 nodes_view = "None"
#             if visited_edge_order:
#                 edges_view = "\n".join(visited_edge_order[-5:])
#             else:
#                 edges_view = "None"
#             visited_context_str = f"**Visited Nodes**:\n{nodes_view}\n\n**Visited Edges**:\n{edges_view}"
                
#             prompt = f"""
# {self.get_full_prompt()}

# **User Question**: "{question}"

# **Exploration History**:
# {history_str}

# **Visited Context**:
# {visited_context_str}

# **{status_str}**
# {view_str}
# """
#             # --- 2. LLM Decision ---
#             try:
#                 res = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant. Output valid JSON only."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     response_format={"type": "json_object"},
#                     temperature=0.0
#                 )
#                 content = res.choices[0].message.content
                
#                 # Clean json string
#                 if "```json" in content:
#                     content = content.split("```json")[1].split("```")[0]
#                 elif "```" in content:
#                     content = content.split("```")[1].split("```")[0]
                
#                 decision = json.loads(content.strip())
#                 tool = decision.get("tool")
#                 logger.info(f"Step {step+1}: {tool} - {decision}")

#                 # --- 3. Execute Tool ---
#                 if tool == "SEARCH":
#                     query = decision.get("query", question)
#                     mode = decision.get("mode", "hybrid")
                    
#                     results = []
                    
#                     def do_keyword():
#                         return self._keyword_search(query)
                        
#                     def do_semantic():
#                         if hasattr(self.graph, "semantic_search"):
#                             return self.graph.semantic_search(query)
#                         return []

#                     if mode == "keyword":
#                         results = do_keyword()
#                     elif mode == "semantic":
#                         results = do_semantic()
#                     else: # hybrid
#                         k_res = do_keyword()
#                         s_res = do_semantic()
#                         # Combine: Keyword matches are often more precise for names, Semantic for concepts.
#                         # We'll prioritize keyword matches, then semantic.
#                         seen = set(k_res)
#                         results = k_res + [x for x in s_res if x not in seen]
                            
#                     if results:
#                         # Keep top 5 to allow broader exploration
#                         current_nodes = results[:5]
#                         history.append(f"SEARCH({mode}, '{query}') -> Found: {current_nodes}")
#                     else:
#                         history.append(f"SEARCH({mode}, '{query}') -> Found nothing.")
                
#                 elif tool == "WALK":
#                     target = decision.get("node")
#                     if self.graph.graph.has_node(target):
#                         current_nodes = [target]
#                         history.append(f"WALK -> Moved to {target}")
#                     else:
#                         history.append(f"WALK -> Failed. Node '{target}' does not exist.")

#                 elif tool == "READ":
#                     ans = self._clean_answer(decision.get("answer"))
#                     return ans
                
#             except Exception as e:
#                 logger.error(f"Step failed: {e}")
#                 history.append(f"Error: {str(e)}")
        
#         # Fallback: Try to answer with whatever history we have
#         logger.warning("Max steps reached. Attempting to answer from history.")
#         try:
#             fallback_prompt = f"""You have explored the graph but reached the step limit.
# Based on the exploration history below, provide the best possible answer to the question.
# If you found partial information, use it to infer the answer according to your common sense and intuition.
# Only say "Unknown" if you have absolutely NO relevant information.

# **Question**: "{question}"

# **History**:
# {history_str}

# **Visited Context**:
# {visited_context_str}

# **Current Context**:
# {view_str}

# Return ONLY the answer text.
# """
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": fallback_prompt}],
#                 temperature=0.0
#             )
#             return self._clean_answer(res.choices[0].message.content)
#         except Exception:
#             return "Unknown"
# import json
# import logging
# import re
# from typing import List, Literal
# from openai import OpenAI
# from amadeus.code.core.graph import MemoryGraph
# from amadeus.code.agents.base import BaseAgent

# logger = logging.getLogger("Amadeus.Answerer")

# class AnswererAgent(BaseAgent):
#     def __init__(self, graph: MemoryGraph, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
#         super().__init__(model_name, api_base, api_key)
#         self.graph = graph
#         self.max_steps = 8  # Increased steps for deeper exploration in granular graphs
#         self.static_prompt = """You are an intelligent Graph RAG Agent.
# **Goal**: Answer the user's question by exploring the knowledge graph.

# **STRATEGY**:
# 1. **SEARCH**: Start here! Use 'hybrid' mode to find multiple entry nodes using both keyword and semantic search.
# 2. **WALK**: Explore the neighborhood of your current nodes to find relevant connections.
# 3. **READ**: Once you have enough information, generate the final answer.

# **AVAILABLE TOOLS**:

# 1. **SEARCH**: Find entry nodes or jump to new nodes.
#    - `query`: The search text.
#    - `mode`: "hybrid" (RECOMMENDED: combines keyword and semantic), "keyword", or "semantic".
#    - *Condition*: Use this if you are nowhere, lost, or need to find a specific entity.

# 2. **WALK**: Move to a connected node.
#    - `node`: The target node name from "Visible Neighbors".
#    - *Condition*: Use this to follow a path that might lead to the answer.

# 3. **READ**: Finish and answer.
#    - `answer`: The final concise answer.
#    - *Condition*: Use this when you have found the answer in "Current Node Content" **OR in the 'Visible Neighbors' (e.g., timestamps on edges)**.

# **RESPONSE FORMAT (JSON ONLY)**:
# {
#   "tool": "SEARCH",
#   "query": "...",
#   "mode": "hybrid"
# }
# OR
# {
#   "tool": "WALK",
#   "node": "TargetNodeName"
# }
# OR
# {
#   "tool": "READ",
#   "answer": "Final Answer Here"
# }
# """

#     def _clean_answer(self, text: str) -> str:
#         """
#         [Sniper Filter] 强力清洗函数
#         去除所有 LLM 习惯性的“废话前缀”，只保留核心事实。
#         """
#         if not text: return ""
#         text = text.strip()
        
#         # 1. 暴力移除废话前缀
#         patterns = [
#             r"^Based on (the|this|my) (memory|conversation|graph|context|information).*?(\.|,|:)",
#             r"^According to .*?(\.|,|:)",
#             r"^The graph indicates (that)?",
#             r"^I found (that)?",
#             r"^The answer is",
#             r"^I can confirm (that)?",
#             r"^It is mentioned (that)?",
#         ]
        
#         for p in patterns:
#             text = re.sub(p, "", text, flags=re.IGNORECASE).strip()
            
#         text = text.lstrip(" ,:.-")

#         # 2. Yes/No 专用截断策略 (LoCoMo 特化)
#         lower_text = text.lower()
#         if lower_text.startswith("yes") or lower_text.startswith("no"):
#             # 如果解释太长，强制截断，只保留第一句或前几个词
#             if len(text.split()) > 10: 
#                 parts = text.split(',')
#                 if len(parts) > 1:
#                     return parts[0].strip() + " " + " ".join(parts[1].split()[:5])
            
#         return text

#     def _keyword_search(self, query: str) -> List[str]:
#         """Local implementation of keyword search to ensure availability."""
#         hits = []
#         stop_words = {"what", "which", "who", "where", "when", "how", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on", "at", "did", "does", "do"}
#         keywords = [k.lower() for k in re.findall(r'\w+', query) if k.lower() not in stop_words]
        
#         # Access the underlying networkx graph
#         nx_graph = self.graph.graph
        
#         for node, data in nx_graph.nodes(data=True):
#             content = f"{node} {data.get('description', '')}".lower()
#             score = 0
#             for k in keywords:
#                 if k in node.lower(): score += 10
#                 elif k in content: score += 1
#             if score > 0:
#                 hits.append((node, score))
        
#         hits.sort(key=lambda x: x[1], reverse=True)
#         return [h[0] for h in hits[:10]]

#     def _format_candidate_descriptions(self, candidates: List[str]) -> str:
#         if not candidates:
#             return "None"
#         lines = []
#         for name in candidates:
#             desc = self.graph.graph.nodes[name].get("description", "")
#             lines.append(f"- [{name}]: {desc}")
#         return "\n".join(lines)

#     def _rank_candidates(self, question: str, candidates: List[str]) -> List[str]:
#         if not candidates:
#             return []
#         candidates_view = self._format_candidate_descriptions(candidates)
#         prompt = f"""You are ranking candidate graph nodes by relevance to the question.
# Return JSON only: {{"ranked_nodes": ["NodeA", "NodeB", "NodeC"]}}

# Rules:
# - Only return JSON.
# - Preserve node names exactly as provided.
# - Use descriptions to decide relevance.

# Question: "{question}"

# Candidates:
# {candidates_view}
# """
#         try:
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"},
#                 temperature=0.0
#             )
#             content = res.choices[0].message.content
#             if "```json" in content:
#                 content = content.split("```json")[1].split("```")[0]
#             elif "```" in content:
#                 content = content.split("```")[1].split("```")[0]
#             data = json.loads(content.strip())
#             ranked = data.get("ranked_nodes", [])
#             if not isinstance(ranked, list):
#                 return candidates
#             ranked_filtered = [n for n in ranked if n in candidates]
#             for n in candidates:
#                 if n not in ranked_filtered:
#                     ranked_filtered.append(n)
#             return ranked_filtered
#         except Exception as e:
#             logger.warning(f"Rank candidates failed: {e}")
#             return candidates

#     def _check_sufficiency(
#         self,
#         question: str,
#         current_content: str,
#         neighbors_view: str,
#         visited_nodes: str,
#         visited_edges: str,
#         search_candidates_view: str
#     ) -> tuple[bool, str]:
#         prompt = f"""Decide if the current graph context is sufficient to answer the user's question.
# Return JSON only: {{"sufficient": true|false, "reason": "short reason"}}.

# Question: "{question}"

# Current Node Content:
# {current_content}

# Visible Neighbors:
# {neighbors_view}

# Search Candidates (Descriptions):
# {search_candidates_view}

# Visited Nodes:
# {visited_nodes}

# Visited Edges:
# {visited_edges}
# """
#         try:
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"},
#                 temperature=0.0
#             )
#             content = res.choices[0].message.content
#             if "```json" in content:
#                 content = content.split("```json")[1].split("```")[0]
#             elif "```" in content:
#                 content = content.split("```")[1].split("```")[0]
#             data = json.loads(content.strip())
#             sufficient = bool(data.get("sufficient", False))
#             reason = str(data.get("reason", "")).strip()
#             return sufficient, reason
#         except Exception as e:
#             logger.warning(f"Sufficiency check failed: {e}")
#             return False, str(e)

#     def _answer_from_context(
#         self,
#         question: str,
#         history_str: str,
#         visited_context_str: str,
#         view_str: str,
#         search_candidates_view: str,
#         reason_line: str
#     ) -> str:
#         prompt = f"""You have explored the graph. {reason_line}
# Based on the exploration history below, provide the best possible answer to the question.
# If you found partial information, use it to infer the answer according to your common sense and intuition.
# Only say "Unknown" if you have absolutely NO relevant information.

# **Question**: "{question}"

# **History**:
# {history_str}

# **Visited Context**:
# {visited_context_str}

# **Search Candidates (Descriptions)**:
# {search_candidates_view}

# **Current Context**:
# {view_str}

# Return ONLY the answer text.
# """
#         try:
#             res = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.0
#             )
#             return self._clean_answer(res.choices[0].message.content)
#         except Exception:
#             return "Unknown"

#     def answer(self, question: str) -> str:
#         logger.info(f"❓ Question: {question}")
        
#         history = []
#         current_nodes = []
#         visited_node_order = []
#         visited_node_content = {}
#         visited_edge_order = []
#         visited_edge_set = set()
#         visited_context_str = "None"
#         search_candidates = []
#         search_candidate_set = set()
#         search_candidates_view = "None"
#         ranked_candidates = []
#         walk_queue = []
#         walked_candidates = set()
#         walked_count = 0
#         max_ranked_walks = 5
#         next_walk_target = None
#         force_read = False
        
#         for step in range(self.max_steps):
#             # --- 1. Prepare Context ---
#             history_str = "\n".join(history[-5:]) if history else "None"
#             performed_ranked_walk = False
#             if next_walk_target and not force_read:
#                 target = next_walk_target
#                 next_walk_target = None
#                 performed_ranked_walk = True
#                 if self.graph.graph.has_node(target):
#                     current_nodes = [target]
#                     walked_candidates.add(target)
#                     walked_count += 1
#                     history.append(f"WALK (ranked) -> Moved to {target}")
#                     logger.info(f"WALK (ranked) -> Moved to {target}")
#                 else:
#                     history.append(f"WALK (ranked) -> Failed. Node '{target}' does not exist.")
#                     logger.info(f"WALK (ranked) -> Failed. Node '{target}' does not exist.")
            
#             if not current_nodes:
#                 status_str = "Status: You are currently NOT at any node. You need to SEARCH to find entry points."
#                 current_content = ""
#                 neighbors_view = ""
#                 view_str = ""
#             else:
#                 # Get neighbors view
#                 neighbors_view = self.graph.primitive_get_neighbors(current_nodes)
#                 # Read current nodes content
#                 current_content = self.graph.primitive_read(current_nodes)
#                 # Cache visited nodes and edges for later answering.
#                 nx_graph = self.graph.graph
#                 for name in current_nodes:
#                     if name not in visited_node_content:
#                         visited_node_content[name] = self.graph.primitive_read([name])
#                         visited_node_order.append(name)
#                     for _, target, data in nx_graph.out_edges(name, data=True):
#                         rel = data.get('relation', 'related')
#                         edge_ts = data.get('timestamp')
#                         edge_ts_str = f" [Time: {edge_ts}]" if edge_ts else ""
#                         edge_line = f"[{name}] --[{rel}{edge_ts_str}]--> [{target}]"
#                         if edge_line not in visited_edge_set:
#                             visited_edge_set.add(edge_line)
#                             visited_edge_order.append(edge_line)
#                 status_str = f"Status: You are at nodes: {current_nodes}"
#                 view_str = f"**Current Node Content**:\n{current_content}\n\n**Visible Neighbors**:\n{neighbors_view}"

#             if visited_node_order:
#                 nodes_view = "\n".join(visited_node_content[n] for n in visited_node_order)
#             else:
#                 nodes_view = "None"
#             if visited_edge_order:
#                 edges_view = "\n".join(visited_edge_order)
#             else:
#                 edges_view = "None"
#             visited_context_str = f"**Visited Nodes**:\n{nodes_view}\n\n**Visited Edges**:\n{edges_view}"
                
#             prompt = f"""
# {self.get_full_prompt()}

# **User Question**: "{question}"

# **Exploration History**:
# {history_str}

# **Visited Context**:
# {visited_context_str}

# **Search Candidates (Descriptions)**:
# {search_candidates_view}

# **{status_str}**
# {view_str}
# """
#             if performed_ranked_walk:
#                 sufficient, reason = self._check_sufficiency(
#                     question,
#                     current_content,
#                     neighbors_view,
#                     nodes_view,
#                     edges_view,
#                     search_candidates_view
#                 )
#                 history.append(f"SUFFICIENT -> {sufficient} ({reason})")
#                 logger.info(f"SUFFICIENT -> {sufficient} ({reason})")
#                 if sufficient:
#                     walk_queue = []
#                 else:
#                     if walked_count < max_ranked_walks and walk_queue:
#                         next_walk_target = walk_queue.pop(0)
#                         continue
#                     force_read = True
#                     history.append("FORCE READ: ranked walk limit reached.")
#                     logger.info("FORCE READ: ranked walk limit reached.")
#                     return self._answer_from_context(
#                         question,
#                         history_str,
#                         visited_context_str,
#                         view_str,
#                         search_candidates_view,
#                         "Ranked walk limit reached."
#                     )
#             # --- 2. LLM Decision ---
#             try:
#                 res = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant. Output valid JSON only."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     response_format={"type": "json_object"},
#                     temperature=0.0
#                 )
#                 content = res.choices[0].message.content
                
#                 # Clean json string
#                 if "```json" in content:
#                     content = content.split("```json")[1].split("```")[0]
#                 elif "```" in content:
#                     content = content.split("```")[1].split("```")[0]
                
#                 decision = json.loads(content.strip())
#                 tool = decision.get("tool")
#                 logger.info(f"Step {step+1}: {tool} - {decision}")

#                 # --- 3. Execute Tool ---
#                 if tool == "SEARCH":
#                     query = decision.get("query", question)
#                     mode = decision.get("mode", "hybrid")
                    
#                     results = []
                    
#                     def do_keyword():
#                         return self._keyword_search(query)
                        
#                     def do_semantic():
#                         if hasattr(self.graph, "semantic_search"):
#                             return self.graph.semantic_search(query)
#                         return []

#                     if mode == "keyword":
#                         results = do_keyword()
#                     elif mode == "semantic":
#                         results = do_semantic()
#                     else: # hybrid
#                         k_res = do_keyword()
#                         s_res = do_semantic()
#                         # Combine: Keyword matches are often more precise for names, Semantic for concepts.
#                         # We'll prioritize keyword matches, then semantic.
#                         seen = set(k_res)
#                         results = k_res + [x for x in s_res if x not in seen]
                            
#                     if results:
#                         # Keep top 5 to allow broader exploration
#                         current_nodes = results[:5]
#                         history.append(f"SEARCH({mode}, '{query}') -> Found: {current_nodes}")
#                         for name in current_nodes:
#                             if name not in search_candidate_set:
#                                 search_candidate_set.add(name)
#                                 search_candidates.append(name)
#                         search_candidates_view = self._format_candidate_descriptions(search_candidates)
#                         ranked_candidates = self._rank_candidates(question, current_nodes)
#                         history.append(f"RANK -> {ranked_candidates}")
#                         logger.info(f"RANK -> {ranked_candidates}")
#                         walk_queue = [n for n in ranked_candidates if n not in walked_candidates]
#                         walked_count = 0
#                         if walk_queue:
#                             next_walk_target = walk_queue.pop(0)
#                     else:
#                         history.append(f"SEARCH({mode}, '{query}') -> Found nothing.")
                
#                 elif tool == "WALK":
#                     target = decision.get("node")
#                     if self.graph.graph.has_node(target):
#                         current_nodes = [target]
#                         history.append(f"WALK -> Moved to {target}")
#                     else:
#                         history.append(f"WALK -> Failed. Node '{target}' does not exist.")

#                 elif tool == "READ":
#                     ans = self._clean_answer(decision.get("answer"))
#                     return ans
                
#             except Exception as e:
#                 logger.error(f"Step failed: {e}")
#                 history.append(f"Error: {str(e)}")
        
#         # Fallback: Try to answer with whatever history we have
#         logger.warning("Max steps reached. Attempting to answer from history.")
#         return self._answer_from_context(
#             question,
#             history_str,
#             visited_context_str,
#             view_str,
#             search_candidates_view,
#             "Step limit reached."
#         )
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
        self.max_steps = 8  # Increased steps for deeper exploration in granular graphs
        self.static_prompt = """You are an intelligent Graph RAG Agent.
**Goal**: Answer the user's question by exploring the knowledge graph.

**STRATEGY**:
1. **SEARCH**: Start here! Use 'hybrid' mode to find multiple entry nodes using both keyword and semantic search.
2. **WALK**: Explore the neighborhood of your current nodes to find relevant connections.
3. **READ**: Once you have enough information, generate the final answer.

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
  "tool": "SEARCH",
  "query": "...",
  "mode": "hybrid"
}
OR
{
  "tool": "WALK",
  "node": "TargetNodeName"
}
OR
{
  "tool": "READ",
  "answer": "Final Answer Here"
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

    def _prepare_query_keywords(self, query: str) -> List[str]:
        stop_words = {"what", "which", "who", "where", "when", "how", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on", "at", "did", "does", "do"}
        keywords = self._llm_extract_keywords(query)
        if not keywords:
            keywords = [k.lower() for k in re.findall(r'\w+', query) if k.lower() not in stop_words]
        else:
            keywords = [k.lower() for k in keywords if k.lower() not in stop_words]
        seen = set()
        filtered = []
        for k in keywords:
            if k and k not in seen:
                seen.add(k)
                filtered.append(k)
        return filtered

    def _keyword_search(self, query: str, keywords: List[str] = None) -> List[str]:
        """Local implementation of keyword search to ensure availability."""
        hits = []
        if keywords is None:
            keywords = self._prepare_query_keywords(query)
        
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

    def _score_neighbor_candidate(
        self,
        keywords: List[str],
        source: str,
        target: str,
        relation: str,
        timestamp: str,
        desc: str
    ) -> int:
        score = 0
        src = source.lower()
        tgt = target.lower()
        rel = (relation or "").lower()
        ts = (timestamp or "").lower()
        dsc = (desc or "").lower()
        for k in keywords:
            if k in src or k in tgt:
                score += 3
            if k in rel:
                score += 2
            if k in ts:
                score += 2
            if k in dsc:
                score += 1
        return score

    def _format_neighbor_evidence(
        self,
        source: str,
        target: str,
        relation: str,
        timestamp: str,
        desc: str,
        desc_limit: int
    ) -> str:
        rel = relation or "related"
        ts_part = f" @ {timestamp}" if timestamp else ""
        trimmed = (desc or "").strip()
        if len(trimmed) > desc_limit:
            trimmed = trimmed[:desc_limit].rstrip() + "..."
        return f"[{source}] --[{rel}{ts_part}]--> [{target}] | desc: \"{trimmed}\""

    def _collect_neighbor_evidence(
        self,
        nodes: List[str],
        keywords: List[str],
        per_node_limit: int,
        desc_limit: int
    ) -> List[tuple[int, str]]:
        if not nodes or not keywords:
            return []
        nx_graph = self.graph.graph
        collected = []
        for node in nodes:
            candidates = []
            for _, target, data in nx_graph.out_edges(node, data=True):
                relation = data.get("relation", "related")
                timestamp = data.get("timestamp")
                desc = nx_graph.nodes[target].get("description", "")
                score = self._score_neighbor_candidate(
                    keywords, node, target, relation, timestamp, desc
                )
                if score > 0:
                    line = self._format_neighbor_evidence(
                        node, target, relation, timestamp, desc, desc_limit
                    )
                    candidates.append((score, line))
            for source, _, data in nx_graph.in_edges(node, data=True):
                relation = data.get("relation", "related")
                timestamp = data.get("timestamp")
                desc = nx_graph.nodes[source].get("description", "")
                score = self._score_neighbor_candidate(
                    keywords, source, node, relation, timestamp, desc
                )
                if score > 0:
                    line = self._format_neighbor_evidence(
                        source, node, relation, timestamp, desc, desc_limit
                    )
                    candidates.append((score, line))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                collected.extend(candidates[:per_node_limit])
        return collected

    def _llm_extract_keywords(self, query: str) -> List[str]:
        """Extract keywords using LLM, with a safe fallback."""
        prompt = (
            "You are a keyword extractor. Given a question, extract 3-8 concise keywords or short phrases.\n"
            "Requirements:\n"
            "- Output JSON only in the form: {\"keywords\": [\"...\", \"...\"]}\n"
            "- Keep each keyword 1-4 words, preserve proper nouns and domain terms.\n"
            "- No explanations, no extra text.\n"
            f"Question: {query}"
        )
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = res.choices[0].message.content
            data = json.loads(content)
            keywords = data.get("keywords", [])
            if isinstance(keywords, list):
                return [str(k).strip() for k in keywords if str(k).strip()]
        except Exception:
            return []
        return []

    def answer(self, question: str) -> str:
        logger.info(f"❓ Question: {question}")
        
        history = []
        current_nodes = []
        visited_node_order = []
        visited_node_content = {}
        visited_edge_order = []
        visited_edge_set = set()
        visited_context_str = "None"
        cached_scores = {}
        max_cached_total = 8
        max_cached_per_node = 4
        cached_desc_limit = 80
        
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
                # Cache visited nodes and edges for later answering.
                nx_graph = self.graph.graph
                for name in current_nodes:
                    if name not in visited_node_content:
                        visited_node_content[name] = self.graph.primitive_read([name])
                        visited_node_order.append(name)
                    for _, target, data in nx_graph.out_edges(name, data=True):
                        rel = data.get('relation', 'related')
                        edge_ts = data.get('timestamp')
                        edge_ts_str = f" [Time: {edge_ts}]" if edge_ts else ""
                        edge_line = f"[{name}] --[{rel}{edge_ts_str}]--> [{target}]"
                        if edge_line not in visited_edge_set:
                            visited_edge_set.add(edge_line)
                            visited_edge_order.append(edge_line)
                status_str = f"Status: You are at nodes: {current_nodes}"
                view_str = f"**Current Node Content**:\n{current_content}\n\n**Visible Neighbors**:\n{neighbors_view}"

            if visited_node_order:
                nodes_view = "\n".join(visited_node_content[n] for n in visited_node_order[-5:])
            else:
                nodes_view = "None"
            if visited_edge_order:
                edges_view = "\n".join(visited_edge_order[-5:])
            else:
                edges_view = "None"
            if cached_scores:
                cached_sorted = sorted(cached_scores.items(), key=lambda x: (-x[1], x[0]))
                cached_evidence_view = "\n".join(line for line, _ in cached_sorted)
            else:
                cached_evidence_view = "None"
            visited_context_str = (
                f"**Visited Nodes**:\n{nodes_view}\n\n"
                f"**Visited Edges**:\n{edges_view}\n\n"
                f"**Cached Neighborhood Evidence**:\n{cached_evidence_view}"
            )
                
            prompt = f"""
{self.get_full_prompt()}

**User Question**: "{question}"

**Exploration History**:
{history_str}

**Visited Context**:
{visited_context_str}

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
                tool = decision.get("tool")
                logger.info(f"Step {step+1}: {tool} - {decision}")

                # --- 3. Execute Tool ---
                if tool == "SEARCH":
                    query = decision.get("query", question)
                    mode = decision.get("mode", "hybrid")
                    
                    results = []
                    query_keywords = self._prepare_query_keywords(query)
                    
                    def do_keyword():
                        return self._keyword_search(query, keywords=query_keywords)
                        
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
                        new_evidence = self._collect_neighbor_evidence(
                            current_nodes,
                            query_keywords,
                            max_cached_per_node,
                            cached_desc_limit
                        )
                        for score, line in new_evidence:
                            if line in cached_scores:
                                if score > cached_scores[line]:
                                    cached_scores[line] = score
                            else:
                                cached_scores[line] = score
                        if len(cached_scores) > max_cached_total:
                            trimmed = sorted(cached_scores.items(), key=lambda x: (-x[1], x[0]))[:max_cached_total]
                            cached_scores = dict(trimmed)
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
                    return ans
                
            except Exception as e:
                logger.error(f"Step failed: {e}")
                history.append(f"Error: {str(e)}")
        
        # Fallback: Try to answer with whatever history we have
        logger.warning("Max steps reached. Attempting to answer from history.")
        try:
            fallback_prompt = f"""You have explored the graph but reached the step limit.
Based on the exploration history below, provide the best possible answer to the question.
If you found partial information, use it to infer the answer according to your common sense and intuition.

**Question**: "{question}"

**History**:
{history_str}

**Visited Context**:
{visited_context_str}

**Current Context**:
{view_str}

Return ONLY the answer text.
"""
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": fallback_prompt}],
                temperature=0.0
            )
            return self._clean_answer(res.choices[0].message.content)
        except Exception:
            return "Unknown"
