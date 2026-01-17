import json
import logging
import re
from datetime import datetime
from typing import List, Literal, Optional

import numpy as np
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

**TEMPORAL NOTE**:
- Edges may include "(ended YYYY-MM-DD)". Use them only when the question is about past states.

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
                if k in node.lower(): score += 2
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

    def _edge_status(self, data: dict) -> Optional[str]:
        return data.get("status")

    def _edge_is_retracted(self, data: dict) -> bool:
        return self._edge_status(data) == "retracted"

    def _edge_is_ended(self, data: dict) -> bool:
        return self._edge_status(data) == "ended" or bool(data.get("ended_at"))

    def _edge_text(
        self,
        source: str,
        target: str,
        relation: str,
        timestamp: Optional[str],
        ended_at: Optional[str] = None
    ) -> str:
        rel = relation or "related"
        ts_part = f" @ {timestamp}" if timestamp else ""
        ended_part = f" (ended {ended_at})" if ended_at else ""
        return f"{source} --{rel}--> {target}{ts_part}{ended_part}"

    def _parse_iso_date(self, timestamp: Optional[str]) -> Optional[datetime]:
        if not timestamp:
            return None
        if re.match(r"^\d{4}-\d{2}-\d{2}$", timestamp):
            try:
                return datetime.strptime(timestamp, "%Y-%m-%d")
            except Exception:
                return None
        return None

    def _wants_history(self, question: str) -> bool:
        if not question:
            return False
        if re.search(r"\b(19|20)\d{2}\b", question) or re.search(r"\d{4}-\d{2}-\d{2}", question):
            return True
        q = question.lower()
        english_markers = [
            "previous", "previously", "formerly", "used to", "before",
            "in the past", "past", "earlier", "prior", "historical",
            "back then", "at the time"
        ]
        if any(m in q for m in english_markers):
            return True
        chinese_markers = ["以前", "之前", "过去", "曾经", "那时", "当时", "曾", "早期", "历史", "原来"]
        return any(m in question for m in chinese_markers)

    def _score_edge_candidate(
        self,
        keywords: List[str],
        source: str,
        target: str,
        relation: str,
        timestamp: Optional[str]
    ) -> int:
        score = 0
        src = source.lower()
        tgt = target.lower()
        rel = (relation or "").lower()
        ts = (timestamp or "").lower()
        for k in keywords:
            if k in src or k in tgt:
                score += 2
            if k in rel:
                score += 3
            if k in ts:
                score += 1
        return score

    def _dedupe_edge_candidates(self, candidates: List[dict]) -> List[dict]:
        by_pair = {}
        for cand in candidates:
            key = (cand["source"], cand["target"])
            prev = by_pair.get(key)
            if prev is None:
                by_pair[key] = cand
                continue
            prev_dt = prev.get("parsed_ts")
            curr_dt = cand.get("parsed_ts")
            if prev_dt and curr_dt:
                if curr_dt > prev_dt or (curr_dt == prev_dt and cand["score"] > prev["score"]):
                    by_pair[key] = cand
                continue
            if prev_dt or curr_dt:
                if curr_dt and not prev_dt:
                    by_pair[key] = cand
                elif not curr_dt and not prev_dt and cand["score"] > prev["score"]:
                    by_pair[key] = cand
                continue
            if cand["score"] > prev["score"]:
                by_pair[key] = cand
        return list(by_pair.values())

    def _edge_semantic_search(self, query: str, top_k: int, include_ended: bool = True) -> List[dict]:
        embedder = self.graph.embedder
        if not embedder:
            return []
        query_vec = embedder.embed(query)
        if query_vec is None:
            return []
        query_vec = np.array(query_vec)
        nx_graph = self.graph.graph
        candidates = []
        for source, target, data in nx_graph.edges(data=True):
            if self._edge_is_retracted(data):
                continue
            if not include_ended and self._edge_is_ended(data):
                continue
            relation = data.get("relation", "related")
            timestamp = data.get("timestamp")
            ended_at = data.get("ended_at")
            edge_text = self._edge_text(source, target, relation, timestamp, ended_at=ended_at)
            edge_vec = embedder.embed(edge_text)
            if edge_vec is None:
                continue
            edge_vec = np.array(edge_vec)
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(edge_vec) + 1e-9)
            score = float(np.dot(query_vec, edge_vec) / denom)
            candidates.append({
                "source": source,
                "target": target,
                "relation": relation,
                "timestamp": timestamp,
                "edge_text": edge_text,
                "ended_at": ended_at,
                "score": score,
                "parsed_ts": self._parse_iso_date(timestamp),
            })
        if not candidates:
            return []
        deduped = self._dedupe_edge_candidates(candidates)
        deduped.sort(key=lambda x: (-x["score"], x["edge_text"]))
        return deduped[:top_k]

    def _edge_keyword_search(self, keywords: List[str], top_k: int, include_ended: bool = True) -> List[dict]:
        if not keywords:
            return []
        nx_graph = self.graph.graph
        candidates = []
        for source, target, data in nx_graph.edges(data=True):
            if self._edge_is_retracted(data):
                continue
            if not include_ended and self._edge_is_ended(data):
                continue
            relation = data.get("relation", "related")
            timestamp = data.get("timestamp")
            ended_at = data.get("ended_at")
            score = self._score_edge_candidate(keywords, source, target, relation, timestamp)
            if score <= 0:
                continue
            edge_text = self._edge_text(source, target, relation, timestamp, ended_at=ended_at)
            candidates.append({
                "source": source,
                "target": target,
                "relation": relation,
                "timestamp": timestamp,
                "ended_at": ended_at,
                "edge_text": edge_text,
                "score": float(score),
                "parsed_ts": self._parse_iso_date(timestamp),
            })
        if not candidates:
            return []
        deduped = self._dedupe_edge_candidates(candidates)
        deduped.sort(key=lambda x: (-x["score"], x["edge_text"]))
        return deduped[:top_k]

    def _collect_neighbor_evidence(
        self,
        nodes: List[str],
        keywords: List[str],
        per_node_limit: int,
        desc_limit: int,
        include_ended: bool = True
    ) -> List[tuple[int, str]]:
        if not nodes or not keywords:
            return []
        nx_graph = self.graph.graph
        collected = []
        for node in nodes:
            candidates = []
            for _, target, data in nx_graph.out_edges(node, data=True):
                if self._edge_is_retracted(data):
                    continue
                if not include_ended and self._edge_is_ended(data):
                    continue
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
                if self._edge_is_retracted(data):
                    continue
                if not include_ended and self._edge_is_ended(data):
                    continue
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
        wants_history = self._wants_history(question)
        
        def edge_is_visible(data: dict) -> bool:
            if self._edge_is_retracted(data):
                return False
            if not wants_history and self._edge_is_ended(data):
                return False
            return True
        
        history = []
        visited_node_order = []
        visited_node_content = {}
        visited_edge_order = []
        visited_edge_set = set()
        visited_context_str = "None"
        cached_scores = {}
        max_cached_total = 8
        max_cached_per_node = 4
        cached_desc_limit = 80
        edge_top_k = 3
        max_current_nodes = 6
        current_nodes = []
        max_rounds = 4
        nx_graph = self.graph.graph
        scored_node_cache = {}
        query_keywords = self._prepare_query_keywords(question)
        edge_evidence_lines = []

        def cache_nodes_and_edges(nodes: List[str]) -> None:
            for name in nodes:
                if name not in visited_node_content:
                    visited_node_content[name] = self.graph.primitive_read([name], include_ended=wants_history)
                    visited_node_order.append(name)
                for _, target, data in nx_graph.out_edges(name, data=True):
                    if not edge_is_visible(data):
                        continue
                    rel = data.get("relation", "related")
                    edge_ts = data.get("timestamp")
                    edge_ts_str = f" [Time: {edge_ts}]" if edge_ts else ""
                    ended_at = data.get("ended_at")
                    ended_str = f" [Ended: {ended_at}]" if ended_at else ""
                    edge_line = f"[{name}] --[{rel}{edge_ts_str}{ended_str}]--> [{target}]"
                    if edge_line not in visited_edge_set:
                        visited_edge_set.add(edge_line)
                        visited_edge_order.append(edge_line)
                for source, _, data in nx_graph.in_edges(name, data=True):
                    if not edge_is_visible(data):
                        continue
                    rel = data.get("relation", "related")
                    edge_ts = data.get("timestamp")
                    edge_ts_str = f" [Time: {edge_ts}]" if edge_ts else ""
                    ended_at = data.get("ended_at")
                    ended_str = f" [Ended: {ended_at}]" if ended_at else ""
                    edge_line = f"[{source}] --[{rel}{edge_ts_str}{ended_str}]--> [{name}]"
                    if edge_line not in visited_edge_set:
                        visited_edge_set.add(edge_line)
                        visited_edge_order.append(edge_line)

        def update_cached_evidence(nodes: List[str]) -> None:
            new_evidence = self._collect_neighbor_evidence(
                nodes,
                query_keywords,
                max_cached_per_node,
                cached_desc_limit,
                include_ended=wants_history
            )
            for score, line in new_evidence:
                if line in cached_scores:
                    if score > cached_scores[line]:
                        cached_scores[line] = score
                else:
                    cached_scores[line] = score
            if len(cached_scores) > max_cached_total:
                trimmed = sorted(cached_scores.items(), key=lambda x: (-x[1], x[0]))[:max_cached_total]
                cached_scores.clear()
                cached_scores.update(trimmed)

        def get_cached_score(
            node: str,
            source: str,
            target: str,
            relation: str,
            timestamp: str,
            desc: str
        ) -> int:
            if node in scored_node_cache:
                return scored_node_cache[node]
            score = self._score_neighbor_candidate(
                query_keywords, source, target, relation, timestamp, desc
            )
            scored_node_cache[node] = score
            return score

        def compute_candidates(frontier: str, path: List[str]) -> List[tuple[int, str]]:
            candidates = {}
            for _, target, data in nx_graph.out_edges(frontier, data=True):
                if not edge_is_visible(data):
                    continue
                if target in path:
                    continue
                relation = data.get("relation", "related")
                timestamp = data.get("timestamp")
                desc = nx_graph.nodes[target].get("description", "")
                score = get_cached_score(target, frontier, target, relation, timestamp, desc)
                if target not in candidates or score > candidates[target]:
                    candidates[target] = score
            for source, _, data in nx_graph.in_edges(frontier, data=True):
                if not edge_is_visible(data):
                    continue
                if source in path:
                    continue
                relation = data.get("relation", "related")
                timestamp = data.get("timestamp")
                desc = nx_graph.nodes[source].get("description", "")
                score = get_cached_score(source, source, frontier, relation, timestamp, desc)
                if source not in candidates or score > candidates[source]:
                    candidates[source] = score
            return sorted(
                [(score, node) for node, score in candidates.items()],
                key=lambda x: (-x[0], x[1])
            )

        def build_prompt(nodes: List[str]) -> tuple[str, str, str, str]:
            history_str = "\n".join(history[-5:]) if history else "None"

            if not nodes:
                status_str = "Status: You are currently NOT at any node. You need to SEARCH to find entry points."
                view_str = ""
            else:
                neighbors_view = self.graph.primitive_get_neighbors(nodes, include_ended=wants_history)
                current_content = self.graph.primitive_read(nodes, include_ended=wants_history)
                cache_nodes_and_edges(nodes)
                status_str = f"Status: You are at nodes: {nodes}"
                view_str = f"**Current Node Content**:\n{current_content}\n\n**Visible Neighbors**:\n{neighbors_view}"

            if visited_node_order:
                nodes_view = "\n".join(visited_node_content[n] for n in visited_node_order)
            else:
                nodes_view = "None"
            if visited_edge_order:
                edges_view = "\n".join(visited_edge_order[-10:])
            else:
                edges_view = "None"
            if edge_evidence_lines:
                edge_evidence_view = "\n".join(edge_evidence_lines)
            else:
                edge_evidence_view = "None"
            if cached_scores:
                cached_sorted = sorted(cached_scores.items(), key=lambda x: (-x[1], x[0]))
                cached_evidence_view = "\n".join(line for line, _ in cached_sorted)
            else:
                cached_evidence_view = "None"
            visited_str = (
                f"**Visited Nodes**:\n{nodes_view}\n\n"
                f"**Visited Edges**:\n{edges_view}\n\n"
                f"**Edge Evidence**:\n{edge_evidence_view}\n\n"
                f"**Cached Neighborhood Evidence**:\n{cached_evidence_view}"
            )

            prompt = f"""
{self.get_full_prompt()}

**User Question**: "{question}"

**Exploration History**:
{history_str}

**Visited Context**:
{visited_str}

**{status_str}**
{view_str}
"""
            return prompt, view_str, visited_str, history_str

        # Step 1: SEARCH (hybrid)
        def do_keyword():
            return self._keyword_search(question, keywords=query_keywords)

        def do_semantic():
            if hasattr(self.graph, "semantic_search"):
                return self.graph.semantic_search(question)
            return []

        edge_results = self._edge_semantic_search(question, edge_top_k, include_ended=wants_history)
        edge_search_mode = "semantic"
        if not edge_results:
            edge_results = self._edge_keyword_search(query_keywords, edge_top_k, include_ended=wants_history)
            edge_search_mode = "keyword"

        if edge_results:
            edge_evidence_lines = [e["edge_text"] for e in edge_results]
            edge_brief = ", ".join(f"{e['source']}->{e['target']}" for e in edge_results)
            history.append(f"EDGE_SEARCH({edge_search_mode}, '{question}') -> Found: {edge_brief}")
        else:
            history.append(f"EDGE_SEARCH({edge_search_mode}, '{question}') -> Found nothing.")

        edge_seed_nodes = []
        edge_seed_set = set()
        for e in edge_results:
            for node in (e["source"], e["target"]):
                if node not in edge_seed_set:
                    edge_seed_set.add(node)
                    edge_seed_nodes.append(node)

        k_res = do_keyword()
        s_res = do_semantic()
        seen = set(k_res)
        node_results = k_res + [x for x in s_res if x not in seen]

        combined_nodes = edge_seed_nodes + [n for n in node_results if n not in edge_seed_set]
        if combined_nodes:
            current_nodes = combined_nodes[:max_current_nodes]
            history.append(f"SEARCH(hybrid, '{question}') -> Found: {current_nodes}")
            cache_nodes_and_edges(current_nodes)
            update_cached_evidence(current_nodes)
        else:
            history.append(f"SEARCH(hybrid, '{question}') -> Found nothing.")
            current_nodes = []

        beams = []
        for node in current_nodes:
            beam = {"path": [node], "frontier": node, "candidates": []}
            beam["candidates"] = compute_candidates(node, beam["path"])
            beams.append(beam)

        last_view_str = ""
        last_history_str = "None"
        last_visited_context_str = "None"

        for round_idx in range(max_rounds):
            prompt, last_view_str, last_visited_context_str, last_history_str = build_prompt(current_nodes)
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

                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                decision = json.loads(content.strip())
                tool = decision.get("tool")
                logger.info(f"Step {round_idx+2}: {tool} - {decision}")

                if tool == "READ":
                    ans = self._clean_answer(decision.get("answer"))
                    return ans
            except Exception as e:
                logger.error(f"Step failed: {e}")
                history.append(f"Error: {str(e)}")

            if not beams:
                break

            beam_rankings = []
            for idx, beam in enumerate(beams):
                if not beam["candidates"]:
                    continue
                best_score, _ = beam["candidates"][0]
                beam_rankings.append((best_score, idx))
            beam_rankings.sort(key=lambda x: (-x[0], x[1]))

            selected = []
            used_nodes = set()
            for _, idx in beam_rankings:
                if len(selected) >= 2:
                    break
                candidate_list = beams[idx]["candidates"]
                chosen_node = None
                chosen_score = None
                for score, node in candidate_list:
                    if node in used_nodes:
                        continue
                    chosen_node = node
                    chosen_score = score
                    break
                if chosen_node is None:
                    continue
                selected.append((idx, chosen_node, chosen_score))
                used_nodes.add(chosen_node)

            if not selected:
                history.append("WALK (beam) -> No candidates.")
                break

            walked_nodes = []
            for idx, node, score in selected:
                beam = beams[idx]
                prev = beam["frontier"]
                beam["path"].append(node)
                beam["frontier"] = node
                beam["candidates"] = compute_candidates(node, beam["path"])
                history.append(f"WALK (beam) -> Moved to {node} (from {prev}, score={score})")
                walked_nodes.append(node)

            update_cached_evidence(walked_nodes)
            current_nodes = walked_nodes

        # Fallback: Try to answer with whatever history we have
        logger.warning("Max steps reached. Attempting to answer from history.")
        try:
            fallback_prompt = f"""You have explored the graph but reached the step limit.
Based on the exploration history below, provide the best possible answer to the question.
If you found partial information, use it to infer the answer according to your common sense and intuition.

**Question**: "{question}"

**History**:
{last_history_str}

**Visited Context**:
{last_visited_context_str}

**Current Context**:
{last_view_str}

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
