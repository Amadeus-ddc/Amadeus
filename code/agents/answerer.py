import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from amadeus_collab.core.graph import MemoryGraph
from amadeus_collab.agents.base import BaseAgent

logger = logging.getLogger("Amadeus.Answerer")


class AnswererAgent(BaseAgent):
    def __init__(self, graph: MemoryGraph, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
        super().__init__(model_name, api_base, api_key)
        self.graph = graph
        self.max_steps = 8

        self.walk_candidate_top_k: int = 10
        self.max_walk_steps_per_round: int = 5
        self.high_degree_threshold: int = 50
        self.node_revisit_limit: int = 2
        self.recent_edges_max_items: int = 6

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
   - `entry_edge_ids`: Optional. Choose any relevant edge IDs from the provided candidate entry edges when you are selecting the best starting evidence.
   - `entry_nodes`: Optional. Choose any relevant node names only from the provided candidate entry edges/nodes.
   - *Condition*: Use this if you are nowhere, lost, or need to find a specific entity. At the first step, if candidate entry edges are shown, use SEARCH to select the best entry evidence.

2. **WALK**: Move to connected nodes.
   - `steps`: 1-5 steps, each in the form `{"from": "CurrentNode", "to": "TargetNode"}`.
   - Choose only from `Candidate Walk Options`.
   - Prefer unvisited nodes. Avoid loops.
   - If a frontier is marked high-degree, rely on edge relation, timestamp, and the evidence pool instead of missing descriptions.

3. **READ**: Finish and answer.
   - `answer`: The final concise answer.
   - Use READ when you can already give the best supported answer from the evidence pool and current context.
   - Do not keep walking for small improvements.

**RESPONSE FORMAT (JSON ONLY)**:
{
  "tool": "SEARCH",
  "query": "...",
  "mode": "hybrid",
  "entry_edge_ids": ["E1"],
  "entry_nodes": ["TargetNodeName"]
}
OR
{
  "tool": "WALK",
  "steps": [
    {"from": "CurrentNodeA", "to": "TargetNodeA"},
    {"from": "CurrentNodeB", "to": "TargetNodeB"}
  ]
}
OR
{
  "tool": "READ",
  "answer": "Final Answer Here"
}
"""

    def _clean_answer(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()

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

        lower_text = text.lower()
        if lower_text.startswith("yes") or lower_text.startswith("no"):
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
        hits = []
        if keywords is None:
            keywords = self._prepare_query_keywords(query)

        nx_graph = self.graph.graph
        for node, data in nx_graph.nodes(data=True):
            content = f"{node} {data.get('description', '')}".lower()
            score = 0
            for k in keywords:
                if k in node.lower():
                    score += 2
                elif k in content:
                    score += 1
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
        desc: str,
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
        desc_limit: int,
    ) -> str:
        rel = relation or "related"
        ts_part = f" @ {timestamp}" if timestamp else ""
        trimmed = (desc or "").strip()
        if len(trimmed) > desc_limit:
            trimmed = trimmed[:desc_limit].rstrip() + "..."
        return f'[{source}] --[{rel}{ts_part}]--> [{target}] | desc: "{trimmed}"'

    def _edge_text(self, source: str, target: str, relation: str, timestamp: Optional[str]) -> str:
        rel = relation or "related"
        ts_part = f" @ {timestamp}" if timestamp else ""
        return f"{source} --{rel}--> {target}{ts_part}"

    def _parse_iso_date(self, timestamp: Optional[str]) -> Optional[datetime]:
        if not timestamp:
            return None
        if re.match(r"^\d{4}-\d{2}-\d{2}$", timestamp):
            try:
                return datetime.strptime(timestamp, "%Y-%m-%d")
            except Exception:
                return None
        return None

    def _score_edge_candidate(
        self,
        keywords: List[str],
        source: str,
        target: str,
        relation: str,
        timestamp: Optional[str],
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

    def _edge_semantic_search(self, query: str, top_k: int) -> List[dict]:
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
            relation = data.get("relation", "related")
            timestamp = data.get("timestamp")
            edge_text = self._edge_text(source, target, relation, timestamp)
            edge_vec = embedder.embed(edge_text)
            if edge_vec is None:
                continue
            edge_vec = np.array(edge_vec)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(edge_vec) + 1e-9
            score = float(np.dot(query_vec, edge_vec) / denom)
            candidates.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "timestamp": timestamp,
                    "edge_text": edge_text,
                    "score": score,
                    "parsed_ts": self._parse_iso_date(timestamp),
                }
            )
        if not candidates:
            return []
        deduped = self._dedupe_edge_candidates(candidates)
        deduped.sort(key=lambda x: (-x["score"], x["edge_text"]))
        return deduped[:top_k]

    def _edge_keyword_search(self, keywords: List[str], top_k: int) -> List[dict]:
        if not keywords:
            return []
        nx_graph = self.graph.graph
        candidates = []
        for source, target, data in nx_graph.edges(data=True):
            relation = data.get("relation", "related")
            timestamp = data.get("timestamp")
            score = self._score_edge_candidate(keywords, source, target, relation, timestamp)
            if score <= 0:
                continue
            edge_text = self._edge_text(source, target, relation, timestamp)
            candidates.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "timestamp": timestamp,
                    "edge_text": edge_text,
                    "score": float(score),
                    "parsed_ts": self._parse_iso_date(timestamp),
                }
            )
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
    ) -> List[Tuple[int, str]]:
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
                score = self._score_neighbor_candidate(keywords, node, target, relation, timestamp, desc)
                if score > 0:
                    line = self._format_neighbor_evidence(node, target, relation, timestamp, desc, desc_limit)
                    candidates.append((score, line))
            for source, _, data in nx_graph.in_edges(node, data=True):
                relation = data.get("relation", "related")
                timestamp = data.get("timestamp")
                desc = nx_graph.nodes[source].get("description", "")
                score = self._score_neighbor_candidate(keywords, source, node, relation, timestamp, desc)
                if score > 0:
                    line = self._format_neighbor_evidence(source, node, relation, timestamp, desc, desc_limit)
                    candidates.append((score, line))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                collected.extend(candidates[:per_node_limit])
        return collected

    def _llm_extract_keywords(self, query: str) -> List[str]:
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
                temperature=0.0,
            )
            self._record_usage(res)
            content = res.choices[0].message.content
            data = json.loads(content)
            keywords = data.get("keywords", [])
            if isinstance(keywords, list):
                return [str(k).strip() for k in keywords if str(k).strip()]
        except Exception:
            return []
        return []

    def _parse_json_response(self, content: str) -> dict:
        if not content:
            return {}
        cleaned = content.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        return json.loads(cleaned.strip())

    def _normalize_fact(self, fact: str) -> str:
        normalized = re.sub(r"\s+", " ", (fact or "").strip())
        return normalized.rstrip(".").lower()

    def _merge_evidence_pool(self, evidence_pool: List[dict], new_facts: List[str], keywords: List[str]) -> List[dict]:
        by_fact = {}
        ordered_facts = []

        for item in evidence_pool:
            fact = re.sub(r"\s+", " ", str(item.get("fact", "")).strip())
            if not fact:
                continue
            norm = self._normalize_fact(fact)
            if not norm or norm in by_fact:
                continue
            candidate = {
                "fact": fact,
                "source": item.get("source", "llm_extractor"),
            }
            by_fact[norm] = candidate
            ordered_facts.append(candidate)

        for fact in new_facts:
            text = re.sub(r"\s+", " ", str(fact).strip())
            if not text:
                continue
            norm = self._normalize_fact(text)
            if not norm or norm in by_fact:
                continue
            candidate = {
                "fact": text,
                "source": "llm_extractor",
            }
            by_fact[norm] = candidate
            ordered_facts.append(candidate)

        return ordered_facts

    def _render_evidence_pool(self, evidence_pool: List[dict]) -> str:
        if not evidence_pool:
            return "None"
        return "\n".join(f"- {item['fact']}" for item in evidence_pool)

    def _extract_step_evidence(
        self,
        question: str,
        node_contents: List[str],
        traversed_edges: List[str],
        current_evidence: str,
    ) -> List[str]:
        if not node_contents and not traversed_edges:
            return []

        node_block = "\n\n".join(c for c in node_contents if c) or "None"
        edge_block = "\n".join(traversed_edges) if traversed_edges else "None"
        prompt = f"""Extract only question-relevant facts.
One fact per list item.
Preserve exact absolute dates, times, counts, names, locations, and negations.
Do not add explanations.
Return JSON only: {{"facts": ["..."]}}

Question: {question}

Current evidence pool:
{current_evidence}

New node content:
{node_block}

New traversed edges:
{edge_block}
"""
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            self._record_usage(res)
            data = self._parse_json_response(res.choices[0].message.content)
            facts = data.get("facts", [])
            if isinstance(facts, list):
                return [str(f).strip() for f in facts if str(f).strip()]
        except Exception as e:
            logger.warning(f"Evidence extraction failed: {e}")
        return []

    def _render_candidate_options(self, candidate_bundles: List[dict]) -> str:
        if not candidate_bundles:
            return "None"

        lines = []
        for bundle in candidate_bundles:
            header = f"From [{bundle['frontier']}] (neighbors={bundle['degree']})"
            if bundle.get("high_degree"):
                header += " [high-degree: edge-only view]"
            lines.append(header)
            options = bundle.get("options", [])
            if not options:
                lines.append("  - No available candidates")
                continue
            for idx, option in enumerate(options, 1):
                line = f"  {idx}. to [{option['node']}] via {option['edge_text']}"
                if not bundle.get("high_degree") and option.get("description"):
                    line += f" | desc: {option['description']}"
                lines.append(line)
        return "\n".join(lines)

    def _validate_walk_steps(
        self,
        steps: List[dict],
        candidate_lookup: Dict[str, Dict[str, dict]],
        node_visit_counts: Dict[str, int],
    ) -> Tuple[List[dict], Optional[str]]:
        if not isinstance(steps, list):
            return [], "steps is not a list"

        valid_steps = []
        reasons = []
        used_from = set()
        used_to = set()

        for raw_step in steps[:self.max_walk_steps_per_round]:
            if not isinstance(raw_step, dict):
                reasons.append("step is not an object")
                continue

            from_node = str(raw_step.get("from", "")).strip()
            to_node = str(raw_step.get("to", "")).strip()
            if not from_node or not to_node:
                reasons.append("missing from/to")
                continue
            if from_node in used_from:
                reasons.append(f"duplicate from {from_node}")
                continue
            if to_node in used_to:
                reasons.append(f"duplicate target {to_node}")
                continue

            frontier_candidates = candidate_lookup.get(from_node)
            if not frontier_candidates or to_node not in frontier_candidates:
                reasons.append(f"invalid step {from_node}->{to_node}")
                continue
            if node_visit_counts.get(to_node, 0) >= self.node_revisit_limit:
                reasons.append(f"visit limit reached for {to_node}")
                continue

            valid_steps.append(
                {
                    "from": from_node,
                    "to": to_node,
                    "candidate": frontier_candidates[to_node],
                }
            )
            used_from.add(from_node)
            used_to.add(to_node)

        reason = "; ".join(reasons) if reasons else None
        return valid_steps, reason

    def _fallback_walk_selection(self, beams: List[dict]) -> List[dict]:
        beam_rankings = []
        for idx, beam in enumerate(beams):
            candidates = beam.get("candidates", [])
            if not candidates:
                continue
            beam_rankings.append((candidates[0]["score"], idx))
        beam_rankings.sort(key=lambda x: (-x[0], x[1]))

        selected = []
        used_from = set()
        used_to = set()
        for _, idx in beam_rankings:
            if len(selected) >= self.max_walk_steps_per_round:
                break
            beam = beams[idx]
            frontier = beam["frontier"]
            if frontier in used_from:
                continue
            chosen = None
            for candidate in beam.get("candidates", []):
                if candidate["node"] in used_to:
                    continue
                chosen = candidate
                break
            if chosen is None:
                continue
            selected.append(
                {
                    "from": frontier,
                    "to": chosen["node"],
                    "candidate": chosen,
                }
            )
            used_from.add(frontier)
            used_to.add(chosen["node"])
        return selected

    def answer(self, question: str) -> str:
        logger.info(f"❓ Question: {question}")

        history = []
        visited_node_order = []
        visited_node_content = {}
        visited_edge_order = []
        visited_edge_set = set()
        cached_scores = {}

        max_cached_total = 8
        max_cached_per_node = 4
        cached_desc_limit = 80
        edge_top_k = 8
        max_current_nodes = 16
        max_rounds = 4

        current_nodes: List[str] = []
        nx_graph = self.graph.graph
        query_keywords = self._prepare_query_keywords(question)
        edge_evidence_lines: List[str] = []
        embedder = self.graph.embedder
        query_vec = None
        query_norm = None
        edge_embed_cache = {}
        edge_score_cache = {}

        evidence_pool: List[dict] = []
        node_visit_counts: Dict[str, int] = {}
        recent_traversed_edges: List[str] = []

        if embedder:
            query_vec = embedder.embed(question)
            if query_vec is not None:
                query_vec = np.array(query_vec)
                query_norm = np.linalg.norm(query_vec) + 1e-9

        def cache_nodes_and_edges(nodes: List[str]) -> None:
            for name in nodes:
                if name not in visited_node_content:
                    visited_node_content[name] = self.graph.primitive_read([name])
                    visited_node_order.append(name)
                for _, target, data in nx_graph.out_edges(name, data=True):
                    rel = data.get("relation", "related")
                    edge_ts = data.get("timestamp")
                    edge_ts_str = f" [Time: {edge_ts}]" if edge_ts else ""
                    edge_line = f"[{name}] --[{rel}{edge_ts_str}]--> [{target}]"
                    if edge_line not in visited_edge_set:
                        visited_edge_set.add(edge_line)
                        visited_edge_order.append(edge_line)
                for source, _, data in nx_graph.in_edges(name, data=True):
                    rel = data.get("relation", "related")
                    edge_ts = data.get("timestamp")
                    edge_ts_str = f" [Time: {edge_ts}]" if edge_ts else ""
                    edge_line = f"[{source}] --[{rel}{edge_ts_str}]--> [{name}]"
                    if edge_line not in visited_edge_set:
                        visited_edge_set.add(edge_line)
                        visited_edge_order.append(edge_line)

        def update_cached_evidence(nodes: List[str]) -> None:
            new_evidence = self._collect_neighbor_evidence(
                nodes,
                query_keywords,
                max_cached_per_node,
                cached_desc_limit,
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

        def get_edge_similarity(edge_text: str) -> Optional[float]:
            if query_vec is None or embedder is None or query_norm is None:
                return None
            if edge_text in edge_score_cache:
                return edge_score_cache[edge_text]
            edge_vec = edge_embed_cache.get(edge_text)
            if edge_vec is None:
                edge_vec = embedder.embed(edge_text)
                if edge_vec is None:
                    edge_score_cache[edge_text] = None
                    return None
                edge_vec = np.array(edge_vec)
                edge_embed_cache[edge_text] = edge_vec
            denom = query_norm * (np.linalg.norm(edge_vec) + 1e-9)
            score = float(np.dot(query_vec, edge_vec) / denom)
            edge_score_cache[edge_text] = score
            return score

        def trim_text(text: str, limit: int = 100) -> str:
            text = (text or "").strip()
            if len(text) > limit:
                return text[:limit].rstrip() + "..."
            return text

        def build_candidate_blocks(edge_candidates: List[dict], node_candidates: List[str]):
            edge_lookup = {}
            allowed_nodes = set(node_candidates)
            edge_lines = []
            for idx, edge in enumerate(edge_candidates, start=1):
                edge_id = f"E{idx}"
                edge_lookup[edge_id] = edge
                allowed_nodes.add(edge["source"])
                allowed_nodes.add(edge["target"])
                source_desc = trim_text(nx_graph.nodes[edge["source"]].get("description", "")) if nx_graph.has_node(edge["source"]) else ""
                target_desc = trim_text(nx_graph.nodes[edge["target"]].get("description", "")) if nx_graph.has_node(edge["target"]) else ""
                ts = edge.get("timestamp") or "None"
                edge_lines.append(
                    f"[{edge_id}]\n"
                    f"source: {edge['source']}\n"
                    f"relation: {edge.get('relation', 'related')}\n"
                    f"target: {edge['target']}\n"
                    f"timestamp: {ts}\n"
                    f"retrieval: score={edge.get('score', 0.0):.3f}\n"
                    f"source_desc: {source_desc}\n"
                    f"target_desc: {target_desc}"
                )
            node_lines = []
            for idx, node in enumerate(node_candidates, start=1):
                desc = trim_text(nx_graph.nodes[node].get("description", "")) if nx_graph.has_node(node) else ""
                node_lines.append(f"[N{idx}] {node} — {desc}")
            edge_block = "\n\n".join(edge_lines) if edge_lines else "None"
            node_block = "\n".join(node_lines) if node_lines else "None"
            return edge_block, node_block, edge_lookup, allowed_nodes

        def run_search(search_query: str, mode: str = "hybrid") -> Tuple[List[str], List[str], str]:
            search_mode = (mode or "hybrid").lower()
            search_query = (search_query or question).strip() or question
            search_keywords = self._prepare_query_keywords(search_query)

            local_edge_results = []
            if search_mode in ("hybrid", "semantic"):
                local_edge_results.extend(self._edge_semantic_search(search_query, edge_top_k))
            if search_mode in ("hybrid", "keyword"):
                kw_edge_hits = self._edge_keyword_search(search_keywords, edge_top_k)
                seen_edges = {
                    (e["source"], e["target"], e.get("relation"), e.get("timestamp"))
                    for e in local_edge_results
                }
                for edge in kw_edge_hits:
                    key = (edge["source"], edge["target"], edge.get("relation"), edge.get("timestamp"))
                    if key not in seen_edges:
                        seen_edges.add(key)
                        local_edge_results.append(edge)

            local_edge_lines = [e["edge_text"] for e in local_edge_results]
            edge_seed_nodes = []
            edge_seed_set = set()
            for edge in local_edge_results:
                for node in (edge["source"], edge["target"]):
                    if node not in edge_seed_set:
                        edge_seed_set.add(node)
                        edge_seed_nodes.append(node)

            keyword_nodes = []
            if search_mode in ("hybrid", "keyword"):
                keyword_nodes = self._keyword_search(search_query, keywords=search_keywords)
            semantic_nodes = []
            if search_mode in ("hybrid", "semantic") and hasattr(self.graph, "semantic_search"):
                semantic_nodes = self.graph.semantic_search(search_query)

            if search_mode == "keyword":
                node_results = keyword_nodes
            elif search_mode == "semantic":
                node_results = semantic_nodes
            else:
                seen_nodes = set(keyword_nodes)
                node_results = keyword_nodes + [x for x in semantic_nodes if x not in seen_nodes]

            combined = edge_seed_nodes + [n for n in node_results if n not in edge_seed_set]
            combined = combined[:max_current_nodes]
            note = f"SEARCH({search_mode}, '{search_query}')"
            return combined, local_edge_lines, note

        def run_rule_search(search_query: str, mode: str) -> List[str]:
            return run_search(search_query, mode)[0]

        def dedupe_beams(beams: List[dict]) -> List[dict]:
            by_frontier = {}
            for beam in beams:
                frontier = beam["frontier"]
                prev = by_frontier.get(frontier)
                if prev is None or len(beam["path"]) < len(prev["path"]):
                    by_frontier[frontier] = beam
            deduped = list(by_frontier.values())
            deduped.sort(key=lambda b: (len(b["path"]), b["frontier"]))
            return deduped

        def build_candidate_bundle(frontier: str, path: List[str]) -> dict:
            total_degree = nx_graph.out_degree(frontier) + nx_graph.in_degree(frontier)
            high_degree = total_degree > self.high_degree_threshold
            raw_candidates: Dict[str, dict] = {}

            def maybe_add(node: str, relation: str, timestamp: Optional[str], edge_text: str) -> None:
                if node in path:
                    return
                visit_count = node_visit_counts.get(node, 0)
                if visit_count >= self.node_revisit_limit:
                    return
                score = get_edge_similarity(edge_text)
                if score is None:
                    score = float(self._score_edge_candidate(query_keywords, frontier, node, relation, timestamp))
                desc = nx_graph.nodes[node].get("description", "")
                preview = desc[:80].rstrip() + ("..." if len(desc) > 80 else "") if desc else ""
                candidate = {
                    "node": node,
                    "score": float(score),
                    "relation": relation,
                    "timestamp": timestamp,
                    "edge_text": edge_text,
                    "description": preview,
                }
                prev = raw_candidates.get(node)
                if prev is None or candidate["score"] > prev["score"]:
                    raw_candidates[node] = candidate

            for _, target, data in nx_graph.out_edges(frontier, data=True):
                relation = data.get("relation", "related")
                timestamp = data.get("timestamp")
                edge_text = self._edge_text(frontier, target, relation, timestamp)
                maybe_add(target, relation, timestamp, edge_text)
            for source, _, data in nx_graph.in_edges(frontier, data=True):
                relation = data.get("relation", "related")
                timestamp = data.get("timestamp")
                edge_text = self._edge_text(source, frontier, relation, timestamp)
                maybe_add(source, relation, timestamp, edge_text)

            ranked = sorted(raw_candidates.values(), key=lambda x: (-x["score"], x["node"]))
            fresh = [c for c in ranked if node_visit_counts.get(c["node"], 0) == 0]
            revisits = [c for c in ranked if 0 < node_visit_counts.get(c["node"], 0) < self.node_revisit_limit]
            options = fresh[:self.walk_candidate_top_k]
            if len(options) < self.walk_candidate_top_k:
                options.extend(revisits[: self.walk_candidate_top_k - len(options)])

            return {
                "frontier": frontier,
                "degree": total_degree,
                "high_degree": high_degree,
                "options": options,
            }

        def refresh_beam_candidates(beams: List[dict]) -> List[dict]:
            bundles = []
            for beam in beams:
                bundle = build_candidate_bundle(beam["frontier"], beam["path"])
                beam["candidates"] = bundle["options"]
                beam["degree"] = bundle["degree"]
                beam["high_degree"] = bundle["high_degree"]
                bundles.append(bundle)
            bundles.sort(
                key=lambda b: (
                    -(b["options"][0]["score"] if b["options"] else -1.0),
                    b["frontier"],
                )
            )
            return bundles

        def rebuild_beams(nodes: List[str]) -> List[dict]:
            rebuilt = []
            seen = set()
            for node in nodes:
                if not node or node in seen:
                    continue
                seen.add(node)
                rebuilt.append({"path": [node], "frontier": node, "candidates": []})
            rebuilt = dedupe_beams(rebuilt)
            refresh_beam_candidates(rebuilt)
            return rebuilt

        def build_prompt(nodes: List[str], candidate_bundles: List[dict]) -> Tuple[str, str, str, str]:
            history_str = "\n".join(history[-5:]) if history else "None"
            evidence_pool_str = self._render_evidence_pool(evidence_pool)
            cached_evidence_view = "\n".join(
                line for line, _ in sorted(cached_scores.items(), key=lambda x: (-x[1], x[0]))
            ) if cached_scores else "None"

            if not nodes:
                status_str = "Status: You are currently NOT at any node. You need to SEARCH to find entry points."
                current_content = "None"
                entry_view = (
                    f"**Candidate Entry Edges**:\n{candidate_edge_block}\n\n"
                    f"**Candidate Entry Nodes**:\n{candidate_node_block}"
                )
                candidate_view = "None"
            else:
                status_str = f"Status: You are at nodes: {nodes}"
                current_content = self.graph.primitive_read(nodes)
                cache_nodes_and_edges(nodes)
                entry_view = "None"
                candidate_view = self._render_candidate_options(candidate_bundles)

            recent_nodes = ", ".join(visited_node_order[-6:]) if visited_node_order else "None"
            recent_edges_view = "\n".join(recent_traversed_edges[-self.recent_edges_max_items:]) if recent_traversed_edges else "None"
            visited_summary = (
                f"Recent visited nodes: {recent_nodes}\n"
                f"Recent traversed edges:\n{recent_edges_view}"
            )

            prompt = f"""
{self.get_full_prompt()}

**User Question**: "{question}"

**Exploration History**:
{history_str}

**Evidence Pool**:
{evidence_pool_str}

**{status_str}**

**Current Node Content**:
{current_content}

**Candidate Entry Evidence**:
{entry_view}

**Candidate Walk Options**:
{candidate_view}

**Cached Neighborhood Evidence**:
{cached_evidence_view}

**Visited Summary**:
{visited_summary}
"""
            return prompt, current_content, visited_summary, history_str

        def apply_search_results(nodes: List[str], new_edge_lines: List[str], note: str) -> List[dict]:
            nonlocal evidence_pool, edge_evidence_lines, recent_traversed_edges

            edge_evidence_lines = new_edge_lines
            recent_traversed_edges = (recent_traversed_edges + new_edge_lines)[-self.recent_edges_max_items:]
            history.append(f"{note} -> Found: {nodes}" if nodes else f"{note} -> Found nothing.")
            if not nodes:
                return []

            cache_nodes_and_edges(nodes)
            update_cached_evidence(nodes)
            for node in nodes:
                node_visit_counts[node] = max(1, node_visit_counts.get(node, 0))

            node_contents = [visited_node_content[n] for n in nodes if n in visited_node_content]
            new_facts = self._extract_step_evidence(
                question,
                node_contents,
                new_edge_lines,
                self._render_evidence_pool(evidence_pool),
            )
            evidence_pool = self._merge_evidence_pool(evidence_pool, new_facts, query_keywords)
            logger.info(f"evidence_pool_init={self._render_evidence_pool(evidence_pool)}")
            return rebuild_beams(nodes)

        edge_results = self._edge_semantic_search(question, edge_top_k)
        edge_search_mode = "semantic"
        if not edge_results:
            edge_results = self._edge_keyword_search(query_keywords, edge_top_k)
            edge_search_mode = "keyword"

        if edge_results:
            edge_evidence_lines = [e["edge_text"] for e in edge_results]
            edge_brief = ", ".join(f"{e['source']}->{e['target']}" for e in edge_results)
            history.append(f"EDGE_SEARCH({edge_search_mode}, '{question}') -> Found: {edge_brief}")
        else:
            history.append(f"EDGE_SEARCH({edge_search_mode}, '{question}') -> Found nothing.")

        edge_seed_nodes = []
        edge_seed_set = set()
        for edge in edge_results:
            for node in (edge["source"], edge["target"]):
                if node not in edge_seed_set:
                    edge_seed_set.add(node)
                    edge_seed_nodes.append(node)

        k_res = self._keyword_search(question, keywords=query_keywords)
        s_res = self.graph.semantic_search(question) if hasattr(self.graph, "semantic_search") else []
        seen = set(k_res)
        node_results = k_res + [x for x in s_res if x not in seen]

        combined_nodes = edge_seed_nodes + [n for n in node_results if n not in edge_seed_set]
        fallback_nodes = combined_nodes[:max_current_nodes]
        candidate_node_list = fallback_nodes[:max_current_nodes]
        candidate_edge_block, candidate_node_block, candidate_edge_lookup, allowed_entry_nodes = build_candidate_blocks(
            edge_results,
            candidate_node_list,
        )

        beams: List[dict] = []
        last_current_content = "None"
        last_visited_summary = "None"
        last_history_str = "None"

        for round_idx in range(max_rounds):
            beams = dedupe_beams(beams)
            current_nodes = [beam["frontier"] for beam in beams]
            candidate_bundles = refresh_beam_candidates(beams) if beams else []
            prompt, last_current_content, last_visited_summary, last_history_str = build_prompt(current_nodes, candidate_bundles)

            decision = {}
            tool = None
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Output valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                self._record_usage(res)
                decision = self._parse_json_response(res.choices[0].message.content)
                tool = decision.get("tool")
                logger.info(f"Step {round_idx + 2}: {tool} - {decision}")
                logger.info(f"walk_decision_raw={decision}")
            except Exception as e:
                logger.error(f"Step failed: {e}")
                history.append(f"Error: {str(e)}")

            if tool == "READ":
                logger.info("read_triggered_by=llm_decision")
                return self._clean_answer(decision.get("answer"))

            if tool == "SEARCH":
                selected_nodes = []
                selected_edge_lines = []

                entry_edge_ids = decision.get("entry_edge_ids") or []
                if isinstance(entry_edge_ids, list):
                    for edge_id in entry_edge_ids:
                        edge = candidate_edge_lookup.get(str(edge_id))
                        if not edge:
                            continue
                        if edge["edge_text"] not in selected_edge_lines:
                            selected_edge_lines.append(edge["edge_text"])
                        for node in (edge["source"], edge["target"]):
                            if node not in selected_nodes:
                                selected_nodes.append(node)

                entry_nodes = decision.get("entry_nodes") or []
                if isinstance(entry_nodes, list):
                    for node in entry_nodes:
                        node = str(node)
                        if node in allowed_entry_nodes and node not in selected_nodes:
                            selected_nodes.append(node)

                search_query = str(decision.get("query") or question)
                search_mode = str(decision.get("mode") or "hybrid")

                if selected_nodes:
                    if not selected_edge_lines:
                        selected_edge_lines = edge_evidence_lines
                    beams = apply_search_results(
                        selected_nodes[:max_current_nodes],
                        selected_edge_lines,
                        f"SEARCH(select, '{search_query}')",
                    )
                    continue

                searched_nodes, searched_edges, note = run_search(search_query, search_mode)
                if searched_nodes:
                    beams = apply_search_results(searched_nodes, searched_edges, note)
                    continue

                fallback_selected_nodes = run_rule_search(search_query, search_mode)
                if fallback_selected_nodes:
                    beams = apply_search_results(
                        fallback_selected_nodes,
                        searched_edges,
                        f"SEARCH({search_mode}, '{search_query}')",
                    )
                    continue

                if fallback_nodes:
                    beams = apply_search_results(
                        fallback_nodes,
                        edge_evidence_lines,
                        f"SEARCH(fallback, '{question}')",
                    )
                    continue

            if not beams:
                break

            candidate_lookup = {
                bundle["frontier"]: {opt["node"]: opt for opt in bundle.get("options", [])}
                for bundle in candidate_bundles
            }
            raw_steps = decision.get("steps", []) if isinstance(decision, dict) else []
            if not raw_steps and decision.get("node"):
                target = str(decision.get("node")).strip()
                for bundle in candidate_bundles:
                    if target in candidate_lookup.get(bundle["frontier"], {}):
                        raw_steps = [{"from": bundle["frontier"], "to": target}]
                        break

            valid_steps, invalid_reason = self._validate_walk_steps(raw_steps, candidate_lookup, node_visit_counts)
            logger.info(f"walk_steps_validated={valid_steps} invalid_reason={invalid_reason}")

            if not valid_steps:
                valid_steps = self._fallback_walk_selection(beams)
                if valid_steps:
                    logger.info(f"walk_fallback_used={[(s['from'], s['to']) for s in valid_steps]}")
                    history.append(f"WALK fallback -> {[(s['from'], s['to']) for s in valid_steps]}")

            if not valid_steps:
                history.append("WALK -> No candidates.")
                break

            walked_nodes = []
            traversed_edges = []
            next_beams = []
            applied_pairs = []
            chosen_by_frontier = {step["from"]: step for step in valid_steps}

            for beam in beams:
                step = chosen_by_frontier.get(beam["frontier"])
                if step is None:
                    next_beams.append(beam)
                    continue
                candidate = step["candidate"]
                new_node = candidate["node"]
                new_beam = {
                    "path": beam["path"] + [new_node],
                    "frontier": new_node,
                    "candidates": [],
                }
                next_beams.append(new_beam)
                walked_nodes.append(new_node)
                traversed_edges.append(candidate["edge_text"])
                applied_pairs.append((step["from"], new_node))
                history.append(
                    f"WALK (llm) -> Moved to {new_node} (from {step['from']}, score={candidate['score']:.3f})"
                )
                node_visit_counts[new_node] = node_visit_counts.get(new_node, 0) + 1

            beams = dedupe_beams(next_beams)
            recent_traversed_edges = (recent_traversed_edges + traversed_edges)[-self.recent_edges_max_items:]
            logger.info(f"walk_steps_applied={applied_pairs}")

            if walked_nodes:
                cache_nodes_and_edges(walked_nodes)
                update_cached_evidence(walked_nodes)
                node_contents = [visited_node_content[n] for n in walked_nodes if n in visited_node_content]
                new_facts = self._extract_step_evidence(
                    question,
                    node_contents,
                    traversed_edges,
                    self._render_evidence_pool(evidence_pool),
                )
                logger.info(f"new_step_evidence={new_facts}")
                evidence_pool = self._merge_evidence_pool(evidence_pool, new_facts, query_keywords)
                logger.info(f"evidence_pool_after_merge={self._render_evidence_pool(evidence_pool)}")

        logger.warning("Max steps reached. Attempting to answer from compact evidence view.")
        try:
            final_evidence = self._render_evidence_pool(evidence_pool)
            recent_edges_view = "\n".join(recent_traversed_edges[-self.recent_edges_max_items:]) if recent_traversed_edges else "None"
            fallback_prompt = f"""You have explored the graph and should now give the best supported answer.
Use the evidence pool as the primary basis.
If the evidence is partial, give the best supported answer without inventing extra facts.

Question:
{question}

Evidence Pool:
{final_evidence}

Current Node Content:
{last_current_content}

Recent Traversed Edges:
{recent_edges_view}

Return ONLY the answer text.
"""
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": fallback_prompt}],
                temperature=0.0,
            )
            self._record_usage(res)
            logger.info("read_triggered_by=max_rounds_fallback")
            return self._clean_answer(res.choices[0].message.content)
        except Exception:
            return "Unknown"
