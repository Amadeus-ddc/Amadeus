import networkx as nx
import json
import logging
import os
import re
import numpy as np
from typing import List, Any

logger = logging.getLogger("Amadeus.Graph")

class MemoryGraph:
    def __init__(self, storage_path: str = "data/memory_graph.json", embedder: Any = None):
        self.storage_path = storage_path
        self.graph = nx.MultiDiGraph()
        self.embedder = embedder
        self._ensure_storage()
        self.load()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def add_node(self, name: str, type: str, description: str):
        if not name: return
        
        # Helper to update embedding
        def update_embedding(n, desc):
            if self.embedder:
                try:
                    text = f"{n}: {desc}"
                    emb = self.embedder.embed(text)
                    if emb is not None:
                        self.graph.nodes[n]["embedding"] = np.array(emb)
                except Exception as e:
                    logger.warning(f"Failed to embed node {n}: {e}")

        if self.graph.has_node(name):
            old_desc = self.graph.nodes[name].get("description", "")
            
            if description and len(description) > 2 and description not in old_desc:
                # 智能合并：简单的字符串拼接，实际可升级为 LLM 摘要
                new_desc = f"{old_desc} | {description}".strip(" | ")
                self.graph.nodes[name]["description"] = new_desc
                update_embedding(name, new_desc)
        else:
            self.graph.add_node(name, type=type, description=description)
            update_embedding(name, description)

    def add_edge(self, source: str, target: str, relation: str, timestamp: str = None):
        if not source or not target: return
        if not self.graph.has_node(source): self.add_node(source, "Unknown", "Created by relation")
        if not self.graph.has_node(target): self.add_node(target, "Unknown", "Created by relation")
        
        # Normalize timestamp
        if timestamp == "None" or timestamp == "Unknown Date": timestamp = None

        # Check for existing edges to update or deduplicate
        if self.graph.has_edge(source, target):
            edges = self.graph[source][target]
            for key, attrs in edges.items():
                if attrs.get('relation') == relation:
                    old_ts = attrs.get('timestamp')
                    if old_ts == "None" or old_ts == "Unknown Date": old_ts = None
                    
                    # Update if refining date (Unknown -> Known)
                    if not old_ts and timestamp:
                        self.graph[source][target][key]['timestamp'] = timestamp
                        return
                    
                    # Deduplicate (Same relation, same date)
                    if old_ts == timestamp:
                        return
        
        # Add new edge (allow NetworkX to generate unique key)
        self.graph.add_edge(source, target, relation=relation, timestamp=timestamp)

    def delete_node(self, name: str):
        if self.graph.has_node(name):
            self.graph.remove_node(name)
            logger.info(f"❌ Node Deleted: {name}")

    def delete_edge(self, source: str, target: str):
        if self.graph.has_edge(source, target):
            # Remove all edges between source and target
            keys = list(self.graph[source][target].keys())
            for k in keys:
                self.graph.remove_edge(source, target, key=k)
            logger.info(f"✂️ Edge Deleted: {source} -> {target}")

    def get_full_state(self) -> str:
        if self.graph.number_of_nodes() == 0: return "Graph is empty."
        nodes = []
        for n, d in self.graph.nodes(data=True):
            desc = d.get('description', '')
            nodes.append(f"{n}: {desc}")
        
        edges = []
        for u, v, d in self.graph.edges(data=True):
            rel = d.get('relation')
            ts = d.get('timestamp')
            ts_str = f" [Time: {ts}]" if ts else ""
            edges.append(f"{u} --{rel}{ts_str}--> {v}")
        
        return "Nodes:\n" + "\n".join(nodes[:500]) + "\nEdges:\n" + "\n".join(edges[:500])

    def primitive_search(self, query: str) -> List[str]:
        # If embedder is available, use semantic search
        if self.embedder:
            return self.semantic_search(query)

        hits = []
        stop_words = {"what", "which", "who", "where", "when", "how", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on", "at", "did", "does", "do"}
        keywords = [k.lower() for k in re.findall(r'\w+', query) if k.lower() not in stop_words]
        
        for node, data in self.graph.nodes(data=True):
            content = f"{node} {data.get('description', '')}".lower()
            score = 0
            for k in keywords:
                if k in node.lower(): score += 10 # 节点名匹配权重高
                elif k in content: score += 1
            if score > 0:
                hits.append((node, score))
        
        hits.sort(key=lambda x: x[1], reverse=True)
        return [h[0] for h in hits[:10]]

    def semantic_search(self, query: str, top_k: int = 10) -> List[str]:
        try:
            query_vec = self.embedder.embed(query)
            if query_vec is None: return []
            query_vec = np.array(query_vec)
            
            candidates = []
            for node, data in self.graph.nodes(data=True):
                if "embedding" in data:
                    vec = data["embedding"]
                    # Cosine similarity
                    score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-9)
                    candidates.append((node, score))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [c[0] for c in candidates[:top_k]]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def primitive_get_neighbors(self, node_names: List[str]) -> str:
        view = []
        for name in node_names:
            if not self.graph.has_node(name): continue
            desc = self.graph.nodes[name].get("description", "")
            
            view.append(f"📍 At Node: [{name}] - {desc}")
            neighbors = self.graph.out_edges(name, data=True)
            for _, target, data in neighbors:
                rel = data.get('relation', 'related')
                # Show Edge timestamp
                edge_ts = data.get('timestamp')
                edge_ts_str = f" [Time: {edge_ts}]" if edge_ts else ""
                
                t_desc = self.graph.nodes[target].get("description", "")
                preview = t_desc[:50] + "..." if len(t_desc) > 50 else t_desc
                view.append(f"   --[{rel}{edge_ts_str}]--> 🔭 Candidate: [{target}] ({preview})")
        return "\n".join(view)

    def primitive_read(self, node_names: List[str]) -> str:
        content = []
        for name in node_names:
            if self.graph.has_node(name):
                d = self.graph.nodes[name]
                
                # Collect temporal context from incoming edges
                temporal_context = []
                in_edges = self.graph.in_edges(name, data=True)
                for u, _, data in in_edges:
                    ts = data.get('timestamp')
                    if ts:
                        rel = data.get('relation', 'related')
                        temporal_context.append(f"   <-[{rel} at {ts}]-- {u}")
                
                temporal_str = "\n" + "\n".join(temporal_context) if temporal_context else ""
                
                content.append(f"Entity: {name} ({d.get('type','Unknown')})\nDesc: {d.get('description','')}{temporal_str}")
        return "\n".join(content)

    def save(self):
        # Convert embeddings to lists for JSON serialization
        for _, data in self.graph.nodes(data=True):
            if "embedding" in data and isinstance(data["embedding"], np.ndarray):
                data["embedding"] = data["embedding"].tolist()

        data = nx.node_link_data(self.graph)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.graph = nx.node_link_graph(json.load(f))
                    # Convert embeddings back to numpy arrays
                    for _, node_data in self.graph.nodes(data=True):
                        if "embedding" in node_data:
                            node_data["embedding"] = np.array(node_data["embedding"])
            except: pass