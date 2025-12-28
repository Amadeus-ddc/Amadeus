import networkx as nx
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Amadeus.Graph")

class MemoryGraph:
    """
    Amadeus 的长期记忆核心。
    这是一个基于 NetworkX 的轻量级图数据库实现，完全白盒化。
    """
    def __init__(self, storage_path: str = "data/memory_graph.json"):
        self.storage_path = storage_path
        # 使用多重有向图 (MultiDiGraph)，允许两点间存在多种关系
        self.graph = nx.MultiDiGraph()
        self._ensure_storage()
        self.load()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    # --- 核心原语 (Primitives) ---

    def add_node(self, name: str, type: str, description: str):
        """
        [ADD/UPDATE] 算子的底层实现。
        如果节点存在，融合描述；如果不存在，创建。
        """
        now = datetime.now().isoformat()
        
        if self.graph.has_node(name):
            # 追加描述而不是覆盖
            old_desc = self.graph.nodes[name].get("description", "")
            if description and description not in old_desc:
                new_desc = f"{old_desc}; {description}"
                self.graph.nodes[name]["description"] = new_desc
                self.graph.nodes[name]["updated_at"] = now
                logger.info(f"🔄 Node Updated: {name}")
        else:
            self.graph.add_node(
                name, 
                type=type, 
                description=description,
                created_at=now,
                updated_at=now
            )
            logger.info(f"➕ Node Created: {name} ({type})")

    def add_edge(self, source: str, target: str, relation: str):
        """
        [ADD] 关系的底层实现。
        """
        # 确保端点存在 (防御性编程)
        if not self.graph.has_node(source):
            self.add_node(source, "Unknown", "Created implicitly by relation")
        if not self.graph.has_node(target):
            self.add_node(target, "Unknown", "Created implicitly by relation")

        # 检查是否已存在完全相同的边，避免重复
        if not self.graph.has_edge(source, target, key=relation):
            self.graph.add_edge(
                source, 
                target, 
                key=relation, 
                relation=relation,
                created_at=datetime.now().isoformat()
            )
            logger.info(f"🔗 Edge Created: {source} --[{relation}]--> {target}")

    def delete_node(self, name: str):
        """
        [DELETE] 算子。
        """
        if self.graph.has_node(name):
            self.graph.remove_node(name)
            logger.info(f"❌ Node Deleted: {name}")

    # --- 检索与上下文构建 (Retrieval) ---

    def get_full_state(self) -> str:
        """
        将图序列化为文本，作为 Builder 的 context。
        """
        if self.graph.number_of_nodes() == 0:
            return "Memory Graph is empty."

        text_representation = ["Current Long-term Memory:"]
        
        # 1. 列出实体
        text_representation.append("\n[Entities]")
        for node, data in self.graph.nodes(data=True):
            desc = data.get('description', 'No description')
            dtype = data.get('type', 'Entity')
            text_representation.append(f"- {node} ({dtype}): {desc}")

        # 2. 列出关系
        text_representation.append("\n[Relationships]")
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'related_to')
            text_representation.append(f"- {u} --{relation}--> {v}")

        return "\n".join(text_representation)

    def search(self, query: str) -> str:
        """
        简单的关键词搜索，模拟 Search 算子。
        """
        # 后续接入 Vector DB (Chroma/FAISS) 做真正的语义检索
        hits = []
        for node, data in self.graph.nodes(data=True):
            if query.lower() in node.lower() or query.lower() in data.get("description", "").lower():
                hits.append(node)
        
        if not hits:
            return "No direct memory found."
        
        # 返回命中节点的一跳邻居 (1-hop sub-graph)
        result_text = []
        for hit in hits:
            result_text.append(f"Found Entity: {hit}")
            neighbors = self.graph[hit]
            for neighbor, edge_data in neighbors.items():
                for _, edge_attr in edge_data.items():
                    result_text.append(f"  -> {edge_attr['relation']} -> {neighbor}")
        
        return "\n".join(result_text)

    # --- 持久化 ---

    def save(self):
        data = nx.node_link_data(self.graph, edges="links")
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data, edges="links")
                logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
                self.graph = nx.MultiDiGraph()