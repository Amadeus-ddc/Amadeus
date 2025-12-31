import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger("Amadeus.Buffer")


class ShortTermBuffer:
    """
    Amadeus 的短期记忆（Short-term Buffer）。
    轻量级实现，用于暂存最近的对话或信息片段。
    
    核心职责：
    1. 存储临时的、未经处理的信息（原始对话文本）
    2. 与 Builder 协作：将 Buffer 内容压缩到 Graph，保留需要 WAIT 的片段
    3. 控制容量，防止无限增长
    """
    
    def __init__(self, max_size: int = 10):
        """
        初始化短期记忆缓冲区。
        
        Args:
            max_size: 缓冲区最大条目数（超出后自动清理最旧的）
        """
        self.max_size = max_size
        self.items: List[dict] = []
        logger.info(f"Buffer initialized with max_size={max_size}")
    
    def add(self, content: str, metadata: Optional[dict] = None):
        """
        向缓冲区添加新内容。
        
        Args:
            content: 文本内容（对话片段、事件描述等）
            metadata: 可选的元数据（如时间戳、来源等）
        """
        if not content or not content.strip():
            logger.warning("Attempted to add empty content to buffer, skipping.")
            return
        
        item = {
            "content": content.strip(),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.items.append(item)
        logger.info(f"📝 Added to buffer (size: {len(self.items)}/{self.max_size})")
        
        # 自动清理：FIFO 策略
        if len(self.items) > self.max_size:
            removed = self.items.pop(0)
            logger.info(f"🗑️ Buffer overflow, removed oldest item: '{removed['content'][:50]}...'")
    
    def get_all_content(self) -> str:
        """
        获取缓冲区中所有内容的文本表示。
        用于传递给 Builder 进行分析。
        
        Returns:
            合并后的文本字符串
        """
        if not self.items:
            return ""
        
        text_parts = []
        for idx, item in enumerate(self.items, 1):
            text_parts.append(f"[{idx}] {item['content']}")
        
        return "\n".join(text_parts)
    
    def clear(self):
        """
        清空缓冲区。
        """
        count = len(self.items)
        self.items = []
        logger.info(f"🧹 Buffer cleared ({count} items removed)")
    
    def keep(self, texts_to_keep: List[str]):
        """
        仅保留指定的文本内容，删除其他所有内容。
        这是 Builder 决策的结果：某些信息需要继续 WAIT。
        
        Args:
            texts_to_keep: Builder 返回的需要保留的文本列表
        """
        if not texts_to_keep:
            self.clear()
            return
        
        # 保留匹配的项
        kept_items = []
        for item in self.items:
            if any(text in item['content'] or item['content'] in text for text in texts_to_keep):
                kept_items.append(item)
        
        removed_count = len(self.items) - len(kept_items)
        self.items = kept_items
        
        logger.info(f"⏳ Kept {len(kept_items)} items in buffer (removed {removed_count})")
    
    def size(self) -> int:
        """
        获取当前缓冲区的条目数。
        
        Returns:
            缓冲区中的条目数
        """
        return len(self.items)
    
    def is_empty(self) -> bool:
        """
        检查缓冲区是否为空。
        
        Returns:
            True if empty, False otherwise
        """
        return len(self.items) == 0
    
    def __repr__(self) -> str:
        return f"ShortTermBuffer(size={len(self.items)}/{self.max_size})"
