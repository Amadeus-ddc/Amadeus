import logging
from typing import List

logger = logging.getLogger("Amadeus.Buffer")

class TimeWindowBuffer:
    def __init__(self, trigger_threshold: int = 3):
        self.buffer_items: List[str] = []
        self.trigger_threshold = trigger_threshold
        self.deferred_items: List[str] = [] # Holds content marked as WAIT

    def add(self, text: str):
        if text:
            self.buffer_items.append(text)

    def is_full(self) -> bool:
        return len(self.buffer_items) >= self.trigger_threshold

    def get_content(self) -> str:
        # Place deferred items first to maintain context continuity
        combined = self.deferred_items + self.buffer_items
        return "\n".join([f"- {item}" for item in combined])

    def clear(self, keep_items: List[str] = None):
        """Clear the current Buffer, but retain items marked as WAIT by the Builder."""
        self.buffer_items = []
        self.deferred_items = keep_items if keep_items else []
        if self.deferred_items:
            logger.info(f"🔄 Buffer carrying over {len(self.deferred_items)} items (WAIT).")