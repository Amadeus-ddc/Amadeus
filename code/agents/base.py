import logging
import os
from typing import Dict, List
from openai import OpenAI

logger = logging.getLogger("Amadeus.Agent")

class BaseAgent:
    def __init__(self, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
        # Priority: Explicit Args > Environment Variables > Default
        base_url = api_base or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.static_prompt: str = ""
        self.operator_guidelines: Dict[str, List[str]] = {}
        self.usage_stats = {
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        self.usage_stats["api_calls"] += 1
        if not usage:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        self.usage_stats["prompt_tokens"] += prompt_tokens
        self.usage_stats["completion_tokens"] += completion_tokens
        self.usage_stats["total_tokens"] += total_tokens

    def _format_guidelines(self) -> str:
        if not self.operator_guidelines:
            return ""
        
        text = "\n\n**DYNAMIC OPERATOR GUIDELINES (EVOLVED STRATEGIES):**\n"
        for op, rules in self.operator_guidelines.items():
            text += f"\n[{op}]:\n"
            for i, rule in enumerate(rules, 1):
                text += f"  {i}. {rule}\n"
        return text

    def get_full_prompt(self) -> str:
        return self.static_prompt + self._format_guidelines()

    def update_guideline(self, operator: str, rule: str):
        if operator not in self.operator_guidelines:
            self.operator_guidelines[operator] = []
        
        # Simple deduplication
        if rule not in self.operator_guidelines[operator]:
            self.operator_guidelines[operator].append(rule)
            logger.info(f"📈 Guideline Updated for [{operator}]: {rule}")
