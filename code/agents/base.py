import json
import logging
import os
import re
from typing import Dict, List, Optional
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

    def _strip_visible_thinking(self, content: Optional[str]) -> str:
        text = (content or "").strip()
        if not text:
            return ""

        think_close = re.search(r"</think\s*>", text, re.IGNORECASE)
        if think_close:
            tail = text[think_close.end():].strip()
            if tail:
                return tail
            return text

        if text.startswith("Thinking Process:"):
            markers = ["\n\n{", "\n\n[", "\n\n最终答案", "\n\nFinal Answer", "\n\nAnswer:"]
            for marker in markers:
                idx = text.find(marker)
                if idx != -1:
                    tail = text[idx + 2 :].strip()
                    if tail:
                        return tail

            lines = text.splitlines()
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if stripped in {"测试成功", "OK"}:
                    tail = "\n".join(lines[idx:]).strip()
                    if tail:
                        return tail

        return text

    def _extract_json_text(self, content: Optional[str]) -> str:
        text = self._strip_visible_thinking(content)
        if not text:
            return ""

        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        start = text.find("{")
        if start == -1:
            start = text.find("[")
        if start == -1:
            return text.strip()

        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1].strip()
        return text[start:].strip()

    def _parse_json_content(self, content: Optional[str]):
        return json.loads(self._extract_json_text(content))

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
