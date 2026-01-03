import logging
import os
import time
import random
from typing import Dict, List, Any
from openai import OpenAI

logger = logging.getLogger("Amadeus.Agent")

class BaseAgent:
    def __init__(self, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None, logger: logging.Logger = None):
        # Priority: Explicit Args > Environment Variables > Default
        base_url = api_base or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.static_prompt: str = ""
        self.operator_guidelines: Dict[str, List[str]] = {}
        self.logger = logger or logging.getLogger("Amadeus.Agent")

    def call_llm(self, messages: List[Dict[str, str]], response_format: Dict[str, str] = None, temperature: float = 0.0, max_retries: int = 3, timeout: float = 300.0) -> Any:
        """
        Wrapper for LLM API calls with retry logic and error handling.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    timeout=timeout
                )
                return response
            except Exception as e:
                self.logger.warning(f"LLM Call Failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"LLM Call Failed after {max_retries} attempts. Error: {e}")
                    raise e

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
            self.logger.info(f"📈 Guideline Updated for [{operator}]: {rule}")
