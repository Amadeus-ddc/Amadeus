import logging
import os
import time
import random
import json
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
        self.max_guidelines = 10
        self.logger = logger or logging.getLogger("Amadeus.Agent")

    def call_llm(self, messages: List[Dict[str, str]], response_format: Dict[str, str] = None, temperature: float = 0.0) -> Any:
        """
        Wrapper for LLM API calls.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format=response_format,
                temperature=temperature
            )
            return response
        except Exception as e:
            self.logger.error(f"LLM Call Failed: {e}")
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

            # Check for consolidation
            if len(self.operator_guidelines[operator]) > self.max_guidelines:
                self._consolidate_guidelines(operator)

    def _consolidate_guidelines(self, operator: str):
        rules = self.operator_guidelines.get(operator, [])
        
        prompt = f"""You are the Guideline Optimizer.
The following list of guidelines for the operator '{operator}' has grown too long and may contain redundancies.

Current Guidelines:
{json.dumps(rules, indent=2)}

**GOAL:**
1. Merge semantically similar rules.
2. Remove obsolete or less important rules.
3. Keep the total number of rules under {self.max_guidelines}.
4. Preserve the most critical instructions for the agent.
5. Ensure the rules are concise.

Output JSON: {{ "consolidated_rules": ["rule1", "rule2", ...] }}
"""
        try:
            response = self.call_llm(
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            res = json.loads(response.choices[0].message.content)
            new_rules = res.get("consolidated_rules", [])
            if new_rules:
                self.operator_guidelines[operator] = new_rules
                self.logger.info(f"♻️ Guidelines Consolidated for [{operator}]: {len(rules)} -> {len(new_rules)}")
        except Exception as e:
            self.logger.error(f"Guideline Consolidation Failed: {e}")
