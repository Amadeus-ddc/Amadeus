import json
import logging
import random
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from amadeus.agents.base import BaseAgent

logger = logging.getLogger("Amadeus.Questioner")

class QuestionerAgent(BaseAgent):
    def __init__(self, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None, logger: logging.Logger = None):
        super().__init__(model_name, api_base, api_key, logger)
        if self.logger is None:
            self.logger = logging.getLogger("Amadeus.Questioner")
        self.static_prompt = """You are 'The Questioner', the Adversarial Attacker of the Amadeus Memory System.
Your goal is to generate questions based on the provided 'Buffer Context' to test if the memory system has correctly compressed and stored the information.

**CRITICAL INSTRUCTION: DO NOT SURRENDER EASILY.**
Even if the text seems simple, you MUST try to generate at least one valid question to verify the memory system has captured the details correctly.

**ATTACK MODES (MIXED STRATEGY):**
1. **FACT_CHECK**: Ask about explicit facts (Who, What, When). Test if basic info is stored.
2. **MULTI_HOP**: Ask questions requiring connecting two pieces of info. Test relation storage.
3. **IMPLICIT_INFER**: Ask about implied details (e.g., feelings, unstated locations). Test deep understanding.

**RULES:**
- Questions must be relevant to the main characters (e.g., Caroline, Melanie) if they appear.
- The 'ground_truth' must be supported by the Buffer text.
- Do NOT ask about meta-data like "What is in line 1?".

**OUTPUT FORMAT (JSON):**
{
  "questions": [
    {
      "question": "Where did Caroline go?",
      "ground_truth": "She went to the kitchen.",
      "type": "FACT_CHECK",
      "reasoning": "Explicitly stated in text."
    }
  ]
}
"""
        self.operator_guidelines = {
            "GENERATE": ["Focus on high-value information.", "Avoid trivial details like background colors unless relevant."]
        }

    def generate_questions(self, buffer_content: str, num_questions: int = 3) -> List[Dict[str, Any]]:
        modes = ["FACT_CHECK", "MULTI_HOP", "IMPLICIT_INFER"]
        selected_modes = [random.choice(modes) for _ in range(num_questions)]
        
        prompt = f"""
{self.get_full_prompt()}

**CURRENT BUFFER CONTEXT:**
{buffer_content}

**TASK:**
Generate {num_questions} questions.
Target Modes for this batch: {', '.join(selected_modes)}
"""
        try:
            response = self.call_llm(
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get("questions", [])
        except Exception as e:
            self.logger.error(f"Failed to generate questions: {e}")
            self.logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return []
