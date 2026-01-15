import json
import logging
import random
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from amadeus.code.agents.base import BaseAgent

logger = logging.getLogger("Amadeus.Questioner")

class QuestionerAgent(BaseAgent):
    def __init__(self, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
        super().__init__(model_name, api_base, api_key)
        self.reading_steiner["SERN_PROTOCOL"] = []
        self.static_prompt = """You are 'The Questioner', the Adversarial Attacker of the Amadeus Memory System.
Your goal is to generate questions based on the provided 'Buffer Context' to test if the memory system has correctly compressed and stored the information.

**ATTACK MODES (MIXED STRATEGY):**
1. **FACT_CHECK**: Ask about explicit facts (Who, What, When). Test if basic info is stored.
2. **MULTI_HOP**: Ask questions requiring connecting two pieces of info. Test relation storage.
3. **IMPLICIT_INFER**: Ask about implied details (e.g., feelings, unstated locations). Test deep understanding.

**RULES:**
- Questions must be relevant to the main characters (e.g., Caroline, Melanie) if they appear.
- The 'ground_truth' must be supported by the Buffer text.
- Do NOT ask about meta-data like "What is in line 1?".

**INSTRUCTIONS FOR REASONING (CoT):**
In your "chain_of_thought" section, you MUST:
1. Analyze the current Buffer Context to identify key entities, events, and relationships.
2. **PROTOCOL REFERENCE**: Check the 'READING STEINER' section below. If specialized IDs (like Q-005) are listed there, cite the specific ID you are using. If the section is empty or says 'No specialized protocols', state "Standard Adversarial Logic" and do NOT invent or hallucinate any IDs.
3. Describe your step-by-step logic for drafting the question and why this specific protocol is effective here(if used).
4. Formulate the questions and verify their ground truth against the text.

**OUTPUT FORMAT (JSON):**
{
  "chain_of_thought": "[Detailed reasoning.]",
  "used_steiner_ids": [],
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

    def generate_questions(self, buffer_content: str, num_questions: int = 3) -> tuple[List[Dict[str, Any]], str, List[str]]:
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            cot = data.get("chain_of_thought", "No CoT provided.")
            used_ids = data.get("used_steiner_ids", [])
            if "chain_of_thought" in data:
                logger.info(f"Questioner CoT: {cot}")
            if used_ids:
                logger.info(f"🏷️ Questioner used protocols: {used_ids}")
            return data.get("questions", []), cot, used_ids
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return [], "", []
