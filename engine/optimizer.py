import logging
import json
from typing import List, Dict, Any
from openai import OpenAI
from amadeus.agents.questioner import QuestionerAgent
from amadeus.agents.builder import BuilderAgent
from amadeus.agents.answerer import AnswererAgent

logger = logging.getLogger("Amadeus.Optimizer")

class AdversarialOptimizer:
    def __init__(self, questioner: QuestionerAgent, builder: BuilderAgent, answerer: AnswererAgent, model_name: str = "gpt-4-turbo"):
        self.questioner = questioner
        self.builder = builder
        self.answerer = answerer
        self.client = OpenAI()
        self.model_name = model_name

    def step(self, buffer_content: str, action_log: List[str] = None):
        logger.info("⚔️ Starting Adversarial Optimization Step...")
        
        # 1. Generate Attack Batch
        questions = self.questioner.generate_questions(buffer_content, num_questions=3)
        logger.info(f"Generated {len(questions)} questions.")

        for q_item in questions:
            question = q_item.get("question")
            ground_truth = q_item.get("ground_truth")
            q_type = q_item.get("type")
            
            logger.info(f"Testing Question ({q_type}): {question}")
            
            # 2. Forward Pass (Answerer)
            prediction = self.answerer.answer(question)
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Ground Truth: {ground_truth}")

            # 3. Evaluation & Gradient
            self._evaluate_and_update(q_item, prediction, buffer_content, action_log)

    def _evaluate_and_update(self, q_item: Dict, prediction: str, buffer_content: str, action_log: List[str] = None):
        # Critic LLM
        action_log_str = "\n".join(action_log) if action_log else "No recent graph updates."
        
        prompt = f"""
You are the 'Critic' of the Amadeus Memory System.
Your job is to evaluate the Agent's performance.
This is a Zero-Sum Game:
- If Prediction is WRONG -> Questioner Wins. We must optimize Builder/Answerer.
- If Prediction is CORRECT -> System Wins. We must optimize Questioner (to ask harder questions).

**CRITICAL CONTEXT MISMATCH HANDLING:**
- The **Ground Truth** is derived ONLY from the current 5-minute conversation fragment (Buffer).
- The **Prediction** is derived from the Global Memory Graph (All history).
- **IF** Ground Truth says "Unknown", "Not mentioned", or "Does not say", **AND** Prediction provides a specific, plausible answer:
  -> **JUDGMENT: CORRECT**. (The system successfully retrieved facts from previous history).
  -> **ACTION**: Optimize Questioner (Tell it to ask about things actually present in the current buffer).

**BLAME ASSIGNMENT LOGIC (Use 'Builder Activity Log'):**
- If Prediction is WRONG (e.g., "Unknown" or incorrect fact):
  - Check the **Builder Activity Log** below.
  - **IF** the missing information appears in the Log (meaning Builder JUST added it):
    -> **BLAME: ANSWERER**. (The info is in the graph, but Answerer failed to find it).
    -> **OPERATOR**: SEARCH/WALK.
  - **IF** the missing information is NOT in the Log:
    -> **BLAME: BUILDER**. (Builder failed to extract/add it).
    -> **OPERATOR**: ADD/UPDATE.

**CONTEXT:**
- Question: "{q_item['question']}"
- Ground Truth: "{q_item['ground_truth']}"
- Prediction: "{prediction}"
- Question Type: {q_item['type']}

**BUILDER ACTIVITY LOG (Recent Graph Updates):**
{action_log_str}

**TASK:**
1. Determine if the Prediction is CORRECT.

**SCENARIO A: PREDICTION IS WRONG (Defender Failed)**
- Blame: **BUILDER** or **ANSWERER** (Based on Logic above).
- Operator: ADD/UPDATE/DELETE/WAIT (Builder) or SEARCH/WALK/READ (Answerer).
- Gradient: Instruction to fix the Defender.

**SCENARIO B: PREDICTION IS CORRECT (Attacker Failed)**
- Blame: **QUESTIONER**.
- Operator: **GENERATE**.
- Gradient: Instruction to find more subtle/implicit/harder details in the Buffer.

**OUTPUT FORMAT (JSON):**
{{
  "correct": boolean,
  "blame": "BUILDER" | "ANSWERER" | "QUESTIONER",
  "operator": "ADD" | "UPDATE" | "DELETE" | "WAIT" | "SEARCH" | "WALK" | "READ" | "GENERATE",
  "gradient": "The instruction to add to the operator's guidelines.",
  "fix_instruction": "If blame is BUILDER, provide a specific instruction to fix the graph state immediately. Else null."
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            
            blame = result.get("blame")
            operator = result.get("operator")
            gradient = result.get("gradient")

            if not result.get("correct"):
                logger.warning(f"❌ DEFENDER FAILED. Blame: {blame}")
                
                # Apply Policy Update to Defender
                if blame == "BUILDER":
                    self.builder.update_guideline(operator, gradient)
                    # Apply State Fix
                    fix = result.get("fix_instruction")
                    if fix:
                        self.builder.force_update(fix)
                        
                elif blame == "ANSWERER":
                    self.answerer.update_guideline(operator, gradient)
            else:
                logger.info(f"✅ DEFENDER SUCCEEDED. Optimizing Questioner...")
                # Apply Policy Update to Attacker
                if blame == "QUESTIONER" or operator == "GENERATE":
                     self.questioner.update_guideline("GENERATE", gradient)
                
        except Exception as e:
            logger.error(f"Optimizer Error: {e}")
