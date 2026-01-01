import logging
import json
from typing import List, Dict, Any
from openai import OpenAI
from amadeus.agents.questioner import QuestionerAgent
from amadeus.agents.builder import BuilderAgent
from amadeus.agents.answerer import AnswererAgent

logger = logging.getLogger("Amadeus.Optimizer")

class AdversarialOptimizer:
    def __init__(self, questioner: QuestionerAgent, builder: BuilderAgent, answerer: AnswererAgent, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
        self.questioner = questioner
        self.builder = builder
        self.answerer = answerer
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name

    def step(self, buffer_content: str, action_log: List[str] = None):
        logger.info("⚔️ Starting True Adaptive Self-Play...")
        
        history = []
        iteration = 0
        # 熔断机制：防止无限烧钱，但上限设高一点
        HARD_LIMIT = 10 
        
        # 初始状态：攻击者非常激进
        consecutive_wins = 0
        consecutive_useless_questions = 0
        
        while iteration < HARD_LIMIT:
            iteration += 1
            logger.info(f"--- Round {iteration} ---")

            # 1. 动态生成攻击 (Attack Generation)
            questions = self._generate_adaptive_attack(buffer_content, history)
            
            if not questions:
                logger.info("🏳️ Questioner surrendered: No more meaningful questions to ask.")
                break

            # 2. 过滤重复 (Deduplication)
            existing_qs = {h['question'] for h in history}
            unique_questions = [q for q in questions if q['question'] not in existing_qs]
            
            if not unique_questions:
                consecutive_useless_questions += 1
                logger.warning(f"⚠️ Questioner generated duplicates. Strike {consecutive_useless_questions}/3")
                if consecutive_useless_questions >= 3:
                    logger.info("🛑 Stopping: Questioner is stuck in a loop.")
                    break
                continue
            else:
                consecutive_useless_questions = 0 # 重置计数器

            logger.info(f"🔥 Attack Batch: {len(unique_questions)} questions")

            # 3. 并行攻防 (Parallel Defense)
            import concurrent.futures
            batch_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self._process_single_duel, q_item, buffer_content, action_log): q_item for q_item in unique_questions}
                for future in concurrent.futures.as_completed(futures):
                    batch_results.append(future.result())

            # 4. 状态更新与收敛检查 (State Update & Convergence)
            round_failed = False
            for res in batch_results:
                history.append(res)
                if res['result'] == "FAIL":
                    round_failed = True
            
            if not round_failed:
                consecutive_wins += 1
                logger.info(f"🛡️ Defenders won this round. Streak: {consecutive_wins}")
                # 收敛条件：如果防御者连续赢了2轮（且每轮都有实质性问题），说明已经很稳了
                if consecutive_wins >= 2:
                    logger.info("🏆 Convergence Reached: System is robust.")
                    break
            else:
                consecutive_wins = 0
                logger.info("💥 Defense breached! Continuing optimization...")

    def _generate_adaptive_attack(self, buffer_content: str, history: List[Dict]) -> List[Dict]:
        """
        让 Questioner 观察历史，决定是否继续攻击，以及攻击什么。
        """
        # 简化的历史摘要
        history_summary = "\n".join([f"Q: {h['question']} -> {'✅ PASS' if h['result']=='PASS' else '❌ FAIL'}" for h in history[-10:]])
        
        prompt = f"""You are the Red Team Leader (Attacker).
Target Memory Buffer: "{buffer_content[:500]}..."

Previous Attacks & Results:
{history_summary}

**YOUR MISSION:**
Determine if there are still unexplored vulnerabilities or missing details in the memory.
- If the Defender failed recently: ATTACK HARDER on that specific topic.
- If the Defender passed: Try a TRICKIER angle or a different detail.
- If the buffer is fully covered and robust: STOP.

**OUTPUT FORMAT (JSON):**
{{
    "stop_attack": boolean, // Set true if no more valid questions exist
    "reason": "...",
    "questions": [ // Empty if stop_attack is true
        {{ "question": "...", "ground_truth": "...", "type": "detail/inference/negative" }}
    ]
}}
"""
        try:
            response = self.questioner.client.chat.completions.create(
                model=self.questioner.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7 # 保持一定的创造性
            )
            res = json.loads(response.choices[0].message.content)
            
            if res.get("stop_attack", False):
                return []
            
            return res.get("questions", [])
        except Exception as e:
            logger.error(f"Attack Generation Failed: {e}")
            return []

    def _process_single_duel(self, q_item, buffer_content, action_log):
        question = q_item.get("question")
        prediction = self.answerer.answer(question)
        eval_result = self._evaluate_and_update(q_item, prediction, buffer_content, action_log)
        
        return {
            "question": question,
            "result": "PASS" if eval_result and eval_result.get("is_correct") else "FAIL",
            "blame": eval_result.get("blame") if eval_result else "UNKNOWN"
        }

    def _evaluate_and_update(self, q_item: Dict, prediction: str, buffer_content: str, action_log: List[str] = None):
        # Critic LLM
        action_log_str = "\n".join(action_log) if action_log else "No recent graph updates."
        buffer_snippet = buffer_content[:500].replace("\n", " ")
        
        prompt = f"""You are the 'Meta-Critic' and 'Gradient Descent Optimizer' of the Amadeus Memory System.
Your goal: arbitrate the adversarial game between the [Questioner] (Attacker) and the [Builder/Answerer] (Defenders).

**GAME RULES (Zero-Sum):**
1. **Defenders Lose (Prediction WRONG)**: 
   - Identify WHY. Was the info missing (Builder fault) or not retrieved (Answerer fault)?
   - Generate a **Graph Patch** to fix the data immediately.
   - Generate a **Textual Gradient** to update the Agent's Prompt to prevent future errors.
2. **Defenders Win (Prediction CORRECT)**:
   - The Questioner failed to trick the system.
   - Generate a **Textual Gradient** to force the Questioner to ask harder/trickier questions next time.

**CRITICAL: GLOBAL vs LOCAL CONTEXT**
- **Ground Truth (GT)** is derived ONLY from the current Buffer.
- **Prediction** comes from the Global Memory Graph.
- **RULE**: If GT says "Unknown/Not mentioned" BUT Prediction gives a specific, plausible fact (likely from history), judge it as **CORRECT**.
  -> In this case, Blame QUESTIONER for asking about old history instead of current events.

**BLAME LOGIC (Who failed?):**
Analyze the [Builder Activity Log] and the [Question]:
- **BLAME BUILDER IF**: The specific *relationship* or *attribute* needed to answer is ABSENT from the Log. (Creating a Node is not enough; the connection must exist).
- **BLAME ANSWERER IF**: The exact answer DOES appear in the Log (meaning it was just added), but the Answerer still hallucinated or said "Unknown".

**INPUT DATA:**
- Text Buffer: "{buffer_snippet}..."
- Question: "{q_item['question']}"
- Ground Truth: "{q_item['ground_truth']}"
- Prediction: "{prediction}"
- Builder Log: "{action_log_str}"

**OUTPUT FORMAT (JSON):**
{{
  "is_correct": boolean,
  "blame": "BUILDER" | "ANSWERER" | "QUESTIONER",
  
  // SECTION 1: DATA REPAIR (Only if Prediction is WRONG and Blame is BUILDER)
  // Generate concrete operations to fix the graph NOW.
  "graph_patch": [
      {{ "action": "ADD", "subject": "...", "object": "...", "content": "..." }}
  ],

  // SECTION 2: PROMPT EVOLUTION (The Meta-Gradient)
  // Explain HOW the blamed agent's System Prompt should change to avoid this failure.
  // If Blame=QUESTIONER: Suggest specific types of questions (e.g., "Focus on implicit causality", "Ask relative time").
  // If Blame=BUILDER: Suggest extraction rules (e.g., "Capture adjectives", "Resolve 'he' to names").
  // If Blame=ANSWERER: Suggest search strategies (e.g., "Don't stop at 1 hop", "Trust graph over priors").
  "meta_gradient": "string description of the prompt update strategy"
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
            is_correct = result.get("is_correct")
            meta_gradient = result.get("meta_gradient")
            graph_patch = result.get("graph_patch")

            if not is_correct:
                logger.warning(f"❌ DEFENDER FAILED. Blame: {blame}")
                
                # Apply Policy Update to Defender
                if blame == "BUILDER":
                    self.builder.update_guideline("UPDATE", meta_gradient)
                    # Apply State Fix
                    if graph_patch:
                        patch_str = json.dumps(graph_patch)
                        self.builder.force_update(f"Apply these fixes: {patch_str}")
                        
                elif blame == "ANSWERER":
                    self.answerer.update_guideline("SEARCH", meta_gradient)
            else:
                logger.info(f"✅ DEFENDER SUCCEEDED. Optimizing Questioner...")
                # Apply Policy Update to Attacker
                if blame == "QUESTIONER":
                     self.questioner.update_guideline("GENERATE", meta_gradient)
            
            return result
                
        except Exception as e:
            logger.error(f"Optimizer Error: {e}")
            logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return {}
