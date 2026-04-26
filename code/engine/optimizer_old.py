import logging
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from amadeus_collab.agents.questioner import QuestionerAgent
from amadeus_collab.agents.builder import BuilderAgent
from amadeus_collab.agents.answerer import AnswererAgent

logger = logging.getLogger("Amadeus.Optimizer")

class AdversarialOptimizer:
    def __init__(self, questioner: QuestionerAgent, builder: BuilderAgent, answerer: AnswererAgent, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
        self.questioner = questioner
        self.builder = builder
        self.answerer = answerer
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name

    def step(self, buffer_content: str, action_log: List[str] = None, mode: str = "adaptive", fixed_loops: int = 3, use_cot: bool = False):
        logger.info(f"⚔️ Starting Self-Play (Mode: {mode}, CoT: {use_cot})...")
        
        history = []
        iteration = 0
        # 熔断机制：防止无限烧钱，但上限设高一点
        # 如果是 fixed 模式，只运行 1 轮，一次性生成指定数量的问题
        HARD_LIMIT = 10 if mode == "adaptive" else 1
        
        # 初始状态：攻击者非常激进
        consecutive_wins = 0
        consecutive_useless_questions = 0
        
        while iteration < HARD_LIMIT:
            iteration += 1
            logger.info(f"--- Round {iteration} ---")

            # 1. 动态生成攻击 (Attack Generation)
            if mode == "adaptive":
                questions = self._generate_adaptive_attack(buffer_content, history)
            else:
                # Fixed Mode: 一次性生成 fixed_loops 个问题
                questions = self.questioner.generate_questions(buffer_content, num_questions=fixed_loops)
            
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
                futures = {executor.submit(self._process_single_duel, q_item, buffer_content, action_log, use_cot): q_item for q_item in unique_questions}
                for future in concurrent.futures.as_completed(futures):
                    batch_results.append(future.result())

            # 4. 状态更新与收敛检查 (State Update & Convergence)
            round_failed = False
            for res in batch_results:
                history.append(res)
                if res['result'] == "FAIL":
                    round_failed = True
            
            if mode == "adaptive":
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

    def _generate_adaptive_attack(self, buffer_content: str, history: List[Dict], fixed_count: int = None) -> List[Dict]:
        """
        让 Questioner 观察历史，决定是否继续攻击，以及攻击什么。
        """
        # 简化的历史摘要
        history_summary = "\n".join([f"Q: {h['question']} -> {'✅ PASS' if h['result']=='PASS' else '❌ FAIL'}" for h in history[-10:]])
        
        if fixed_count:
            mission_prompt = f"""**YOUR MISSION:**
Generate exactly {fixed_count} challenging questions based on the Target Memory Buffer.
Do NOT stop. You must generate {fixed_count} questions.
"""
            output_format = """**OUTPUT FORMAT (JSON):**
{
    "questions": [
        { "question": "...", "ground_truth": "...", "type": "detail/inference/negative" }
    ]
}
"""
        else:
            mission_prompt = """**YOUR MISSION:**
Determine if there are still unexplored vulnerabilities or missing details in the memory.
- If the Defender failed recently: ATTACK HARDER on that specific topic.
- If the Defender passed: Try a TRICKIER angle or a different detail.
- If the buffer is fully covered and robust: STOP.
"""
            output_format = """**OUTPUT FORMAT (JSON):**
{
    "stop_attack": boolean, // Set true if no more valid questions exist
    "reason": "...",
    "questions": [ // Empty if stop_attack is true
        { "question": "...", "ground_truth": "...", "type": "detail/inference/negative" }
    ]
}
"""

        prompt = f"""You are the Red Team Leader (Attacker).
Target Memory Buffer: "{buffer_content[:500]}..."

Previous Attacks & Results:
{history_summary}

{mission_prompt}

{output_format}
"""
        try:
            response = self.questioner.client.chat.completions.create(
                model=self.questioner.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0 # 保持一定的创造性
            )
            res = self._parse_json_content(response.choices[0].message.content)
            
            # Only check stop_attack if NOT in fixed mode
            if not fixed_count and res.get("stop_attack", False):
                return []
            
            return res.get("questions", [])
        except Exception as e:
            logger.error(f"Attack Generation Failed: {e}")
            return []

    def _process_single_duel(self, q_item, buffer_content, action_log, use_cot=False):
        question = q_item.get("question")
        prediction = self.answerer.answer(question)
        
        if use_cot:
            eval_result = self._evaluate_and_update_cot(q_item, prediction, buffer_content, action_log)
        else:
            eval_result = self._evaluate_and_update(q_item, prediction, buffer_content, action_log)
        
        return {
            "question": question,
            "result": "PASS" if eval_result and eval_result.get("is_correct") else "FAIL",
            "blame": eval_result.get("blame") if eval_result else "UNKNOWN"
        }

    def _evaluate_and_update_cot(self, q_item: Dict, prediction: str, buffer_content: str, action_log: List[str] = None):
        """
        Chain-of-Thought Evaluation: Split the complex task into 3 smaller steps.
        """
        action_log_str = "\n".join(action_log) if action_log else "No recent graph updates."
        buffer_snippet = buffer_content[:500].replace("\n", " ")
        
        # Step 1: Judge & Blame
        prompt_1 = f"""You are the Judge of the Amadeus Memory System.
Goal: Determine if the Prediction matches the Ground Truth (derived from Buffer).

Buffer: "{buffer_snippet}..."
Question: "{q_item['question']}"
Ground Truth: "{q_item['ground_truth']}"
Prediction: "{prediction}"
Builder Log: "{action_log_str}"

**RULES:**
1. If Prediction matches Ground Truth -> CORRECT.
2. If Prediction is plausible but not in Buffer -> CORRECT (Blame Questioner).
3. If Prediction contradicts Buffer -> WRONG.

**BLAME (if WRONG):**
- BUILDER: Info missing from Builder Log.
- ANSWERER: Info exists in Log but Answerer missed it.

**CHAIN OF THOUGHT:**
Think step-by-step:
1. Compare Prediction vs Ground Truth.
2. IF Prediction is CORRECT:
   - Blame QUESTIONER. Turn to Step 4.
3. IF Prediction is WRONG:
   - Check Builder Log.
   - If Info missing -> Blame BUILDER.
   - If Info exists -> Blame ANSWERER.
4. Analyze WHY the failure happened and give high-level guidelines.
Output JSON: {{ "chain_of_thought": "...", "is_correct": boolean, "blame": "BUILDER" | "ANSWERER" | "QUESTIONER" | "NONE", "reason": "..." }}
"""
        try:
            res1 = self._call_llm(prompt_1)
            if "chain_of_thought" in res1:
                logger.info(f"Optimizer (Judge) CoT: {res1['chain_of_thought']}")
            is_correct = res1.get("is_correct", False)
            blame = res1.get("blame", "NONE")
            
            graph_patch = []
            meta_gradient = ""

            if not is_correct:
                logger.warning(f"❌ [CoT] DEFENDER FAILED. Blame: {blame}")
                
                # Step 2: Patch (Only if Builder failed)
                if blame == "BUILDER":
                    prompt_2 = f"""You are the Data Repair Agent.
The Builder failed to extract info for: "{q_item['question']}"
Buffer: "{buffer_snippet}..."

Generate a JSON Graph Patch to fix this.
Output JSON: {{ "graph_patch": [ {{ "action": "ADD", "subject": "...", "object": "...", "content": "..." }} ] }}
"""
                    res2 = self._call_llm(prompt_2)
                    graph_patch = res2.get("graph_patch", [])
                    if graph_patch:
                        patch_str = json.dumps(graph_patch)
                        self.builder.force_update(f"Apply these fixes: {patch_str}")

                # Step 3: Gradient (For the blamed agent)
                prompt_3 = f"""You are the Optimization Coach.
The agent '{blame}' failed because: {res1.get('reason')}
Question: "{q_item['question']}"

Determine which operator needs improvement:
- BUILDER: ADD, UPDATE, DELETE, WAIT
- ANSWERER: SEARCH, WALK, READ
- QUESTIONER: GENERATE

Suggest a short, actionable instruction (Meta-Gradient) to update the agent's system prompt to prevent this.
Output JSON: {{ "target_operator": "...", "meta_gradient": "..." }}
"""
                res3 = self._call_llm(prompt_3)
                meta_gradient = res3.get("meta_gradient", "")
                target_operator = res3.get("target_operator")
                
                if blame == "BUILDER":
                    self.builder.update_guideline(target_operator, meta_gradient)
                elif blame == "ANSWERER":
                    self.answerer.update_guideline(target_operator, meta_gradient)
            
            else:
                logger.info(f"✅ [CoT] DEFENDER SUCCEEDED. Optimizing Questioner...")
                # Step 3 (Alt): Gradient for Questioner
                if blame == "QUESTIONER":
                    prompt_3 = f"""You are the Red Team Coach.
The Questioner failed to trick the system.
Question: "{q_item['question']}"

Suggest a strategy to generate harder/trickier questions.
Output JSON: {{ "meta_gradient": "..." }}
"""
                    res3 = self._call_llm(prompt_3)
                    meta_gradient = res3.get("meta_gradient", "")
                    self.questioner.update_guideline("GENERATE", meta_gradient)

            return {
                "is_correct": is_correct,
                "blame": blame,
                "graph_patch": graph_patch,
                "meta_gradient": meta_gradient
            }

        except Exception as e:
            logger.error(f"[CoT] Error: {e}")
            return {}

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
            markers = ["\n\n{", "\n\n[", "\n\nFinal Answer", "\n\nAnswer:"]
            for marker in markers:
                idx = text.find(marker)
                if idx != -1:
                    tail = text[idx + 2 :].strip()
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

    def _call_llm(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return self._parse_json_content(response.choices[0].message.content)

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
   - Generate a **Textual Gradient** to force the Questioner to ask trickier and more discriminative questions next time.

**CRITICAL: GLOBAL vs LOCAL CONTEXT**
- **Ground Truth (GT)** is derived ONLY from the current Buffer.
- **Prediction** comes from the Global Memory Graph.
- **RULE**: If GT says "Unknown/Not mentioned" BUT Prediction gives a specific, plausible fact (likely from history), judge it as **CORRECT**.
  -> In this case, Blame QUESTIONER for asking about old history instead of current events.

**BLAME LOGIC (Who failed?):**
Analyze the [Builder Activity Log] and the [Question]:
- **BLAME BUILDER IF**: The specific *relationship* or *attribute* needed to answer is ABSENT from the Log. (Creating a Node is not enough; the connection must exist).
- **BLAME ANSWERER IF**: The exact answer DOES appear in the Log (meaning it was just added), but the Answerer still hallucinated or said "Unknown".

**CHAIN OF THOUGHT:**
Before generating the final JSON, you must perform a step-by-step analysis:
1. **Compare Prediction vs Ground Truth**: Is it correct? Is it plausible?
2. **IF Prediction is CORRECT**:
   - Blame QUESTIONER. Turn to Step 4.
3. **IF Prediction is WRONG**:
   - **Analyze Causality**: Look at the Builder Log. Was the info captured? If yes, why did Answerer miss it? If no, why did Builder miss it?
4. **Formulate Strategy**: Based on the blame, what high-level instruction (Gradient) would improve this in the future?

**INPUT DATA:**
- Text Buffer: "{buffer_snippet}..."
- Question: "{q_item['question']}"
- Ground Truth: "{q_item['ground_truth']}"
- Prediction: "{prediction}"
- Builder Log: "{action_log_str}"

**OUTPUT FORMAT (JSON):**
{{
  "chain_of_thought": "Step 1: Comparing... Step 2/3: Blaming... Step 4: Strategy...",
  "is_correct": boolean,
  "blame": "BUILDER" | "ANSWERER" | "QUESTIONER",
  
  // SECTION 1: DATA REPAIR (Only if Prediction is WRONG and Blame is BUILDER)
  // Generate concrete operations to fix the graph NOW.
  "graph_patch": [
      {{ "action": "ADD", "subject": "...", "object": "...", "content": "..." }}
  ],

  // SECTION 2: PROMPT EVOLUTION (The Meta-Gradient)
  // Determine which operator needs improvement:
  // - BUILDER: ADD (if info missed), UPDATE (if info wrong), DELETE, WAIT
  // - ANSWERER: SEARCH, WALK, READ
  // - QUESTIONER: GENERATE
  "target_operator": "The operator responsible. Valid values: ADD, UPDATE, DELETE, WAIT, SEARCH, WALK, READ, GENERATE",

  // Explain HOW the blamed agent's System Prompt should change to avoid this failure.
  // If Blame=QUESTIONER: Suggest how to ask trickier and more discriminative questions.
  // If Blame=BUILDER: Suggest how to improve memory graph management.
  // If Blame=ANSWERER: Suggest how to retrieve information more effectively and accurately, and construct the information to answer better.
  // CRITICAL: Write this as a DIRECT INSTRUCTION or RULE for the Agent.
  // BAD: "The Builder should ensure..."
  // GOOD: "ALWAYS convert relative dates..."
  "meta_gradient": "string description of the prompt update strategy"
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            result = self._parse_json_content(response.choices[0].message.content)
            
            if "chain_of_thought" in result:
                logger.info(f"Optimizer CoT: {result['chain_of_thought']}")

            blame = result.get("blame")
            is_correct = result.get("is_correct")
            meta_gradient = result.get("meta_gradient")
            graph_patch = result.get("graph_patch")
            target_operator = result.get("target_operator")

            if not is_correct:
                logger.warning(f"❌ DEFENDER FAILED. Blame: {blame}")
                
                # Apply Policy Update to Defender
                if blame == "BUILDER":
                    op = target_operator
                    self.builder.update_guideline(op, meta_gradient)
                    # Apply State Fix
                    if graph_patch:
                        patch_str = json.dumps(graph_patch)
                        self.builder.force_update(f"Apply these fixes: {patch_str}")
                        
                elif blame == "ANSWERER":
                    op = target_operator
                    self.answerer.update_guideline(op, meta_gradient)
            else:
                logger.info(f"✅ DEFENDER SUCCEEDED. Optimizing Questioner...")
                # Apply Policy Update to Attacker
                if blame == "QUESTIONER":
                     op = "GENERATE"
                     self.questioner.update_guideline(op, meta_gradient)
            
            return result
                
        except Exception as e:
            logger.error(f"Optimizer Error: {e}")
            logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return {}
