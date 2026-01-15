import logging
import json
from typing import List, Dict, Any
from openai import OpenAI
from amadeus.code.agents.questioner import QuestionerAgent
from amadeus.code.agents.builder import BuilderAgent
from amadeus.code.agents.answerer import AnswererAgent

logger = logging.getLogger("Amadeus.Optimizer")

class AdversarialOptimizer:
    def __init__(self, questioner: QuestionerAgent, builder: BuilderAgent, answerer: AnswererAgent, model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
        self.questioner = questioner
        self.builder = builder
        self.answerer = answerer
        
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name

    def reset_steiner(self):
        """完全重置三个 Agent 的策略库，为下一个样本做准备"""
        logger.info("🧹 Clearing Reading Steiner protocols for all agents...")
        self.questioner.reset_steiner()
        self.builder.reset_steiner()
        self.answerer.reset_steiner()

    def _get_steiner_records(self, steiner_type: str) -> List[Dict[str, Any]]:
        """Helper to get records from the appropriate agent."""
        if steiner_type == "SERN_PROTOCOL":
            return self.questioner.reading_steiner.get("SERN_PROTOCOL", [])
        elif steiner_type == "PHONEWAVE_LOGIC":
            return self.builder.reading_steiner.get("PHONEWAVE_LOGIC", [])
        elif steiner_type == "SKULD_ORACLE":
            return self.answerer.reading_steiner.get("SKULD_ORACLE", [])
        return []

    def _format_steiner_records(self, steiner_type: str) -> str:
        """Formats the Reading Steiner records for the prompt ."""
        records = self._get_steiner_records(steiner_type)
        if not records:
            return "No established protocols yet."
        return "\n".join([f"[{r['id']}] helpful={r.get('help',0)} harmful={r.get('harm',0)} :: {r['content']}" for r in records])

    def step(self, buffer_content: str, action_log: List[str] = None, mode: str = "adaptive", fixed_loops: int = 3, use_cot: bool = False, builder_cot: str = "", builder_steiner_ids: List[str] = None):
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
                questions, q_cot, q_steiner_ids = self._generate_adaptive_attack(buffer_content, history)
            else:
                # Fixed Mode: 一次性生成 fixed_loops 个问题
                questions, q_cot, q_steiner_ids = self.questioner.generate_questions(buffer_content, num_questions=fixed_loops)
            
            if not questions:
                logger.info("🏳️ Questioner surrendered: No more meaningful questions to ask.")
                break

            # 2. Sanitization & Deduplication
            # Ensure all questions are strings and have required keys
            sanitized_questions = []
            for q in questions:
                if not isinstance(q, dict): continue
                q_text = q.get("question")
                if not q_text: continue
                if isinstance(q_text, dict):
                    q["question"] = json.dumps(q_text)
                elif not isinstance(q_text, str):
                    q["question"] = str(q_text)
                sanitized_questions.append(q)
            questions = sanitized_questions

            existing_qs = {h['question'] for h in history if isinstance(h.get('question'), str)}
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
                futures = {executor.submit(self._process_single_duel, q_item, buffer_content, action_log, use_cot, q_cot, builder_cot, q_steiner_ids, builder_steiner_ids): q_item for q_item in unique_questions}
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

    def _generate_adaptive_attack(self, buffer_content: str, history: List[Dict], fixed_count: int = None) -> (List[Dict], str, List[str]):
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
    "chain_of_thought": "...", // Analysis of current memory vulnerabilities
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
    "chain_of_thought": "...", // Analysis of current memory vulnerabilities
    "stop_attack": boolean, // Set true if no more valid questions exist
    "reason": "...",
    "questions": [ // Empty if stop_attack is true
        { "question": "...", "ground_truth": "...", "type": "detail/inference/negative" }
    ]
}
"""

        prompt = f"""You are the Red Team Leader (Attacker).
Target Memory Buffer: "{buffer_content[:500]}..."

**CURRENT SERN PROTOCOLS (Your evolving strategies):**
{self._format_steiner_records("SERN_PROTOCOL")}

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
            content = response.choices[0].message.content
            res = json.loads(content)
            cot = res.get("chain_of_thought", "No CoT provided.")
            used_ids = res.get("used_steiner_ids", [])
            
            # Only check stop_attack if NOT in fixed mode
            if not fixed_count and res.get("stop_attack", False):
                return [], cot, used_ids
            
            return res.get("questions", []), cot, used_ids
        except Exception as e:
            logger.error(f"Attack Generation Failed: {e}")
            return [], "", []

    def _process_single_duel(self, q_item, buffer_content, action_log, use_cot=False, q_cot="", b_cot="", q_sids=None, b_sids=None):
        question = q_item.get("question")
        # Ensure question is a string to avoid "unhashable dict" errors in deduplication
        if isinstance(question, dict):
            question = json.dumps(question)
        elif not isinstance(question, str):
            question = str(question)

        prediction, a_cot, a_sids = self.answerer.answer(question)
        
        if use_cot:
            eval_result = self._evaluate_and_update_cot(q_item, prediction, buffer_content, action_log, 
                                                       q_cot=q_cot, a_cot=a_cot, b_cot=b_cot,
                                                       q_sids=q_sids, a_sids=a_sids, b_sids=b_sids)
        else:
            eval_result = self._evaluate_and_update(q_item, prediction, buffer_content, action_log, q_cot=q_cot, b_cot=b_cot)
        
        return {
            "question": question,
            "result": "PASS" if eval_result and eval_result.get("is_correct") else "FAIL",
            "blame": eval_result.get("blame") if eval_result else "UNKNOWN"
        }

    def _get_global_steiner_stats(self) -> str:
        """Aggregates stats from all agents into a JSON string for the prompt."""
        all_stats = {
            "QUESTIONER": self.questioner.get_steiner_stats(),
            "BUILDER": self.builder.get_steiner_stats(),
            "ANSWERER": self.answerer.get_steiner_stats()
        }
        return json.dumps(all_stats, indent=2)

    def _evaluate_and_update_cot(self, q_item: Dict, prediction: str, buffer_content: str, action_log: List[str] = None, q_cot: str = "", a_cot: str = "", b_cot: str = "", q_sids=None, a_sids=None, b_sids=None):
        """
        Three-stage Evaluation & Evolution.
        Stage 1: Judge & Blame Allocation (Logic only)
        Stage 2: Reflector Diagnosis (Self-Correction & Tagging)
        Stage 3: Curator Evolution (Strategic Update)
        """
        action_log_str = "\n".join(action_log) if action_log else "No recent graph updates."
        buffer_snippet = buffer_content[:800].replace("\n", " ")
        
        try:
            # --- STAGE 1: JUDGE & BLAME ALLOCATION (Strict Evidence Audit) ---
            prompt_1 = f"""You are the High Judge of Amadeus. Your sole mission is to judge who should be blamed.

**EVIDENCE DATA:**
- Question: "{q_item['question']}"
- Ground Truth (The Target Fact): "{q_item['ground_truth']}"
- Predicted Answer: "{prediction}"
- [BUILDER_LOG] (What was actually stored in Graph):
{action_log_str}

**CRITICAL: GLOBAL vs LOCAL CONTEXT**
- **Ground Truth (GT)** is derived ONLY from the current Buffer.
- **Prediction** comes from the Global Memory Graph.
- **RULE**: If GT says "Unknown/Not mentioned" BUT Prediction gives a specific, plausible fact (likely from history), judge it as **CORRECT**.
  -> In this case, Blame QUESTIONER.

**BLAME LOGIC (Who failed?):**
IF The Prediction matches the Ground Truth: 
- **BLAME QUESTIONER**
IF The Prediction is WRONG:
Analyze the [Builder_Log] and the [Question]:
- **BLAME BUILDER IF**: The specific *relationship* or *attribute* needed to answer is ABSENT from the Log. (Creating a Node is not enough; the connection must exist).
- **BLAME ANSWERER IF**: The exact answer DOES appear in the Log (meaning it was just added), but the Answerer still hallucinated or said "Unknown".

**Example Output:**
{{
  "reasoning": "The Ground Truth asks for Evan's previous job, but the Predicted Answer incorrectly states his current job (Teacher). Looking at the [BUILDER_LOG], the Builder performed an UPDATE that overwrote the old occupation string with the new one. Since the historical state was deleted from the graph during the update, the Builder failed to maintain the temporal sequence required to answer this question.",
  "is_correct": false,
  "blame": "BUILDER"
}}

Output ONLY a valid JSON object:
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process]",
  "is_correct": boolean,
  "blame": "BUILDER" | "ANSWERER" | "QUESTIONER"
}}
"""
            res1 = self._call_llm(prompt_1)
            is_correct = res1.get("is_correct", False)
            blame = res1.get("blame", "NONE").upper()
            logger.info(f"⚖️ Stage 1 [Judge]: correct={is_correct}, blame={blame}, reason={res1.get('reasoning')}")
            
            # --- Convergence & Early Exit ---
            if is_correct:
                # If correct, we don't need to refine the defenders. 
                # We could refine the Questioner, but for now we skip to keep the loop efficient.
                return {
                    "is_correct": True,
                    "blame": blame,
                    "reasoning": res1.get("reasoning")
                }

            steiner_type_map = {
                "QUESTIONER": "SERN_PROTOCOL",
                "BUILDER": "PHONEWAVE_LOGIC",
                "ANSWERER": "SKULD_ORACLE"
            }
            
            if blame not in steiner_type_map:
                logger.warning(f"⚠️ Stage 1 returned invalid blame: {blame}. Skipping refinement.")
                return {
                    "is_correct": is_correct,
                    "blame": blame,
                    "reasoning": res1.get("reasoning")
                }

            target_type = steiner_type_map.get(blame)
            target_agent = getattr(self, blame.lower(), None)
            if not target_agent:
                logger.error(f"❌ Could not find agent for blame: {blame}")
                return { "is_correct": is_correct, "blame": blame }

            target_cot = q_cot if blame == "QUESTIONER" else (b_cot if blame == "BUILDER" else a_cot)
            target_used_ids = q_sids if blame == "QUESTIONER" else (b_sids if blame == "BUILDER" else a_sids)

            # --- STAGE 2: REFLECTOR DIAGNOSIS ---
            prompt_2 = f"""You are an Expert Analyst (Reflector). Diagnose why the [{blame}] agent failed.

**CONTEXT:**
- Question: "{q_item['question']}"
- Ground Truth: "{q_item['ground_truth']}"
- Prediction: "{prediction}"
- Buffer: "{buffer_snippet}"
- [{blame}_COT]: {target_cot}

**IDS REPORTED AS USED BY AGENT:**
{target_used_ids or "No IDs reported."}

**CURRENT {target_type} (steiner):**
{self._format_steiner_records(target_type)}

**INSTRUCTIONS:**
- Carefully analyze the agent's cot to identify where it went wrong
- Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the agent avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the agent should have done differently
- You will receive bulletpoints that are part of steiner that's used by the agent.
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the agent to do the correct action)

Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy, formula, or principle should be remembered to avoid this error?
  - steiner_tags: a list of json objects with bullet_id and tag for each bulletpoint used by the generator

**High-Quality Diagnosis Example:**
{{
  "reasoning": "The user asked about Evan's previous occupation before he became a teacher. The [BUILDER_LOG] shows that when the Builder processed the latest session, it performed an UPDATE that overwrote Evan's old job (Artist) with his new one. Because the historical data was deleted, the Answerer could only see his current role and failed to answer the historical question.",
  "error_identification": "Destructive update of state-changing information. The agent prioritized current-state accuracy over historical completeness.",
  "root_cause_analysis": "The agent followed a 'Latest-Truth' bias, assuming that new information should always replace old information in the graph to avoid conflicts, rather than realizing memory requires preserving states across a timeline.",
  "correct_approach": "Instead of deleting or overwriting the old node/edge, the Builder should have kept the old information, added a 'valid_until' or 'archived' timestamp, and then added the new fact as the current state.",
  "key_insight": "When encountering a conflict or an update to an existing fact, NEVER simply delete the old information. Instead, preserve it with a 'deprecated' or 'end_time' marker and add the new information as a parallel state. This ensures historical traceability needed for temporal or multi-hop questions.",
  "steiner_tags": []
}}
  
**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the agent have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
  "steiner_tags": [
    {{"id": "Q-001", "tag": "helpful"}},
    {{"id": "B-002", "tag": "harmful"}}
  ]
}}
"""
            res2 = self._call_llm(prompt_2)
            steiner_tags = res2.get("steiner_tags", [])
            logger.info(f"🔍 Stage 2 [Reflector]: Error={res2.get('error_identification')}, Insight={res2.get('key_insight')}")
            
            # Apply Feedback to used strategies using ID prefixes
            for tag_item in steiner_tags:
                if not isinstance(tag_item, dict): continue
                sid = tag_item.get("id")
                tag_value = tag_item.get("tag")
                if sid and tag_value in ["helpful", "harmful"]:
                    # Validation: Only apply feedback if the ID actually exists to prevent hallucinated tags
                    if sid.startswith("Q-"):
                        existing = [r['id'] for r in self.questioner.reading_steiner.get("SERN_PROTOCOL", [])]
                        if sid in existing:
                            logger.info(f"🏷️ Tagging {sid} as {tag_value}")
                            self.questioner.update_steiner("SERN_PROTOCOL", "FEEDBACK", sid, feedback=tag_value)
                    elif sid.startswith("B-"):
                        existing = [r['id'] for r in self.builder.reading_steiner.get("PHONEWAVE_LOGIC", [])]
                        if sid in existing:
                            logger.info(f"🏷️ Tagging {sid} as {tag_value}")
                            self.builder.update_steiner("PHONEWAVE_LOGIC", "FEEDBACK", sid, feedback=tag_value)
                    elif sid.startswith("A-"):
                        existing = [r['id'] for r in self.answerer.reading_steiner.get("SKULD_ORACLE", [])]
                        if sid in existing:
                            logger.info(f"🏷️ Tagging {sid} as {tag_value}")
                            self.answerer.update_steiner("SKULD_ORACLE", "FEEDBACK", sid, feedback=tag_value)

            # --- STAGE 3: CURATOR EVOLUTION ---
            prefix_map = {
                "SERN_PROTOCOL": "Q",
                "PHONEWAVE_LOGIC": "B",
                "SKULD_ORACLE": "A"
            }
            target_prefix = prefix_map.get(target_type, "ST")

            prompt_3 = f"""You are the Master Curator of Reading Steiner. Update the [{target_type}] based on Reflector feedback.

**REFLECTOR FEEDBACK:**
- Error Type: {res2.get('error_identification')}
- Root Cause: {res2.get('root_cause_analysis')}
- Insight: {res2.get('key_insight')}

**EXISTING steiner:**
{self._format_steiner_records(target_type)}

**Context:**
- The steiner you created will be used to help handle similar situations. 
- The reflection is generated using ground truth answers that will NOT be available when the steiner is being used. So you need to come up with content that can aid the steiner user to create predictions that likely align with ground truth. 

**CRITICAL: You MUST respond with valid JSON only.**

**Instructions:**
- Review the existing steiner and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current steiner
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing steiner
- Do NOT regenerate the entire steiner - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized steiner is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable

**High-Quality Update Example:**
{{
  "reasoning": "The reflector identified a critical 'Destructive Update' failure: the Builder overwrites old attributes (like job or location) with new ones, losing historical data. I will update protocol B-001 to mandate versioning/appending values instead of overwriting, and merge similar logic into a single 'Temporal Integrity' rule.",
  "operations": [
    {{
      "action": "UPDATE",
      "id": "B-001",
      "content": "When an attribute changes (e.g., job title, residence), do NOT overwrite. Append the new value to a list or use a 'current' vs 'previous' schema to preserve history."
    }},
    {{
      "action": "MERGE",
      "id": ["B-003", "B-007"],
      "content": "Temporal Logic: Always extract the 'valid_since' or 'relative_time' from the text to order state changes correctly in the graph."
    }}
  ]
}}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the steiner
- action: the type of operation to be performed
- id: used only for UPDATE and DELETE to identify the target bullet
- section: the section to add the bullet to
- content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.
2. UPDATE: Modify existing bullet points
    - id: the bullet_id to update
    - content: the updated content of the bullet
3. DELETE: Remove obsolete bullet points
    - id: the bullet_id to be removed
4. MERGE: Combine similar bullet points into one 
    - id: a list of bullet_ids to be merged (e.g. ["Q-003", "Q-005"])
    - content: the new merged content

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{{
  "operations": [
    {{
      "action": "ADD", 
      "section": "formulas_and_calculations",
      "content": "[New calculation method...]"
    }},
    {{
      "action": "UPDATE",
      "id": "{target_prefix}-001",
      "content": "[Updated content...]"
    }},
    {{
        "action": "DELETE",
        "id": "{target_prefix}-002"
    }},
    {{
        "action": "MERGE",
        "id": ["{target_prefix}-003", "{target_prefix}-005"],
        "content": "[Merged content from multiple similar bullets...]"
    }}
  ],
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]"
}}
"""
            res3 = self._call_llm(prompt_3)
            logger.info(f"🛠️ Stage 3 [Curator] reasoning: {res3.get('reasoning')}")
            
            for op in res3.get("operations", []):
                action = op.get("action")
                sid = op.get("id")
                content = op.get("content")
                
                if action in ["ADD", "UPDATE", "DELETE", "MERGE"]:
                    if action == "ADD" and content:
                        # Improved ID generation to avoid conflicts
                        existing_ids = []
                        for r in self._get_steiner_records(target_type):
                            try:
                                if '-' in r['id']:
                                    existing_ids.append(int(r['id'].split('-')[1]))
                            except (ValueError, IndexError):
                                continue
                        next_id = max(existing_ids) + 1 if existing_ids else 1
                        sid = f"{target_prefix}-{next_id:03d}"
                        logger.info(f"✨ Curator: ADD new strategy {sid}")
                    elif action == "UPDATE" and sid and content:
                        logger.info(f"🔄 Curator: UPDATE strategy {sid}")
                    elif action == "DELETE" and sid:
                        logger.info(f"🗑️ Curator: DELETE strategy {sid}")
                    elif action == "MERGE" and sid:
                        logger.info(f"🤝 Curator: MERGE strategies {sid}")
                    else:
                        continue
                        
                    target_agent.update_steiner(target_type, action, sid, content)

            # Optional: Apply State Fix if Builder failed
            graph_patch = []
            if blame == "BUILDER" and not is_correct:
                repair_prompt = f"""You are the Data Repair Agent.
The Builder failed to extract info for: "{q_item['question']}"
Buffer: "{buffer_snippet}"

Generate a JSON Graph Patch to fix this.
Output JSON: {{ "graph_patch": [ {{ "action": "ADD", "subject": "...", "object": "...", "content": "..." }} ] }}
"""
                res_patch = self._call_llm(repair_prompt)
                graph_patch = res_patch.get("graph_patch", [])
                if graph_patch:
                    logger.info(f"🩹 Applying Graph Patch: {len(graph_patch)} operations")
                    self.builder.force_update(json.dumps(graph_patch))

            return {
                "is_correct": is_correct,
                "blame": blame,
                "diagnosis": res2.get("key_insight"),
                "graph_patch": graph_patch
            }

        except Exception as e:
            logger.error(f"[CoT] Error: {e}")
            return {{}}

    def _safe_json_load(self, text: str) -> dict:
        if not text: return {}
        text = text.strip()
        
        # Pre-cleaning
        if text.startswith("```json"):
            text = text.replace("```json", "", 1).replace("```", "", -1).strip()
        elif text.startswith("```"):
            text = text.replace("```", "", 1).replace("```", "", -1).strip()

        try:
            return json.loads(text)
        except:
            # Attempt to find the outermost JSON object
            import re
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass

            logger.warning("Detected malformed JSON in Optimizer, attempting recovery...")
            
            # Robust Regex extract for operations array
            try:
                potential_objs = []
                # Match objects that contain typical optimizer keys
                obj_matches = re.finditer(r'\{\s*"(?:action|is_correct|blame|type)":\s*[^}]+\}', text)
                for m in obj_matches:
                    try:
                        obj_data = m.group(0)
                        if obj_data.count('{') > obj_data.count('}'):
                            obj_data += '}'
                        obj = json.loads(obj_data)
                        potential_objs.append(obj)
                    except:
                        continue
                
                if potential_objs:
                    # If we find blame/correctness, prioritize that
                    final_res = {}
                    ops = []
                    for obj in potential_objs:
                        if "is_correct" in obj: final_res["is_correct"] = obj["is_correct"]
                        if "blame" in obj: final_res["blame"] = obj["blame"]
                        if "action" in obj: ops.append(obj)
                    
                    if ops: final_res["operations"] = ops
                    if final_res: return final_res
            except:
                pass

            return {}

    def _call_llm(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return self._safe_json_load(response.choices[0].message.content)

    def _evaluate_and_update(self, q_item: Dict, prediction: str, buffer_content: str, action_log: List[str] = None, q_cot: str = "", b_cot: str = ""):
        # Critic LLM
        action_log_str = "\n".join(action_log) if action_log else "No recent graph updates."
        buffer_snippet = buffer_content[:500].replace("\n", " ")
        
        prompt = f"""You are the 'Meta-Critic' and 'Gradient Descent Optimizer' of the Amadeus Memory System.
Your goal: arbitrate the adversarial game between the [Questioner] (Attacker) and the [Builder/Answerer] (Defenders).

**CURRENT READING STEINER (Persistent Strategies):**
- [SERN Protocols]: {self._format_steiner_records("SERN_PROTOCOL")}
- [Phonewave Logic]: {self._format_steiner_records("PHONEWAVE_LOGIC")}
- [Skuld Oracle]: {self._format_steiner_records("SKULD_ORACLE")}

**READING STEINER STATS (Aligned with ACE):**
{self._get_global_steiner_stats()}

**GAME RULES (Zero-Sum):**
1. **Defenders Lose (Prediction WRONG)**: 
   - Identify WHY. Was the info missing (Builder fault) or not retrieved (Answerer fault)?
   - Generate a **Graph Patch** to fix the data immediately.
   - Generate a **Steiner Update** (ADD/UPDATE). Follow ACE principles: avoid redundancy, focus on root cause.
2. **Defenders Win (Prediction CORRECT)**:
   - The Questioner failed to trick the system.
   - Generate a **SERN Protocol Update** to ask harder questions.

**BLAME LOGIC (Strict Evidence Audit):**
Analyze the [Builder Log] vs the [Ground Truth]:
- **BLAME BUILDER IF**: The specific nodes, edges, or attributes needed to answer the question are MISSING or VAGUE in the [Builder Log]. (e.g., GT is a specific location but Log only says "traveling").
- **BLAME ANSWERER IF**: The required facts ARE clearly recorded in the [Builder Log], but the Answerer failed to find them, hallucinated, or said "Unknown".
- **BLAME QUESTIONER IF**: The prediction matches the ground truth.

**INPUT DATA:**
- Text Buffer: "{buffer_snippet}..."
- Question: "{q_item['question']}"
- Ground Truth: "{q_item['ground_truth']}"
- Prediction: "{prediction}"
- Questioner CoT: "{q_cot}"
- Builder CoT: "{b_cot}"
- Builder Log: "{action_log_str}"

**OUTPUT FORMAT (JSON):**
{{
  "chain_of_thought": "Step-by-step reasoning...",
  "is_correct": boolean,
  "blame": "BUILDER" | "ANSWERER" | "QUESTIONER",
  
  "graph_patch": [
      {{ "action": "ADD", "subject": "...", "object": "...", "content": "..." }}
  ],

  "steiner_update": {{
      "target": "SERN_PROTOCOL" | "PHONEWAVE_LOGIC" | "SKULD_ORACLE",
      "action": "ADD" | "UPDATE",
      "id": "Q-00X",
      "content": "A direct instruction (e.g. 'ALWAYS convert relative dates...')"
  }},
  "steiner_tags": [
      {{"id": "A-001", "tag": "helpful" | "harmful" | "neutral"}}
  ]
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
            
            if "chain_of_thought" in result:
                logger.info(f"Optimizer CoT: {result['chain_of_thought']}")

            blame = result.get("blame", "NONE").upper()
            is_correct = result.get("is_correct")
            steiner_update = result.get("steiner_update")
            graph_patch = result.get("graph_patch")
            steiner_tags = result.get("steiner_tags", [])

            # Apply Feedback to used IDs based on prefixes
            for tag_item in steiner_tags:
                if not isinstance(tag_item, dict): continue
                sid = tag_item.get("id")
                tag_value = tag_item.get("tag")
                if sid and tag_value in ["helpful", "harmful"]:
                    if sid.startswith("Q-"):
                        existing = [r['id'] for r in self.questioner.reading_steiner.get("SERN_PROTOCOL", [])]
                        if sid in existing:
                            self.questioner.update_steiner("SERN_PROTOCOL", "FEEDBACK", sid, feedback=tag_value)
                    elif sid.startswith("B-"):
                        existing = [r['id'] for r in self.builder.reading_steiner.get("PHONEWAVE_LOGIC", [])]
                        if sid in existing:
                            self.builder.update_steiner("PHONEWAVE_LOGIC", "FEEDBACK", sid, feedback=tag_value)
                    elif sid.startswith("A-"):
                        existing = [r['id'] for r in self.answerer.reading_steiner.get("SKULD_ORACLE", [])]
                        if sid in existing:
                            self.answerer.update_steiner("SKULD_ORACLE", "FEEDBACK", sid, feedback=tag_value)

            if not is_correct:
                logger.warning(f"❌ DEFENDER FAILED. Blame: {blame}")
                
                # Apply Policy Update to Defender
                if steiner_update:
                    target = steiner_update.get("target")
                    action = steiner_update.get("action", "ADD")
                    
                    # Prefix mapping for non-CoT mode
                    prefix_map = {"SERN_PROTOCOL": "Q", "PHONEWAVE_LOGIC": "B", "SKULD_ORACLE": "A"}
                    target_prefix = prefix_map.get(target, "ST")
                    
                    sid = steiner_update.get("id")
                    if action == "ADD" or not sid:
                        existing_ids = []
                        for r in self._get_steiner_records(target):
                            try:
                                if '-' in r['id']:
                                    existing_ids.append(int(r['id'].split('-')[1]))
                            except (ValueError, IndexError):
                                continue
                        next_id = max(existing_ids) + 1 if existing_ids else 1
                        sid = f"{target_prefix}-{next_id:03d}"
                        
                    content = steiner_update.get("content")
                    
                    if target == "PHONEWAVE_LOGIC":
                        self.builder.update_steiner(target, action, sid, content)
                    elif target == "SKULD_ORACLE":
                        self.answerer.update_steiner(target, action, sid, content)
                    elif target == "SERN_PROTOCOL":
                        self.questioner.update_steiner(target, action, sid, content)
                        
            else:
                logger.info(f"✅ DEFENDER SUCCEEDED. Optimizing Questioner...")
                # Questioner update in success case
                if steiner_update and steiner_update.get("target") == "SERN_PROTOCOL":
                    action = steiner_update.get("action", "ADD")
                    sid = steiner_update.get("id")
                    if action == "ADD" or not sid:
                        existing_ids = [int(r['id'].split('-')[1]) for r in self._get_steiner_records("SERN_PROTOCOL") if '-' in r['id']]
                        next_id = max(existing_ids) + 1 if existing_ids else 1
                        sid = f"Q-{next_id:03d}"
                    content = steiner_update.get("content")
                    self.questioner.update_steiner("SERN_PROTOCOL", action, sid, content)
            
            return result
                
        except Exception as e:
            logger.error(f"Optimizer Error: {e}")
            logger.error(f"Debug Info: Base URL: {self.client.base_url}, Model: {self.model_name}")
            return {}
