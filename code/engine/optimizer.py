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
    def __init__(self, questioner: QuestionerAgent, builder: BuilderAgent, answerer: AnswererAgent,
                 model_name: str = "gpt-4-turbo", api_base: str = None, api_key: str = None):
        self.questioner = questioner
        self.builder = builder
        self.answerer = answerer
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
        self.experiences: List[Dict] = []  # 元优化经验: [{"trigger": ..., "measure": ..., "target_agent": ..., "target_operator": ...}, ...]
        self.usage_stats = {
            "api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # ------------------------------------------------------------------ #
    #                          PUBLIC ENTRY POINT                         #
    # ------------------------------------------------------------------ #
    def step(self, buffer_content: str, action_log: List[str] = None,
             mode: str = "fixed", fixed_loops: int = 3, use_cot: bool = False):
        """
        新自博弈逻辑:
        1. 出 1 道题
        2. Answerer 答题 → Judge 评判
           若错 → Optimizer 生成策略更新 → 应用 → Builder 重建图 → 重试 (最多 MAX_RETRIES 次)
        3. 记录全部尝试
        4. 若同时出现 FAIL 和 PASS → CoT 对比总结经验
        """
        MAX_RETRIES = 3

        logger.info(f"⚔️ Self-Play Start | Mode: iterative-retry, MaxRetries: {MAX_RETRIES}")

        # ---------- 1. 生成 1 道题 ----------
        questions = self.questioner.generate_questions(buffer_content, num_questions=1)
        if not questions:
            logger.info("🏳️ Questioner generated no questions, skipping self-play.")
            return

        q_item = questions[0]
        question = q_item.get("question", "")
        ground_truth = q_item.get("ground_truth", "")
        logger.info(f"⚔️ Question: {question}")
        logger.info(f"⚔️ Ground Truth: {ground_truth}")

        # ---------- 2. 迭代重试循环 ----------
        attempts: List[Dict] = []

        for attempt_idx in range(MAX_RETRIES):
            logger.info(f"🔄 Attempt {attempt_idx + 1}/{MAX_RETRIES}")

            # Answerer 答题
            prediction = self.answerer.answer(question)
            logger.info(f"🔄 Attempt {attempt_idx + 1} | Prediction: {prediction[:200]}")

            # Judge 评判 + 策略建议
            eval_result = self._evaluate_attempt(
                q_item, prediction, buffer_content, action_log, attempt_idx, attempts
            )

            is_correct = eval_result.get("is_correct", False)
            blame = eval_result.get("blame", "UNKNOWN")
            error_category = eval_result.get("error_category", "")
            reason = eval_result.get("reason", "")

            # 构建本次记录
            record = {
                "attempt": attempt_idx + 1,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "is_correct": is_correct,
                "blame": blame,
                "error_category": error_category,
                "reason": reason,
                "strategy_update": eval_result.get("strategy_update"),
                "result": "PASS" if is_correct else "FAIL",
            }
            attempts.append(record)

            if is_correct:
                logger.info(f"✅ Attempt {attempt_idx + 1} PASS | Blame: {blame}")
                break

            logger.warning(f"❌ Attempt {attempt_idx + 1} FAIL | Blame: {blame} | ErrorCat: {error_category} | Reason: {reason}")

            # 如果还有重试机会，应用策略更新并重建图
            if attempt_idx < MAX_RETRIES - 1:
                strategy = eval_result.get("strategy_update")
                if strategy:
                    self._apply_strategy_update(strategy, blame)
                    # Builder 用新策略重新处理 buffer，重建图
                    logger.info("🔨 Builder re-processing buffer with updated strategy...")
                    self.builder.process_buffer(buffer_content)

        # ---------- 3. 检查是否提炼经验 ----------
        has_fail = any(a["result"] == "FAIL" for a in attempts)
        has_success = any(a["result"] == "PASS" for a in attempts)
        final_result = attempts[-1]["result"]

        if has_fail and has_success:
            logger.info("🧠 Both FAIL and PASS detected — extracting experience...")
            experience = self._extract_experience(attempts)
            if experience:
                self.experiences.append(experience)
                logger.info(f"🧠 Experience Extracted!")
                logger.info(f"🧠 TARGET: {experience.get('target_agent', '?')}.{experience.get('target_operator', '?')}")
                logger.info(f"🧠 TRIGGER: {experience['trigger']}")
                logger.info(f"🧠 MEASURE: {experience['measure']}")

        logger.info(f"⚔️ Self-Play End | Attempts: {len(attempts)} | Final: {final_result} | Experiences Total: {len(self.experiences)}")

    # ------------------------------------------------------------------ #
    #                      EVALUATE A SINGLE ATTEMPT                      #
    # ------------------------------------------------------------------ #
    def _evaluate_attempt(self, q_item: Dict, prediction: str, buffer_content: str,
                          action_log: List[str], attempt_idx: int,
                          previous_attempts: List[Dict]) -> Dict:
        """判断对错 + 归因 + 生成策略更新建议"""
        MAX_RETRIES = 3
        action_log_str = "\n".join(action_log) if action_log else "No recent graph updates."
        buffer_snippet = buffer_content[:800].replace("\n", " ")

        # 前几次尝试的上下文
        prev_section = self._format_previous_attempts(previous_attempts)
        # 已有经验
        exp_section = self._format_experiences()

        prompt = f"""You are the Judge and Strategy Advisor of the Amadeus Memory System.

## Task
1. Evaluate whether the Prediction correctly answers the Question.
2. If incorrect, diagnose the root cause and propose a STRATEGY-LEVEL update.

## Input
- Buffer: "{buffer_snippet}..."
- Question: "{q_item['question']}"
- Ground Truth: "{q_item['ground_truth']}"
- Prediction: "{prediction}"
- Builder Activity Log: "{action_log_str}"
- Attempt: {attempt_idx + 1} / {MAX_RETRIES}

{prev_section}

{exp_section}

## Evaluation Rules
1. Prediction semantically matches Ground Truth → CORRECT
2. Prediction gives a plausible fact not contradicted by Buffer → CORRECT (Blame: QUESTIONER)
3. Prediction contradicts or misses key info from Buffer → WRONG

## Blame Logic (if WRONG)
- BUILDER: The needed fact/relationship is ABSENT from Builder Log (info was never extracted)
- ANSWERER: The fact EXISTS in Builder Log / Graph but Answerer failed to find or use it

## Strategy Update Requirements (CRITICAL)
Your strategy update (meta_gradient) must be a **procedural rule about HOW to process information**, not about specific facts.

**BAD examples (too vague — NEVER write rules like these):**
- "Be more careful when extracting information"
- "Pay more attention to details"
- "Improve retrieval accuracy"

**BAD examples (too fact-specific — these are just memorizing answers):**
- "Remember that the visitor arrived at the station"
- "The answer to questions about the device status is offline"
- "Store that the package was delivered on Tuesday"

**GOOD examples (procedural, reusable — aim for this level):**
- "When an entity changes state or location, remove or overwrite the outdated state and keep the most recent one"
- "When searching for temporal questions (when/what date), prioritize edges with timestamp fields over node descriptions"
- "When the buffer contains third-person references (he/she/they), resolve the pronoun to an entity name BEFORE creating any edge"

The rule must be: someone encountering a SIMILAR PATTERN in the future can follow this instruction alone to avoid the same class of error.

## Chain of Thought
1. Compare Prediction vs Ground Truth — is it correct?
2. If WRONG: Examine Builder Log — was the info captured? → Assign blame
3. What CATEGORY of processing error is this? (e.g., missing_temporal_link, unresolved_coreference, shallow_search, missing_causal_relation)
4. What procedural rule would prevent this CATEGORY of errors?

## Output (JSON)
{{
  "chain_of_thought": "Step 1: ... Step 2: ... Step 3: ... Step 4: ...",
  "is_correct": boolean,
  "blame": "BUILDER" | "ANSWERER" | "QUESTIONER",
  "error_category": "a short snake_case label for the error type",
  "reason": "one-sentence diagnosis",

  "strategy_update": {{
    "target_agent": "BUILDER" | "ANSWERER",
    "target_operator": "ADD|UPDATE|DELETE|WAIT|SEARCH|WALK|READ",
    "meta_gradient": "A procedural rule (see requirements above)",
    "graph_patch": [
      {{ "action": "ADD|UPDATE|DELETE", "subject": "...", "object": "...", "content": "..." }}
    ]
  }}
}}

Notes:
- "strategy_update" is REQUIRED when is_correct == false.
- "graph_patch" inside strategy_update is only needed when blame == BUILDER.
- When is_correct == true, you may omit "strategy_update" or set it to null.
"""
        try:
            result = self._call_llm(prompt)
            if "chain_of_thought" in result:
                logger.info(f"📋 Judge CoT: {result['chain_of_thought']}")
            return result
        except Exception as e:
            logger.error(f"Evaluate attempt failed: {e}")
            return {"is_correct": False, "blame": "UNKNOWN", "error_category": "llm_error", "reason": str(e)}

    # ------------------------------------------------------------------ #
    #                      APPLY STRATEGY UPDATE                          #
    # ------------------------------------------------------------------ #
    def _apply_strategy_update(self, strategy: Dict, blame: str):
        """将策略更新应用到对应的 agent，并记录日志"""
        if not strategy:
            return

        target_agent = strategy.get("target_agent", blame)
        target_operator = strategy.get("target_operator", "ADD")
        meta_gradient = strategy.get("meta_gradient", "")
        graph_patch = strategy.get("graph_patch")

        if not meta_gradient:
            return

        logger.info(f"📈 Strategy Update | Agent: {target_agent} | Operator: {target_operator}")
        logger.info(f"📈 Meta-Gradient: {meta_gradient}")

        if blame == "BUILDER" or target_agent == "BUILDER":
            self.builder.update_guideline(target_operator, meta_gradient)
            if graph_patch:
                logger.info(f"🔧 Graph Patch: {json.dumps(graph_patch, ensure_ascii=False)}")
                self.builder.force_update(f"Apply these fixes: {json.dumps(graph_patch)}")
        elif blame == "ANSWERER" or target_agent == "ANSWERER":
            self.answerer.update_guideline(target_operator, meta_gradient)

    # ------------------------------------------------------------------ #
    #                      EXTRACT EXPERIENCE (CoT)                       #
    # ------------------------------------------------------------------ #
    def _extract_experience(self, attempts: List[Dict]) -> Dict:
        """对比成功和失败的策略更新，CoT 提炼元优化经验（教 optimizer 怎么更新 builder/answerer）"""
        formatted = self._format_attempts_for_experience(attempts)

        prompt = f"""You are the Meta-Optimization Coach of the Amadeus Memory System.

## Your Role
You are NOT writing rules for Builder or Answerer to follow directly.
You are writing rules for the **Optimizer** — teaching it HOW to diagnose errors and WHAT KIND of strategy updates to generate for Builder/Answerer in future self-play rounds.

## Background: How the System Works
- **Builder**: Processes conversation buffer → extracts entities and relationships → builds a memory graph (nodes + edges)
- **Answerer**: Receives a question → searches/walks the graph → reads node content → generates an answer
- **Optimizer** (you are teaching this): When Answerer gets a question wrong, the Optimizer must:
  1. Diagnose the error type and assign blame (BUILDER or ANSWERER)
  2. Generate a strategy update (meta_gradient) targeting the right agent and operator
  3. Optionally generate a graph_patch to fix the immediate data gap
- Your experience will be injected into the Optimizer's prompt in future rounds to help it generate BETTER strategy updates.

## Attempt History
{formatted}

## Chain of Thought (Follow these steps strictly)

Step 1 — **Locate the Turning Point**:
Which strategy update turned failure into success? Quote the meta_gradient, its target_agent, and target_operator.

Step 2 — **Diagnose Why Earlier Updates Failed**:
Did earlier updates target the wrong agent? Wrong operator? Were they too vague to be actionable? Did they address symptoms instead of root cause?

Step 3 — **What Made the Successful Update Work**:
What did the winning update get right that the others missed? Focus on:
- Was it targeting the correct agent (Builder vs Answerer)?
- Was it targeting the correct operator (ADD vs SEARCH vs READ etc.)?
- Was the meta_gradient specific enough to change behavior?

Step 4 — **Generalize into an Optimizer Heuristic**:
Abstract this into a rule that tells the Optimizer: "When you see error pattern X, you should update agent Y's operator Z with a meta_gradient that does W."
Replace specific entities/facts with categories (e.g., "Taylor" → "a participant", "2024-03-02" → "an absolute date", "last Friday" → "a relative time expression", "station" → "a location entity").

Step 5 — **Formulate the Experience**:
Write the final experience with these fields:
- trigger: The error pattern the Optimizer should recognize (based on error_category, blame, question type, or graph state)
- measure: What the Optimizer should DO — which agent to target, which operator, and what kind of meta_gradient to write
- target_agent: BUILDER or ANSWERER (who should the Optimizer update)
- target_operator: The operator to focus the update on

## CRITICAL: This is a Meta-Level Rule
**BAD (operational rule for Builder/Answerer directly):**
- trigger: "When temporal references appear in the buffer"
- measure: "Anchor relative times to the conversation date"
→ This tells Builder what to do. But the Optimizer already knows the task — it needs to know WHEN and HOW to generate such an update.

**GOOD (meta-optimization rule for the Optimizer):**
- trigger: "When error_category is 'missing_temporal_link' and blame is BUILDER, indicating the graph lacks temporal anchoring for relative time expressions"
- measure: "Update Builder's ADD operator with a meta_gradient requiring conversion of relative temporal references (e.g., 'yesterday', 'last week') to absolute dates using the session date as anchor. Also generate a graph_patch that adds the correctly anchored temporal edge."
- target_agent: "BUILDER"
- target_operator: "ADD"

## Output (JSON)
{{
  "chain_of_thought": "Step 1: ... Step 2: ... Step 3: ... Step 4: ... Step 5: ...",
  "experience": {{
    "trigger": "When the Optimizer observes [error pattern / blame / error_category]",
    "measure": "The Optimizer should update [AGENT]'s [OPERATOR] operator with a meta_gradient that [specific type of procedural instruction to generate]",
    "target_agent": "BUILDER" | "ANSWERER",
    "target_operator": "ADD|UPDATE|DELETE|SEARCH|WALK|READ"
  }}
}}
"""
        try:
            result = self._call_llm(prompt)
            if "chain_of_thought" in result:
                logger.info(f"🧠 Experience CoT: {result['chain_of_thought']}")
            exp = result.get("experience")
            if exp and exp.get("trigger") and exp.get("measure"):
                exp.setdefault("target_agent", "UNKNOWN")
                exp.setdefault("target_operator", "UNKNOWN")
                return exp
            return None
        except Exception as e:
            logger.error(f"Experience extraction failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #                         FORMAT HELPERS                               #
    # ------------------------------------------------------------------ #
    def _format_previous_attempts(self, attempts: List[Dict]) -> str:
        """格式化前几次尝试的上下文，注入到评判 prompt 中"""
        if not attempts:
            return ""

        lines = ["## Previous Attempts (This Round)",
                 "The system has already tried and failed. Learn from previous mistakes — do NOT repeat the same type of strategy update.\n"]
        for a in attempts:
            mg = ""
            if a.get("strategy_update") and a["strategy_update"].get("meta_gradient"):
                mg = a["strategy_update"]["meta_gradient"]
            lines.append(
                f"- Attempt {a['attempt']}: Prediction=\"{a['prediction'][:100]}\" | "
                f"Result={a['result']} | Blame={a['blame']} | ErrorCat={a.get('error_category','')}\n"
                f"  Strategy Update Applied: \"{mg}\"\n"
                f"  Why insufficient: the next attempt still failed after this update"
            )
        return "\n".join(lines)

    def _format_experiences(self) -> str:
        """格式化已有元优化经验列表，注入到评判 prompt 中"""
        if not self.experiences:
            return ""

        lines = ["## Accumulated Meta-Optimization Experiences",
                 "These are proven heuristics from past self-play rounds. When you observe the trigger pattern, "
                 "follow the measure to generate a better strategy update.\n"]
        for i, exp in enumerate(self.experiences, 1):
            target = f"{exp.get('target_agent', '?')}.{exp.get('target_operator', '?')}"
            lines.append(f"- Experience {i} [Target: {target}]:")
            lines.append(f"  RECOGNIZE: {exp['trigger']}")
            lines.append(f"  THEN UPDATE: {exp['measure']}")
        return "\n".join(lines)

    def _format_attempts_for_experience(self, attempts: List[Dict]) -> str:
        """格式化全部尝试记录，供经验提炼 prompt 使用"""
        lines = []
        for a in attempts:
            mg = ""
            target_agent = ""
            target_op = ""
            if a.get("strategy_update"):
                su = a["strategy_update"]
                mg = su.get("meta_gradient", "")
                target_agent = su.get("target_agent", "")
                target_op = su.get("target_operator", "")
            lines.append(
                f"Attempt {a['attempt']} [{a['result']}]:\n"
                f"  Question: \"{a['question']}\"\n"
                f"  Ground Truth: \"{a['ground_truth']}\"\n"
                f"  Prediction: \"{a['prediction'][:200]}\"\n"
                f"  Blame: {a['blame']} | Error Category: {a.get('error_category','')}\n"
                f"  Strategy Update Target: {target_agent}.{target_op}\n"
                f"  Strategy Update (meta_gradient): \"{mg}\""
            )
        return "\n\n".join(lines)

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

    # ------------------------------------------------------------------ #
    #                           LLM CALL                                   #
    # ------------------------------------------------------------------ #
    def _call_llm(self, prompt: str) -> Dict:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        self._record_usage(response)
        return self._parse_json_content(response.choices[0].message.content)
