"""
Data converter for BABILong to LightMemory format.

Transforms BABILong context (facts) into LightMemory-compatible format.
Does NOT modify LightMemory's core mechanisms.
"""

import re
from typing import List, Dict, Any


class DataConverter:
    """Convert BABILong format to LightMemory format."""

    def __init__(self, max_context_length: int = 4000):
        """
        Initialize converter.

        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length

    def parse_babilong_context(self, context: str) -> List[Dict[str, Any]]:
        """
        Parse BABILong context into structured facts.

        BABILong context is a sequence of facts separated by periods.
        Each fact is a sentence describing an event or state.

        Args:
            context: Raw BABILong context string

        Returns:
            List of fact dictionaries with metadata
        """
        # Split by periods to get individual facts
        sentences = context.replace("\n", " ").split(". ")

        facts = []
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Ensure sentence ends with period
            if not sentence.endswith("."):
                sentence = sentence + "."

            fact = {
                "id": f"fact_{idx}",
                "content": sentence,
                "sequence": idx,
                "type": self._classify_fact(sentence)
            }
            facts.append(fact)

        return facts

    def _classify_fact(self, sentence: str) -> str:
        """
        Classify fact type based on content.

        Args:
            sentence: Fact sentence

        Returns:
            Fact type: "location", "possession", "action", "state", "other"
        """
        sentence_lower = sentence.lower()

        # Location facts
        if any(word in sentence_lower for word in ["went", "moved", "travelled", "is in", "is at"]):
            return "location"

        # Possession facts
        if any(word in sentence_lower for word in ["got", "picked", "dropped", "has", "have"]):
            return "possession"

        # Action facts
        if any(word in sentence_lower for word in ["gave", "took", "passed", "handed"]):
            return "action"

        # State facts
        if any(word in sentence_lower for word in ["is", "are", "was", "were"]):
            return "state"

        return "other"

    def format_for_lightmem(self, facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format facts for LightMemory input.

        LightMemory expects messages with time_stamp field.

        Args:
            facts: List of parsed facts

        Returns:
            List of messages formatted for LightMemory
        """
        messages = []

        for idx, fact in enumerate(facts):
            # Create a message with timestamp
            # LightMemory requires time_stamp in format like "2023/05/20 (Sat) 00:44"
            message = {
                "role": "user",
                "content": fact["content"],
                "time_stamp": f"2024/01/01 (Mon) 00:{idx:02d}",  # Sequential timestamps
                "metadata": {
                    "fact_id": fact["id"],
                    "fact_type": fact["type"],
                    "sequence": fact["sequence"]
                }
            }
            messages.append(message)

        return messages

    def prepare_question(self, question: str) -> Dict[str, str]:
        """
        Prepare question for LightMemory query.

        Args:
            question: BABILong question

        Returns:
            Formatted question for LightMemory
        """
        return {
            "role": "user",
            "content": question,
            "time_stamp": "2024/01/01 (Mon) 01:00"
        }

    def convert_sample(self, context: str, question: str) -> Dict[str, Any]:
        """
        Convert a complete BABILong sample to LightMemory format.

        Args:
            context: BABILong context (facts)
            question: BABILong question

        Returns:
            Dictionary with formatted facts and question
        """
        facts = self.parse_babilong_context(context)
        messages = self.format_for_lightmem(facts)
        question_msg = self.prepare_question(question)

        return {
            "facts": facts,
            "messages": messages,
            "question": question_msg,
            "num_facts": len(facts)
        }
