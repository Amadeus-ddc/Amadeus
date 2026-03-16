"""
Result processor for LightMemory output on BABILong.

Transforms LightMemory answers to BABILong format.
Handles answer cleaning and task-specific label extraction.
"""

import re
import logging
from typing import Dict, Any, Optional
import sys
import os

# Add paths
BABILONG_PATH = os.environ.get("BABILONG_PATH", "/data/hzy/Amadeus/amadeus/experiments/babilong")
if BABILONG_PATH not in sys.path:
    sys.path.insert(0, BABILONG_PATH)

try:
    from babilong.metrics import compare_answers, TASK_LABELS
except ImportError:
    TASK_LABELS = {}
    compare_answers = None

logger = logging.getLogger("ResultProcessor")


class ResultProcessor:
    """Process LightMemory output for BABILong evaluation."""

    def __init__(self):
        """Initialize result processor."""
        self.task_labels = TASK_LABELS if TASK_LABELS else self._get_default_labels()

    def _get_default_labels(self) -> Dict[str, list]:
        """Get default task labels if import failed."""
        return {
            'qa1': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
            'qa2': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
            'qa3': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
            'qa4': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
            'qa5': ['Bill', 'Fred', 'Jeff', 'Mary', 'apple', 'football', 'milk'],
            'qa6': ['no', 'yes'],
            'qa7': ['none', 'one', 'three', 'two'],
            'qa8': ['apple', 'football', 'milk', 'nothing'],
            'qa9': ['no', 'yes'],
            'qa10': ['maybe', 'no', 'yes'],
            'qa11': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
            'qa12': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
            'qa13': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'],
            'qa14': ['bedroom', 'cinema', 'kitchen', 'office', 'park', 'school'],
            'qa15': ['cat', 'mouse', 'sheep', 'wolf'],
            'qa16': ['gray', 'green', 'white', 'yellow'],
            'qa17': ['no', 'yes'],
            'qa18': ['no', 'yes'],
            'qa19': ['e,e', 'e,n', 'e,s', 'n,e', 'n,n', 'n,w', 's,e', 's,s', 's,w', 'w,n', 'w,s', 'w,w'],
            'qa20': ['bedroom', 'bored', 'garden', 'hungry', 'kitchen', 'thirsty', 'tired']
        }

    def clean_answer(self, answer: str) -> str:
        """
        Clean LightMemory answer by removing LLM fluff.

        Args:
            answer: Raw answer from LightMemory

        Returns:
            Cleaned answer
        """
        if not answer:
            return ""

        answer = answer.strip()

        # Remove common LLM prefixes
        patterns = [
            r"^Based on (the|this|my) (memory|conversation|graph|context|information).*?(\.|,|:)",
            r"^According to .*?(\.|,|:)",
            r"^The (memory|graph|context) indicates (that)?",
            r"^I found (that)?",
            r"^The answer is",
            r"^I can confirm (that)?",
            r"^It is mentioned (that)?",
            r"^From the (facts|context|memory)",
            r"^Based on the provided (facts|context)",
        ]

        for pattern in patterns:
            answer = re.sub(pattern, "", answer, flags=re.IGNORECASE).strip()

        # Take only first sentence
        answer = answer.split('.')[0].strip()

        # Remove common suffixes
        suffixes = [
            r"\s*\(.*?\)$",  # Remove parenthetical remarks
            r"\s*\[.*?\]$",  # Remove bracketed remarks
        ]

        for suffix in suffixes:
            answer = re.sub(suffix, "", answer).strip()

        return answer

    def extract_label(self, answer: str, task: str, question: str) -> Optional[str]:
        """
        Extract task-specific label from answer.

        Args:
            answer: Cleaned answer
            task: Task name (e.g., 'qa1')
            question: Original question

        Returns:
            Extracted label or None
        """
        if task not in self.task_labels:
            return answer

        task_labels = self.task_labels[task]
        answer_lower = answer.lower()
        question_lower = question.lower()

        # Extract labels mentioned in question
        labels_in_question = {label.lower() for label in task_labels if label.lower() in question_lower}

        # Extract labels mentioned in answer
        labels_in_answer = {label.lower() for label in task_labels if label.lower() in answer_lower}

        # Filter out labels mentioned in question (they're never targets)
        labels_in_answer = labels_in_answer - labels_in_question

        if not labels_in_answer:
            return None

        # Return the first extracted label
        return list(labels_in_answer)[0]

    def process_answer(
        self,
        answer: str,
        task: Optional[str] = None,
        question: Optional[str] = None,
        use_task_labels: bool = True
    ) -> str:
        """
        Process LightMemory answer for BABILong evaluation.

        Args:
            answer: Raw answer from LightMemory
            task: Task name for label extraction
            question: Original question
            use_task_labels: Whether to extract task-specific labels

        Returns:
            Processed answer ready for evaluation
        """
        # Clean answer
        cleaned = self.clean_answer(answer)

        # Extract label if requested
        if use_task_labels and task and question:
            label = self.extract_label(cleaned, task, question)
            if label:
                return label

        return cleaned

    def format_result(
        self,
        target: str,
        output: str,
        question: str,
        task: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format result for saving.

        Args:
            target: Expected answer
            output: Model output
            question: Question
            task: Task name
            metadata: Additional metadata

        Returns:
            Formatted result dictionary
        """
        processed_output = self.process_answer(output, task, question)

        result = {
            "target": target,
            "output": processed_output,
            "question": question,
            "raw_output": output,
        }

        if task:
            result["task"] = task

        if metadata:
            result.update(metadata)

        return result

    def evaluate_answer(
        self,
        target: str,
        output: str,
        question: str,
        task: Optional[str] = None
    ) -> bool:
        """
        Evaluate if output matches target using BABILong metrics.

        Args:
            target: Expected answer
            output: Model output
            question: Question
            task: Task name

        Returns:
            True if answer is correct
        """
        processed_output = self.process_answer(output, task, question)

        # Use BABILong's compare_answers if available
        if compare_answers and task:
            try:
                task_labels = self.task_labels.get(task, [])
                return compare_answers(target, processed_output, question, task_labels)
            except Exception as e:
                logger.warning(f"Error using compare_answers: {e}")

        # Fallback to simple matching
        target_lower = target.lower()
        output_lower = processed_output.lower()

        # Exact match
        if target_lower == output_lower:
            return True

        # Substring match
        if target_lower in output_lower:
            return True

        return False
