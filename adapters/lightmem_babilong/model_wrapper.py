"""
Model wrapper for LightMemory on BABILong.

Wraps LightMemory to work with BABILong benchmark.
Does NOT modify LightMemory's core reading/processing/retrieval mechanisms.
"""

import logging
import json
from typing import Dict, Any, Optional, List
import sys
import os
from pathlib import Path

# Add LightMem to path
LIGHTMEM_PATH = os.environ.get("LIGHTMEM_PATH", "/data/hzy/Amadeus/lightmem/LightMem")
if LIGHTMEM_PATH not in sys.path:
    sys.path.insert(0, LIGHTMEM_PATH)

try:
    from lightmem.memory.lightmem import LightMemory
except ImportError as e:
    logging.warning(f"Could not import LightMemory: {e}. Will use mock for testing.")
    LightMemory = None

from prompts import BABILONG_METADATA_PROMPT, build_answer_prompt

logger = logging.getLogger("LightMemWrapper")


class LightMemWrapper:
    """
    Wrapper for LightMemory to work with BABILong.

    This wrapper:
    - Initializes LightMemory with provided config
    - Adds BABILong facts to memory using add_memory()
    - Queries memory with questions
    - Returns answers in BABILong format
    """

    def __init__(self, config: Dict[str, Any], sample_id: str):
        """
        Initialize LightMemory wrapper.

        Args:
            config: Configuration dictionary with lightmem settings
            sample_id: Unique identifier for this sample (used as collection_name)
        """
        self.config = config
        self.sample_id = sample_id
        self.lightmem_config = config.get("lightmem", {})
        self.babilong_config = config.get("babilong", {})

        # Initialize LightMemory with sample-specific collection
        self.memory = self._initialize_lightmem(sample_id)
        self.llm_client = self._initialize_llm()

        logger.info(f"LightMemWrapper initialized for sample: {sample_id}")

    def _initialize_lightmem(self, sample_id: str) -> Optional[Any]:
        """
        Initialize LightMemory with configuration.

        Args:
            sample_id: Sample identifier for collection_name

        Returns:
            LightMemory instance or None if import failed
        """
        if LightMemory is None:
            logger.warning("LightMemory not available, using mock mode")
            return None

        try:
            # Build LightMemory config
            model_path = os.environ.get("MODEL_PATH", "/data/hzy/models/Qwen2.5-7B-Instruct")
            embedding_model_path = os.environ.get(
                "EMBEDDING_MODEL_PATH",
                "/data/hzy/Amadeus/amadeus/models/all-MiniLM-L6-v2"
            )
            qdrant_dir = os.environ.get(
                "QDRANT_DIR",
                "/data/hzy/Amadeus/amadeus/adapters/lightmem_babilong/qdrant_data"
            )

            # Create Qdrant directory if it doesn't exist
            Path(qdrant_dir).mkdir(parents=True, exist_ok=True)

            config_dict = {
                "pre_compress": False,
                "topic_segment": False,
                "metadata_generate": True,
                "text_summary": True,
                "memory_manager": {
                    "model_name": "transformers",
                    "configs": {
                        "model": model_path,
                        "num_gpu": -1,
                        "gpu_memory_utilization": 0.9,
                        "trust_remote_code": True,
                        "max_tokens": 512,
                    },
                },
                "extract_threshold": 0.1,
                "index_strategy": "embedding",
                "text_embedder": {
                    "model_name": "huggingface",
                    "configs": {
                        "model": embedding_model_path,
                        "embedding_dims": 384,
                        "model_kwargs": {"device": "cpu"},
                    },
                },
                "retrieve_strategy": "embedding",
                "embedding_retriever": {
                    "model_name": "qdrant",
                    "configs": {
                        "collection_name": sample_id,
                        "embedding_model_dims": 384,
                        "path": f"{qdrant_dir}/{sample_id}",
                    }
                },
                "update": "offline",
                "logging": {
                    "level": "INFO",
                    "file_enabled": False,
                },
            }

            memory = LightMemory.from_config(config_dict)
            logger.info(f"LightMemory initialized successfully for sample {sample_id}")
            return memory

        except Exception as e:
            logger.error(f"Failed to initialize LightMemory: {e}")
            return None

    def _initialize_llm(self) -> Optional[Any]:
        """
        Initialize LLM client for answer generation.

        Returns:
            LLM client or None
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_path = os.environ.get("MODEL_PATH", "/data/hzy/models/Qwen2.5-7B-Instruct")

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            return {"model": model, "tokenizer": tokenizer}

        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            return None

    def add_facts(self, messages: List[Dict[str, str]]) -> bool:
        """
        Add facts to LightMemory using add_memory().

        Args:
            messages: List of message dictionaries with content and time_stamp

        Returns:
            True if successful
        """
        if self.memory is None:
            logger.warning("LightMemory not available")
            return False

        try:
            # Call add_memory with BABILong-specific prompt
            result = self.memory.add_memory(
                messages=messages,
                METADATA_GENERATE_PROMPT=BABILONG_METADATA_PROMPT,
                force_extract=True,
            )

            logger.info(f"Added {len(messages)} facts to LightMemory")
            logger.debug(f"add_memory result: {result}")
            return True

        except Exception as e:
            logger.error(f"Error adding facts to LightMemory: {e}")
            return False

    def retrieve(self, question: str, limit: int = 10) -> str:
        """
        Retrieve relevant facts from LightMemory.

        Args:
            question: Question to answer
            limit: Maximum number of facts to retrieve

        Returns:
            Formatted string of retrieved facts
        """
        if self.memory is None:
            logger.warning("LightMemory not available")
            return ""

        try:
            # Call retrieve method
            retrieved = self.memory.retrieve(question, limit=limit)

            logger.info(f"Retrieved facts for question: {question[:50]}...")
            logger.debug(f"Retrieved content:\n{retrieved}")
            return retrieved

        except Exception as e:
            logger.error(f"Error retrieving from LightMemory: {e}")
            return ""

    def generate_answer(self, question: str, retrieved_facts: str, task: Optional[str] = None) -> str:
        """
        Generate answer using LLM based on retrieved facts.

        Uses task-specific prompts from amadeus's DEFAULT_PROMPTS.

        Args:
            question: Original question
            retrieved_facts: Retrieved facts from LightMemory
            task: Task name (e.g., 'qa1') for task-specific prompts

        Returns:
            Generated answer
        """
        if self.llm_client is None:
            logger.warning("LLM not available, returning mock answer")
            return self._mock_answer(question)

        try:
            # Build prompt using task-specific template
            context = retrieved_facts if retrieved_facts else "No relevant facts found."
            prompt = build_answer_prompt(task or "qa1", context, question)

            # Generate answer
            model = self.llm_client["model"]
            tokenizer = self.llm_client["tokenizer"]

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part (after the prompt)
            answer = answer[len(prompt):].strip()

            logger.info(f"Generated answer: {answer}")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self._mock_answer(question)

    def _mock_answer(self, question: str) -> str:
        """
        Generate mock answer for testing.

        Args:
            question: Question to answer

        Returns:
            Mock answer
        """
        return "unknown"

    def process_sample(self, context: str, question: str, converted_data: Dict[str, Any], task: Optional[str] = None) -> str:
        """
        Process a complete BABILong sample.

        Args:
            context: BABILong context (facts)
            question: BABILong question
            converted_data: Output from DataConverter.convert_sample()
            task: Task name for task-specific prompts

        Returns:
            Answer from LightMemory
        """
        # Add facts to memory
        self.add_facts(converted_data["messages"])

        # Retrieve relevant facts
        retrieved_facts = self.retrieve(question)

        # Generate answer using task-specific prompt
        answer = self.generate_answer(question, retrieved_facts, task=task)

        return answer

    def reset(self) -> None:
        """Reset memory for next sample."""
        # Reinitialize LightMemory for fresh state
        if self.memory is not None:
            try:
                self.memory = self._initialize_lightmem(self.sample_id)
            except Exception as e:
                logger.error(f"Error resetting LightMemory: {e}")
