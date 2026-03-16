#!/usr/bin/env python3
"""
Test script for LightMemory BABILong adapter.

Tests:
1. Configuration loading
2. Data conversion
3. LightMemory initialization
4. Answer generation
"""

import sys
import os
import logging
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestAdapter")

# Set environment variables for testing
os.environ.setdefault("MODEL_PATH", "/data/hzy/models/Qwen2.5-7B-Instruct")
os.environ.setdefault("EMBEDDING_MODEL_PATH", "/data/hzy/Amadeus/amadeus/models/all-MiniLM-L6-v2")
os.environ.setdefault("QDRANT_DIR", str(SCRIPT_DIR / "qdrant_data"))
os.environ.setdefault("LIGHTMEM_PATH", "/data/hzy/Amadeus/lightmem/LightMem")
os.environ.setdefault("BABILONG_PATH", "/data/hzy/Amadeus/amadeus/experiments/babilong")


def test_imports():
    """Test that all imports work."""
    logger.info("=" * 70)
    logger.info("TEST 1: Imports")
    logger.info("=" * 70)

    try:
        from data_converter import DataConverter
        logger.info("✓ DataConverter imported")

        from result_processor import ResultProcessor
        logger.info("✓ ResultProcessor imported")

        from prompts import BABILONG_METADATA_PROMPT, build_answer_prompt
        logger.info("✓ Prompts imported")

        from model_wrapper import LightMemWrapper
        logger.info("✓ LightMemWrapper imported")

        logger.info("✓ All imports successful\n")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}\n")
        return False


def test_data_converter():
    """Test data converter."""
    logger.info("=" * 70)
    logger.info("TEST 2: Data Converter")
    logger.info("=" * 70)

    try:
        from data_converter import DataConverter

        converter = DataConverter()
        logger.info("✓ DataConverter initialized")

        # Test with sample BABILong context
        context = """
        Charlie went to the hallway. Judith came back to the kitchen.
        Charlie travelled to balcony. Mary went to the office.
        """
        question = "Where is Charlie?"

        result = converter.convert_sample(context, question)

        logger.info(f"✓ Converted {result['num_facts']} facts")
        logger.info(f"✓ Generated {len(result['messages'])} messages")

        # Check message format
        if result['messages']:
            msg = result['messages'][0]
            assert 'role' in msg, "Missing 'role' in message"
            assert 'content' in msg, "Missing 'content' in message"
            assert 'time_stamp' in msg, "Missing 'time_stamp' in message"
            logger.info("✓ Message format correct")

        logger.info("✓ Data converter test passed\n")
        return True
    except Exception as e:
        logger.error(f"✗ Data converter test failed: {e}\n")
        return False


def test_result_processor():
    """Test result processor."""
    logger.info("=" * 70)
    logger.info("TEST 3: Result Processor")
    logger.info("=" * 70)

    try:
        from result_processor import ResultProcessor

        processor = ResultProcessor()
        logger.info("✓ ResultProcessor initialized")

        # Test answer cleaning
        raw_answer = "Based on the facts provided, the answer is: balcony."
        cleaned = processor.clean_answer(raw_answer)
        logger.info(f"✓ Cleaned answer: '{cleaned}'")

        # Test label extraction
        label = processor.extract_label("balcony", "qa1", "Where is Charlie?")
        logger.info(f"✓ Extracted label: {label}")

        # Test result formatting
        result = processor.format_result(
            target="balcony",
            output="The most recent location of Charlie is balcony.",
            question="Where is Charlie?",
            task="qa1"
        )
        logger.info(f"✓ Formatted result: {result}")

        # Test evaluation
        is_correct = processor.evaluate_answer(
            target="balcony",
            output="The most recent location of Charlie is balcony.",
            question="Where is Charlie?",
            task="qa1"
        )
        logger.info(f"✓ Evaluation result: {is_correct}")

        logger.info("✓ Result processor test passed\n")
        return True
    except Exception as e:
        logger.error(f"✗ Result processor test failed: {e}\n")
        return False


def test_prompts():
    """Test prompt generation."""
    logger.info("=" * 70)
    logger.info("TEST 4: Prompts")
    logger.info("=" * 70)

    try:
        from prompts import build_answer_prompt, BABILONG_METADATA_PROMPT

        # Test metadata prompt
        logger.info(f"✓ BABILONG_METADATA_PROMPT length: {len(BABILONG_METADATA_PROMPT)}")

        # Test answer prompt for different tasks
        facts = "Charlie went to the hallway. Charlie travelled to balcony."
        question = "Where is Charlie?"

        for task in ["qa1", "qa2", "qa3", "qa4", "qa5"]:
            prompt = build_answer_prompt(task, facts, question)
            logger.info(f"✓ Generated prompt for {task} (length: {len(prompt)})")

        # Test fallback for unknown task
        prompt = build_answer_prompt("unknown_task", facts, question)
        logger.info(f"✓ Generated fallback prompt (length: {len(prompt)})")

        logger.info("✓ Prompts test passed\n")
        return True
    except Exception as e:
        logger.error(f"✗ Prompts test failed: {e}\n")
        return False


def test_lightmem_wrapper():
    """Test LightMemory wrapper initialization."""
    logger.info("=" * 70)
    logger.info("TEST 5: LightMemory Wrapper")
    logger.info("=" * 70)

    try:
        from model_wrapper import LightMemWrapper

        config = {
            "lightmem": {},
            "babilong": {}
        }

        wrapper = LightMemWrapper(config, sample_id="test_sample_001")
        logger.info("✓ LightMemWrapper initialized")

        # Check attributes
        assert hasattr(wrapper, 'memory'), "Missing 'memory' attribute"
        assert hasattr(wrapper, 'llm_client'), "Missing 'llm_client' attribute"
        logger.info("✓ LightMemWrapper attributes correct")

        # Test mock answer (when LLM not available)
        mock_answer = wrapper._mock_answer("test question")
        logger.info(f"✓ Mock answer: '{mock_answer}'")

        logger.info("✓ LightMemory wrapper test passed\n")
        return True
    except Exception as e:
        logger.error(f"✗ LightMemory wrapper test failed: {e}\n")
        return False


def test_end_to_end():
    """Test end-to-end pipeline (without actual LLM)."""
    logger.info("=" * 70)
    logger.info("TEST 6: End-to-End Pipeline")
    logger.info("=" * 70)

    try:
        from data_converter import DataConverter
        from result_processor import ResultProcessor
        from model_wrapper import LightMemWrapper

        # Setup
        converter = DataConverter()
        processor = ResultProcessor()
        config = {"lightmem": {}, "babilong": {}}
        wrapper = LightMemWrapper(config, sample_id="test_e2e_001")

        # Sample data
        context = "Charlie went to the hallway. Charlie travelled to balcony."
        question = "Where is Charlie?"
        target = "balcony"
        task = "qa1"

        # Convert
        converted = converter.convert_sample(context, question)
        logger.info(f"✓ Converted {converted['num_facts']} facts")

        # Process (will use mock answer since LLM not available)
        output = wrapper.process_sample(context, question, converted, task=task)
        logger.info(f"✓ Generated output: '{output}'")

        # Process result
        result = processor.format_result(
            target=target,
            output=output,
            question=question,
            task=task
        )
        logger.info(f"✓ Formatted result")

        # Evaluate
        is_correct = processor.evaluate_answer(target, output, question, task)
        logger.info(f"✓ Evaluation: {is_correct}")

        logger.info("✓ End-to-end pipeline test passed\n")
        return True
    except Exception as e:
        logger.error(f"✗ End-to-end pipeline test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 70)
    logger.info("LightMemory BABILong Adapter Test Suite")
    logger.info("=" * 70 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Data Converter", test_data_converter),
        ("Result Processor", test_result_processor),
        ("Prompts", test_prompts),
        ("LightMemory Wrapper", test_lightmem_wrapper),
        ("End-to-End Pipeline", test_end_to_end),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"✗ {name} test crashed: {e}\n")
            results[name] = False

    # Summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info("=" * 70 + "\n")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
