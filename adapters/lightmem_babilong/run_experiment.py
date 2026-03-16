"""
Main experiment runner for LightMemory on BABILong.

Orchestrates the complete pipeline:
1. Load configuration
2. Load dataset
3. Process each sample
4. Save results
5. Compute metrics
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml

import datasets
import pandas as pd
from tqdm import tqdm

# Add paths
SCRIPT_DIR = Path(__file__).parent
BABILONG_DIR = Path("/data/hzy/Amadeus/amadeus/experiments/babilong")
AMADEUS_DIR = Path("/data/hzy/Amadeus/amadeus")

if str(BABILONG_DIR) not in sys.path:
    sys.path.insert(0, str(BABILONG_DIR))
if str(AMADEUS_DIR) not in sys.path:
    sys.path.insert(0, str(AMADEUS_DIR))

from data_converter import DataConverter
from model_wrapper import LightMemWrapper
from result_processor import ResultProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BABILong.LightMem")


class BABILongExperiment:
    """Run LightMemory on BABILong benchmark."""

    def __init__(self, config_path: str, output_dir: str = "./results"):
        """
        Initialize experiment.

        Args:
            config_path: Path to configuration file
            output_dir: Directory to save results
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.converter = DataConverter(
            max_context_length=self.config.get("babilong", {}).get("max_context_length", 4000)
        )
        self.processor = ResultProcessor()

        logger.info(f"Experiment initialized with config: {config_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Replace environment variables
        config = self._replace_env_vars(config)
        return config

    def _replace_env_vars(self, obj: Any) -> Any:
        """
        Recursively replace ${VAR} with environment variables.

        Args:
            obj: Object to process

        Returns:
            Object with environment variables replaced
        """
        if isinstance(obj, str):
            import re
            def replace_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            return re.sub(r'\$\{(\w+)\}', replace_var, obj)
        elif isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        return obj

    def load_dataset(self, dataset_name: str, split: str) -> List[Dict[str, Any]]:
        """
        Load BABILong dataset from local, HPC cache, or Hub.

        Args:
            dataset_name: Dataset name (e.g., 'babilong')
            split: Split name (e.g., '1k', '4k', '16k')

        Returns:
            List of samples
        """
        logger.info(f"Loading dataset: {dataset_name}, split: {split}")

        try:
            # Try loading from local test data first
            local_path = Path(__file__).parent / "test_data" / f"{split}.json"
            if local_path.exists():
                logger.info(f"Loading from local test data: {local_path}")
                with open(local_path, 'r') as f:
                    data = json.load(f)

                samples = []
                for task_name, task_data in data.items():
                    for idx, item in enumerate(task_data):
                        sample = {
                            'sample_id': f"{task_name}_{split}_{idx}",
                            'task': task_name,
                            'split': split,
                            'context': item.get('context', ''),
                            'question': item.get('question', ''),
                            'answer': item.get('answer', ''),
                        }
                        samples.append(sample)

                logger.info(f"Loaded {len(samples)} samples from local test data")
                return samples

            # Try loading from HPC cache
            hpc_cache_path = os.environ.get('HPC_DATASET_PATH')
            if hpc_cache_path:
                hpc_dataset_path = Path(hpc_cache_path) / split
                if hpc_dataset_path.exists():
                    logger.info(f"Loading from HPC cache: {hpc_dataset_path}")
                    data = datasets.load_dataset(str(hpc_dataset_path))
                    samples = []

                    # Handle different dataset structures
                    if 'context' in data.column_names:
                        for idx, item in enumerate(data):
                            sample = {
                                'sample_id': f"{item.get('task', 'qa1')}_{split}_{idx}",
                                'task': item.get('task', 'qa1'),
                                'split': split,
                                'context': item.get('context', ''),
                                'question': item.get('question', ''),
                                'answer': item.get('answer', ''),
                            }
                            samples.append(sample)
                    else:
                        # Handle task-based structure
                        for task_name in data.column_names:
                            task_data = data[task_name]
                            for idx, item in enumerate(task_data):
                                sample = {
                                    'sample_id': f"{task_name}_{split}_{idx}",
                                    'task': task_name,
                                    'split': split,
                                    'context': item.get('context', ''),
                                    'question': item.get('question', ''),
                                    'answer': item.get('answer', ''),
                                }
                                samples.append(sample)

                    logger.info(f"Loaded {len(samples)} samples from HPC cache")
                    return samples

            # Fall back to Hub
            logger.info(f"Loading from Hugging Face Hub: {dataset_name}")
            data = datasets.load_dataset(dataset_name, split)
            samples = []

            for task_name in data.column_names:
                task_data = data[task_name]
                for idx, item in enumerate(task_data):
                    sample = {
                        'sample_id': f"{task_name}_{split}_{idx}",
                        'task': task_name,
                        'split': split,
                        'context': item.get('context', ''),
                        'question': item.get('question', ''),
                        'answer': item.get('answer', ''),
                    }
                    samples.append(sample)

            logger.info(f"Loaded {len(samples)} samples")
            return samples

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []

    def process_sample(self, sample: Dict[str, Any], wrapper: LightMemWrapper) -> Dict[str, Any]:
        """
        Process a single sample.

        Args:
            sample: Sample dictionary
            wrapper: LightMemWrapper instance

        Returns:
            Result dictionary
        """
        try:
            context = sample['context']
            question = sample['question']
            target = sample['answer']
            task = sample['task']

            # Convert data
            converted = self.converter.convert_sample(context, question)

            # Process with LightMemory (pass task for task-specific prompts)
            output = wrapper.process_sample(context, question, converted, task=task)

            # Process result
            result = self.processor.format_result(
                target=target,
                output=output,
                question=question,
                task=task,
                metadata={
                    'sample_id': sample['sample_id'],
                    'split': sample['split'],
                }
            )

            # Evaluate
            is_correct = self.processor.evaluate_answer(target, output, question, task)
            result['correct'] = is_correct

            return result

        except Exception as e:
            logger.error(f"Error processing sample {sample['sample_id']}: {e}")
            return {
                'sample_id': sample['sample_id'],
                'error': str(e),
                'correct': False,
            }

    def run(self, dataset_name: str = "babilong", splits: List[str] = None, tasks: List[str] = None):
        """
        Run experiment on BABILong dataset.

        Args:
            dataset_name: Dataset name
            splits: List of splits to evaluate
            tasks: List of tasks to evaluate (empty = all)
        """
        if splits is None:
            splits = self.config.get("babilong", {}).get("splits", ["1k"])

        logger.info(f"Starting experiment on {dataset_name}")
        logger.info(f"Splits: {splits}")
        logger.info(f"Tasks: {tasks if tasks else 'all'}")

        all_results = []

        for split in splits:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing split: {split}")
            logger.info(f"{'='*70}")

            # Load dataset
            samples = self.load_dataset(dataset_name, split)

            if not samples:
                logger.warning(f"No samples loaded for split {split}")
                continue

            # Filter by task if specified
            if tasks:
                samples = [s for s in samples if s['task'] in tasks]

            # Process samples
            results = []
            for sample in tqdm(samples, desc=f"Processing {split}"):
                # Create wrapper for each sample (fresh LightMemory instance)
                wrapper = LightMemWrapper(self.config, sample['sample_id'])

                # Process sample
                result = self.process_sample(sample, wrapper)
                results.append(result)

                # Reset wrapper
                wrapper.reset()

            # Save results for this split
            self._save_results(results, split)
            all_results.extend(results)

            # Print statistics
            self._print_statistics(results, split)

        # Print overall statistics
        logger.info(f"\n{'='*70}")
        logger.info("OVERALL STATISTICS")
        logger.info(f"{'='*70}")
        self._print_statistics(all_results, "overall")

    def _save_results(self, results: List[Dict[str, Any]], split: str):
        """
        Save results to CSV and JSON.

        Args:
            results: List of result dictionaries
            split: Split name
        """
        # Save as CSV
        csv_path = self.output_dir / f"results_{split}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")

        # Save as JSON
        json_path = self.output_dir / f"results_{split}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {json_path}")

    def _print_statistics(self, results: List[Dict[str, Any]], split: str):
        """
        Print evaluation statistics.

        Args:
            results: List of result dictionaries
            split: Split name
        """
        if not results:
            logger.warning(f"No results for {split}")
            return

        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            logger.warning(f"No valid results for {split}")
            return

        # Calculate metrics
        total = len(valid_results)
        correct = sum(1 for r in valid_results if r.get('correct', False))
        accuracy = correct / total if total > 0 else 0

        logger.info(f"\n[{split}] Statistics:")
        logger.info(f"  Total samples: {total}")
        logger.info(f"  Correct: {correct}")
        logger.info(f"  Accuracy: {accuracy:.2%}")

        # Per-task statistics
        tasks = set(r.get('task', 'unknown') for r in valid_results)
        for task in sorted(tasks):
            task_results = [r for r in valid_results if r.get('task') == task]
            task_correct = sum(1 for r in task_results if r.get('correct', False))
            task_accuracy = task_correct / len(task_results) if task_results else 0
            logger.info(f"  {task}: {task_accuracy:.2%} ({task_correct}/{len(task_results)})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run LightMemory on BABILong')
    parser.add_argument(
        '--config',
        default=str(Path(__file__).parent / 'config.yaml'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir',
        default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--dataset',
        default='babilong',
        help='Dataset name'
    )
    parser.add_argument(
        '--splits',
        default='1k',
        help='Comma-separated list of splits to evaluate'
    )
    parser.add_argument(
        '--tasks',
        default='',
        help='Comma-separated list of tasks to evaluate (empty = all)'
    )

    args = parser.parse_args()

    # Parse arguments
    splits = [s.strip() for s in args.splits.split(',')]
    tasks = [t.strip() for t in args.tasks.split(',')] if args.tasks else None

    # Run experiment
    experiment = BABILongExperiment(args.config, args.output_dir)
    experiment.run(dataset_name=args.dataset, splits=splits, tasks=tasks)


if __name__ == '__main__':
    main()
