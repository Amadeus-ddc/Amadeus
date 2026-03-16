"""Main entry point for the pipeline"""

import argparse
import sys
from pathlib import Path
from .pipeline import ModelDatasetAdapterPipeline
from .logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Model-Dataset Adapter Pipeline'
    )
    parser.add_argument(
        'model_repo',
        help='Path to model repository'
    )
    parser.add_argument(
        'dataset_repo',
        help='Path to dataset repository'
    )
    parser.add_argument(
        '--config',
        default='adapters/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--stages',
        default='analyze,plan,code,run,debug,deploy',
        help='Comma-separated list of stages to run'
    )
    parser.add_argument(
        '--resume-from',
        help='Resume from specific stage'
    )
    parser.add_argument(
        '--skip',
        help='Comma-separated list of stages to skip'
    )
    parser.add_argument(
        '--state-file',
        default='pipeline_state.json',
        help='Path to pipeline state file'
    )

    args = parser.parse_args()

    # Validate input paths
    if not Path(args.model_repo).exists():
        logger.error(f"Model repo not found: {args.model_repo}")
        return 1

    if not Path(args.dataset_repo).exists():
        logger.error(f"Dataset repo not found: {args.dataset_repo}")
        return 1

    # Parse stages
    stages = [s.strip() for s in args.stages.split(',')]
    if args.skip:
        skip_stages = [s.strip() for s in args.skip.split(',')]
        stages = [s for s in stages if s not in skip_stages]

    logger.info(f"Model repo: {args.model_repo}")
    logger.info(f"Dataset repo: {args.dataset_repo}")
    logger.info(f"Stages: {stages}")

    try:
        # Initialize pipeline
        pipeline = ModelDatasetAdapterPipeline(args.config, args.state_file)

        # Run pipeline
        success = pipeline.run(stages=stages, resume_from=args.resume_from)

        # Print summary
        pipeline.print_summary()

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
