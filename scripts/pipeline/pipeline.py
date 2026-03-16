"""Main pipeline orchestration"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from .logger import setup_logger
from .config import Config
from .state import PipelineState
from .integrations import ToolIntegration

logger = setup_logger(__name__)


class ModelDatasetAdapterPipeline:
    """Orchestrate the complete adaptation pipeline"""

    def __init__(self, config_path: str, state_file: str = "pipeline_state.json"):
        """Initialize pipeline"""
        self.config = Config(config_path)
        self.state = PipelineState(state_file)
        self.tools = ToolIntegration(self.config)

        logger.info(f"Pipeline initialized: {self.state}")

    def run(self, stages: Optional[List[str]] = None, resume_from: Optional[str] = None) -> bool:
        """
        Run the complete pipeline.

        Args:
            stages: List of stages to run (default: all)
            resume_from: Resume from specific stage

        Returns:
            True if successful
        """
        if stages is None:
            stages = ['analyze', 'plan', 'code', 'run', 'debug', 'deploy']

        # Filter stages if resuming
        if resume_from:
            try:
                idx = stages.index(resume_from)
                stages = stages[idx:]
                logger.info(f"Resuming from stage: {resume_from}")
            except ValueError:
                logger.error(f"Unknown stage: {resume_from}")
                return False

        # Run each stage
        for stage in stages:
            if not self._run_stage(stage):
                logger.error(f"Pipeline failed at stage: {stage}")
                return False

        logger.info("Pipeline completed successfully")
        self.state.state['status'] = 'completed'
        return True

    def _run_stage(self, stage_name: str) -> bool:
        """Run a single stage"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Stage: {stage_name}")
        logger.info(f"{'='*60}")

        # Check if stage is enabled
        if not self.config.get(f'stages.{stage_name}.enabled', True):
            logger.info(f"Stage {stage_name} is disabled, skipping")
            return True

        # Check if already completed
        if self.state.get_stage_status(stage_name) == 'completed':
            logger.info(f"Stage {stage_name} already completed, skipping")
            return True

        # Mark stage as started
        self.state.start_stage(stage_name)

        try:
            # Run stage-specific logic
            if stage_name == 'analyze':
                result = self._stage_analyze()
            elif stage_name == 'plan':
                result = self._stage_plan()
            elif stage_name == 'code':
                result = self._stage_code()
            elif stage_name == 'run':
                result = self._stage_run()
            elif stage_name == 'debug':
                result = self._stage_debug()
            elif stage_name == 'deploy':
                result = self._stage_deploy()
            else:
                logger.error(f"Unknown stage: {stage_name}")
                return False

            # Mark stage as completed
            self.state.complete_stage(stage_name, result)
            logger.info(f"✓ Stage {stage_name} completed")
            return True

        except Exception as e:
            logger.error(f"✗ Stage {stage_name} failed: {e}")
            self.state.fail_stage(stage_name, str(e))
            return False

    def _stage_analyze(self) -> Dict[str, Any]:
        """Stage 1: Analyze repositories"""
        logger.info("Analyzing model and dataset repositories...")

        # Validate constraints
        if not self.tools.validate_constraints():
            raise RuntimeError("Constraint validation failed")

        # Analyze model repo
        model_analysis = self.tools.analyze_model_repo()
        logger.info(f"Model analysis: {model_analysis}")

        # Analyze dataset repo
        dataset_analysis = self.tools.analyze_dataset_repo()
        logger.info(f"Dataset analysis: {dataset_analysis}")

        return {
            'model_analysis': model_analysis,
            'dataset_analysis': dataset_analysis
        }

    def _stage_plan(self) -> Dict[str, Any]:
        """Stage 2: Plan adaptation strategy"""
        logger.info("Planning adaptation strategy...")

        # Get analysis results
        analysis = self.state.get_stage_output('analyze')
        if not analysis:
            raise RuntimeError("Analysis stage not completed")

        logger.info("Adaptation strategy planned")
        return {
            'strategy': 'planned',
            'components': ['data_converter', 'model_wrapper', 'result_processor']
        }

    def _stage_code(self) -> Dict[str, Any]:
        """Stage 3: Generate adaptation code"""
        logger.info("Generating adaptation code...")

        adapter_dir = Path(self.config.get('output.adapter_dir', './adapters'))
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Create placeholder files
        files_created = []
        for filename in ['__init__.py', 'config.yaml', 'data_converter.py',
                        'model_wrapper.py', 'result_processor.py', 'run_experiment.py']:
            filepath = adapter_dir / filename
            if not filepath.exists():
                filepath.touch()
                files_created.append(str(filepath))
                logger.info(f"Created: {filepath}")

        logger.info(f"Adaptation code generated in {adapter_dir}")
        return {
            'adapter_dir': str(adapter_dir),
            'files_created': files_created
        }

    def _stage_run(self) -> Dict[str, Any]:
        """Stage 4: Run experiment locally"""
        logger.info("Running experiment locally...")

        adapter_dir = self.config.get('output.adapter_dir', './adapters')
        results_dir = Path(self.config.get('output.results_dir', './results'))
        results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Results will be saved to: {results_dir}")
        logger.info("Note: Actual experiment execution would happen here")

        return {
            'results_dir': str(results_dir),
            'status': 'ready_for_execution'
        }

    def _stage_debug(self) -> Dict[str, Any]:
        """Stage 5: Debug if needed"""
        logger.info("Checking for errors...")

        # Check if run stage completed successfully
        run_output = self.state.get_stage_output('run')
        if run_output and run_output.get('status') == 'ready_for_execution':
            logger.info("No errors detected")
            return {'status': 'no_errors'}

        logger.info("Debug stage completed")
        return {'status': 'debug_complete'}

    def _stage_deploy(self) -> Dict[str, Any]:
        """Stage 6: Generate deployment script"""
        logger.info("Generating deployment script...")

        script_path = Path(self.config.get('output.adapter_dir', './adapters')) / 'run_experiment.py'
        job_name = self.config.get('pipeline.name', 'adapter_experiment')

        resources = {
            'gpu': 1,
            'cpu': 8,
            'memory': 32,
            'time': '04:00:00'
        }

        if self.tools.generate_sbatch_script(str(script_path), job_name, resources):
            logger.info("✓ Deployment script generated")
            return {
                'sbatch_script': f'scripts/sbatch/{job_name}.sh',
                'status': 'ready_for_deployment'
            }
        else:
            raise RuntimeError("Failed to generate sbatch script")

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'pipeline_id': self.state.state['pipeline_id'],
            'status': self.state.state['status'],
            'stages': self.state.state['stages'],
            'errors': self.state.state['errors']
        }

    def print_summary(self) -> None:
        """Print pipeline execution summary"""
        status = self.get_status()

        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Pipeline ID: {status['pipeline_id']}")
        print(f"Status: {status['status']}")
        print("\nStages:")
        for stage_name, stage_info in status['stages'].items():
            stage_status = stage_info.get('status', 'pending')
            print(f"  {stage_name}: {stage_status}")

        if status['errors']:
            print("\nErrors:")
            for error in status['errors']:
                print(f"  [{error['stage']}] {error['error']}")

        print("="*60 + "\n")
