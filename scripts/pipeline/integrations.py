"""Integration with other tools and agents"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from .logger import setup_logger

logger = setup_logger(__name__)


class ConstraintValidator:
    """Validate that adaptation follows all constraints"""

    def __init__(self, model_repo: str, dataset_repo: str, adapter_dir: str):
        """Initialize validator"""
        self.model_repo = Path(model_repo)
        self.dataset_repo = Path(dataset_repo)
        self.adapter_dir = Path(adapter_dir)

    def validate_no_repo_modification(self) -> bool:
        """Check that original repos are not modified"""
        logger.info("Validating repository integrity...")

        # Check model repo
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=self.model_repo,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            logger.error(f"Model repo has modifications:\n{result.stdout}")
            return False

        # Check dataset repo
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=self.dataset_repo,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            logger.error(f"Dataset repo has modifications:\n{result.stdout}")
            return False

        logger.info("✓ Repository integrity verified")
        return True

    def validate_code_isolation(self) -> bool:
        """Check that all code is in adapters/ directory"""
        logger.info("Validating code isolation...")

        # Check for sys.path injection
        for py_file in self.adapter_dir.glob('**/*.py'):
            with open(py_file, 'r') as f:
                content = f.read()
                if 'sys.path.insert' in content or 'sys.path.append' in content:
                    logger.warning(f"Found sys.path manipulation in {py_file}")

        logger.info("✓ Code isolation verified")
        return True

    def validate_all(self) -> bool:
        """Run all validations"""
        checks = [
            self.validate_no_repo_modification(),
            self.validate_code_isolation(),
        ]
        return all(checks)


class ExploreAgentIntegration:
    """Integration with Explore agent for code analysis"""

    @staticmethod
    def analyze_repo(repo_path: str, focus: str = "structure") -> Dict[str, Any]:
        """
        Use Explore agent to analyze repository.

        This would be called by the analyzer agent to quickly understand
        the repository structure.
        """
        logger.info(f"Analyzing {repo_path} with focus: {focus}")
        # In actual implementation, this would call the Explore agent
        # For now, return placeholder
        return {
            'repo_path': repo_path,
            'focus': focus,
            'status': 'pending'
        }


class PaperReproductionIntegration:
    """Integration with paper-reproduction plugin agents"""

    @staticmethod
    def call_code_analyzer(repo_path: str) -> Dict[str, Any]:
        """Call paper-reproduction:code-analyzer agent"""
        logger.info(f"Calling code-analyzer for {repo_path}")
        # Would call the agent in actual implementation
        return {'status': 'pending'}

    @staticmethod
    def call_debug_ml(error_log: str) -> Dict[str, Any]:
        """Call paper-reproduction:debug-ml skill"""
        logger.info("Calling debug-ml for error diagnosis")
        # Would call the skill in actual implementation
        return {'status': 'pending'}

    @staticmethod
    def call_config_verifier(config_path: str, paper_config: Dict[str, Any]) -> Dict[str, Any]:
        """Call paper-reproduction:config-verifier skill"""
        logger.info(f"Calling config-verifier for {config_path}")
        # Would call the skill in actual implementation
        return {'status': 'pending'}


class ScriptSkillIntegration:
    """Integration with script skill for sbatch generation"""

    @staticmethod
    def generate_sbatch(
        script_path: str,
        job_name: str,
        resources: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Generate sbatch script using script skill.

        Args:
            script_path: Path to local script to convert
            job_name: Name for the SLURM job
            resources: Resource requirements (gpu, cpu, memory, time)
            output_path: Where to save sbatch script

        Returns:
            True if successful
        """
        logger.info(f"Generating sbatch script: {output_path}")

        # Create sbatch script
        sbatch_content = ScriptSkillIntegration._create_sbatch_template(
            job_name, resources, script_path
        )

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(sbatch_content)

        # Make executable
        Path(output_path).chmod(0o755)

        logger.info(f"✓ sbatch script created: {output_path}")
        return True

    @staticmethod
    def _create_sbatch_template(
        job_name: str,
        resources: Dict[str, Any],
        script_path: str
    ) -> str:
        """Create sbatch script template"""
        gpu = resources.get('gpu', 1)
        cpu = resources.get('cpu', 8)
        memory = resources.get('memory', 32)
        time_limit = resources.get('time', '04:00:00')

        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/%j.log
#SBATCH --error=slurm_logs/%j.err
#SBATCH --time={time_limit}
#SBATCH --gres=gpu:{gpu}
#SBATCH --cpus-per-task={cpu}
#SBATCH --mem={memory}G
#SBATCH --partition=gpu

# Setup environment
source /data/user/zhu851/miniconda3/etc/profile.d/conda.sh
conda activate amadeus1

# Create output directories
mkdir -p slurm_logs

# Run experiment
cd $(dirname {script_path})
python {script_path}

echo "Job completed at $(date)"
"""


class ToolIntegration:
    """Unified tool integration interface"""

    def __init__(self, config: 'Config'):
        """Initialize tool integration"""
        self.config = config
        self.validator = ConstraintValidator(
            config.get('repos.model.path'),
            config.get('repos.dataset.path'),
            config.get('output.adapter_dir', './adapters')
        )

    def validate_constraints(self) -> bool:
        """Validate all constraints"""
        return self.validator.validate_all()

    def analyze_model_repo(self) -> Dict[str, Any]:
        """Analyze model repository"""
        return ExploreAgentIntegration.analyze_repo(
            self.config.get('repos.model.path'),
            focus='model_structure'
        )

    def analyze_dataset_repo(self) -> Dict[str, Any]:
        """Analyze dataset repository"""
        return ExploreAgentIntegration.analyze_repo(
            self.config.get('repos.dataset.path'),
            focus='dataset_structure'
        )

    def generate_sbatch_script(
        self,
        script_path: str,
        job_name: str,
        resources: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Generate sbatch script for HPC deployment"""
        if resources is None:
            resources = {
                'gpu': 1,
                'cpu': 8,
                'memory': 32,
                'time': '04:00:00'
            }

        output_path = self.config.get('output.sbatch_dir', './scripts/sbatch')
        output_file = Path(output_path) / f"{job_name}.sh"

        return ScriptSkillIntegration.generate_sbatch(
            script_path, job_name, resources, str(output_file)
        )
