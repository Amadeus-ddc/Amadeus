"""Configuration validation tool"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple


class ConfigValidator:
    """Validate pipeline configuration"""

    REQUIRED_KEYS = {
        'pipeline': ['name'],
        'repos': ['model', 'dataset'],
        'repos.model': ['path'],
        'repos.dataset': ['path'],
        'stages': [],
        'output': ['adapter_dir', 'results_dir', 'logs_dir']
    }

    def __init__(self, config_path: str):
        """Initialize validator"""
        self.config_path = Path(config_path)
        self.config = None
        self.errors = []
        self.warnings = []

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate configuration.

        Returns:
            (is_valid, errors, warnings)
        """
        # Load config
        if not self._load_config():
            return False, self.errors, self.warnings

        # Run validations
        self._validate_structure()
        self._validate_paths()
        self._validate_constraints()

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _load_config(self) -> bool:
        """Load and parse YAML config"""
        if not self.config_path.exists():
            self.errors.append(f"Config file not found: {self.config_path}")
            return False

        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML: {e}")
            return False

    def _validate_structure(self) -> None:
        """Validate config structure"""
        for key_path, required_subkeys in self.REQUIRED_KEYS.items():
            keys = key_path.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                    if value is None:
                        self.errors.append(f"Missing required key: {key_path}")
                        return
                else:
                    self.errors.append(f"Invalid config structure at: {key_path}")
                    return

            # Check subkeys
            if isinstance(value, dict):
                for subkey in required_subkeys:
                    if subkey not in value:
                        self.errors.append(f"Missing required key: {key_path}.{subkey}")

    def _validate_paths(self) -> None:
        """Validate that paths exist"""
        model_path = self.config.get('repos', {}).get('model', {}).get('path')
        dataset_path = self.config.get('repos', {}).get('dataset', {}).get('path')

        if model_path:
            if not Path(model_path).exists():
                self.errors.append(f"Model path does not exist: {model_path}")
            else:
                if not Path(model_path).is_dir():
                    self.errors.append(f"Model path is not a directory: {model_path}")

        if dataset_path:
            if not Path(dataset_path).exists():
                self.errors.append(f"Dataset path does not exist: {dataset_path}")
            else:
                if not Path(dataset_path).is_dir():
                    self.errors.append(f"Dataset path is not a directory: {dataset_path}")

    def _validate_constraints(self) -> None:
        """Validate constraint definitions"""
        constraints = self.config.get('constraints', {})

        if not constraints:
            self.warnings.append("No constraints defined in config")
            return

        # Check for required constraint sections
        required_sections = ['model_integrity', 'dataset_integrity', 'code_isolation']
        for section in required_sections:
            if section not in constraints:
                self.warnings.append(f"Missing constraint section: {section}")

    def print_report(self) -> None:
        """Print validation report"""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION REPORT")
        print("="*60)
        print(f"Config file: {self.config_path}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n✓ Configuration is valid")

        print("="*60 + "\n")


def main():
    """Main entry point for validation"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate pipeline configuration')
    parser.add_argument('config', help='Path to configuration file')
    args = parser.parse_args()

    validator = ConfigValidator(args.config)
    is_valid, errors, warnings = validator.validate()

    validator.print_report()

    return 0 if is_valid else 1


if __name__ == '__main__':
    sys.exit(main())
