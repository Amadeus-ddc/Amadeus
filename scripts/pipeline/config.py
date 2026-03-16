"""Configuration management for the pipeline"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Load and manage pipeline configuration"""

    def __init__(self, config_path: str):
        """Load configuration from YAML file"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Replace environment variables
        config = self._replace_env_vars(config)
        return config

    def _replace_env_vars(self, obj: Any) -> Any:
        """Recursively replace ${VAR} with environment variables"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Replace ${VAR} with environment variable
            import re
            def replace_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            return re.sub(r'\$\{([^}]+)\}', replace_var, obj)
        else:
            return obj

    def _validate_config(self) -> None:
        """Validate configuration structure"""
        required_keys = ['pipeline', 'repos', 'stages', 'output']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Validate repos
        if 'model' not in self.config['repos'] or 'dataset' not in self.config['repos']:
            raise ValueError("Missing model or dataset in repos config")

        # Validate paths exist
        model_path = self.config['repos']['model']['path']
        dataset_path = self.config['repos']['dataset']['path']

        if not Path(model_path).exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not Path(dataset_path).exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Get configuration value"""
        return self.get(key)

    def __repr__(self) -> str:
        return f"Config({self.config_path})"
