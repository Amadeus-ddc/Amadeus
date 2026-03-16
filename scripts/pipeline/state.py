"""Pipeline state management"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class PipelineState:
    """Manage pipeline execution state"""

    def __init__(self, state_file: str = "pipeline_state.json"):
        """Initialize state manager"""
        self.state_file = Path(state_file)
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or create new"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'pipeline_id': self._generate_id(),
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
                'stages': {},
                'errors': [],
                'checkpoints': {}
            }

    def _generate_id(self) -> str:
        """Generate unique pipeline ID"""
        return f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def start_stage(self, stage_name: str) -> None:
        """Mark stage as started"""
        if stage_name not in self.state['stages']:
            self.state['stages'][stage_name] = {}

        self.state['stages'][stage_name]['status'] = 'running'
        self.state['stages'][stage_name]['started_at'] = datetime.now().isoformat()
        self.state['status'] = 'running'
        self._save_state()

    def complete_stage(self, stage_name: str, output: Optional[Dict[str, Any]] = None) -> None:
        """Mark stage as completed"""
        if stage_name not in self.state['stages']:
            self.state['stages'][stage_name] = {}

        self.state['stages'][stage_name]['status'] = 'completed'
        self.state['stages'][stage_name]['completed_at'] = datetime.now().isoformat()
        if output:
            self.state['stages'][stage_name]['output'] = output

        self.state['checkpoints']['last_completed_stage'] = stage_name
        self._save_state()

    def fail_stage(self, stage_name: str, error: str) -> None:
        """Mark stage as failed"""
        if stage_name not in self.state['stages']:
            self.state['stages'][stage_name] = {}

        self.state['stages'][stage_name]['status'] = 'failed'
        self.state['stages'][stage_name]['error'] = error
        self.state['status'] = 'failed'
        self.state['errors'].append({
            'stage': stage_name,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self._save_state()

    def get_stage_status(self, stage_name: str) -> str:
        """Get status of a stage"""
        if stage_name in self.state['stages']:
            return self.state['stages'][stage_name].get('status', 'pending')
        return 'pending'

    def get_stage_output(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get output of a completed stage"""
        if stage_name in self.state['stages']:
            return self.state['stages'][stage_name].get('output')
        return None

    def can_resume_from(self, stage_name: str) -> bool:
        """Check if pipeline can resume from a stage"""
        if stage_name not in self.state['stages']:
            return False
        status = self.state['stages'][stage_name].get('status')
        return status in ['completed', 'failed']

    def _save_state(self) -> None:
        """Save state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def __repr__(self) -> str:
        return f"PipelineState(id={self.state['pipeline_id']}, status={self.state['status']})"
