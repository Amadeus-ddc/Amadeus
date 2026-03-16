"""History-based streaming baseline for ALFWorld.

Aligned with Evo-Memory paper: stores (Goal, Trajectory, Correctness) for each
completed episode and injects the most recent K episodes into the prompt.
"""

from typing import List, Tuple
from .base import MemoryModule


class HistoryMemory(MemoryModule):
    """Streaming baseline: inject recent task histories into prompt context."""

    def __init__(self, history_window: int = 5, key_steps: int = 5, **kwargs):
        self.history_window = history_window
        self.key_steps = key_steps
        self._buffer: List[dict] = []

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument(
            "--history_window", type=int, default=5,
            help="History: number of recent episodes to include in context (default: 5)"
        )
        parser.add_argument(
            "--history_key_steps", type=int, default=5,
            help="History: max key steps to include per episode (default: 5)"
        )

    def search(self, query: str) -> str:
        """Return the most recent history_window episodes formatted as paper template."""
        if not self._buffer:
            return ""

        recent = self._buffer[-self.history_window:]
        parts = []
        for i, ep in enumerate(recent, 1):
            correctness = "success" if ep["success"] else "failure"
            parts.append(f"[Experience #{i}]")
            parts.append(f"Goal: {ep['task_description']}")
            if ep["trajectory_str"]:
                parts.append(f"Trajectory: {ep['trajectory_str']}")
            parts.append(f"Correctness: {correctness}")
            parts.append("")
        return "\n".join(parts)

    def evolve(
        self,
        task_type: str,
        task_description: str,
        trajectory: List[Tuple[str, str]],
        success: bool,
        episode_idx: int,
    ) -> None:
        """Store episode into history buffer using paper format."""
        # Sample up to key_steps (obs, action) pairs evenly
        traj_parts = []
        if trajectory:
            step_count = len(trajectory)
            if step_count <= self.key_steps:
                sampled = trajectory
            else:
                indices = [
                    int(i * (step_count - 1) / (self.key_steps - 1))
                    for i in range(self.key_steps)
                ]
                sampled = [trajectory[i] for i in indices]
            for obs, action in sampled:
                traj_parts.append(f"Action: {action}")

        self._buffer.append({
            "task_type": task_type,
            "task_description": task_description,
            "success": success,
            "episode_idx": episode_idx,
            "trajectory_str": " -> ".join(traj_parts),
        })
