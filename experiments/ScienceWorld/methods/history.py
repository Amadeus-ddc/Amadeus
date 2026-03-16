"""History-based streaming baseline for ScienceWorld.

Stores (goal, success, progress, key_steps) for each completed episode
and injects the most recent K episodes into the prompt as experience context.
"""

from typing import List
from .base import MemoryModule


class HistoryMemory(MemoryModule):
    """Streaming baseline: concatenate recent task histories into prompt context."""

    def __init__(self, history_window: int = 5, key_steps: int = 5, **kwargs):
        """
        Args:
            history_window: Number of recent episodes to include in context.
            key_steps: Max trajectory steps to include as key steps per episode.
        """
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
        """Return the most recent history_window episodes as formatted context."""
        if not self._buffer:
            return ""

        recent = self._buffer[-self.history_window:]
        parts = ["Past task experiences (most recent last):"]
        for i, ep in enumerate(recent, 1):
            status = "SUCCESS" if ep["success"] else "FAILURE"
            parts.append(f"\n[Experience {i}]")
            parts.append(f"Task: {ep['goal']}")
            parts.append(f"Outcome: {status} (progress: {ep['progress']:.0%})")
            if ep["key_steps"]:
                parts.append("Key steps taken:")
                for step in ep["key_steps"]:
                    parts.append(f"  - Action: {step['action']} → Obs: {step['obs'][:80]}...")
        return "\n".join(parts)

    def evolve(
        self,
        task_name: str,
        goal: str,
        trajectory: List[dict],
        success: bool,
        progress: float,
        episode_idx: int,
    ) -> None:
        """Store episode into history buffer."""
        # Extract (action, observation) pairs from trajectory dicts
        action_obs_pairs = []
        pending_action = None
        for item in trajectory:
            if "Action" in item:
                pending_action = item["Action"]
            elif "Observation" in item and pending_action is not None:
                action_obs_pairs.append((pending_action, item["Observation"]))
                pending_action = None

        # Evenly sample up to key_steps steps
        key_steps = []
        if action_obs_pairs:
            n = len(action_obs_pairs)
            if n <= self.key_steps:
                sampled = action_obs_pairs
            else:
                indices = [
                    int(i * (n - 1) / (self.key_steps - 1))
                    for i in range(self.key_steps)
                ]
                sampled = [action_obs_pairs[i] for i in indices]
            for action, obs in sampled:
                key_steps.append({"action": action, "obs": obs})

        self._buffer.append({
            "task_name": task_name,
            "goal": goal,
            "success": success,
            "progress": progress,
            "episode_idx": episode_idx,
            "key_steps": key_steps,
        })
