"""RS-CL targets aligned with SpatialVLA's native Bridge training windows."""
import numpy as np
import torch

from cohort_planning.baseline_adapters import (
    BridgeSpatialVLAWindowTransform, _numpy, _steps, normalize_q99,
)


class BridgeRSCLWindowTransform(BridgeSpatialVLAWindowTransform):
    """Add current-state supervision without changing the native task inputs."""

    def prepare_episode(self, episode, trajectory):
        prepared = super().prepare_episode(episode, trajectory)
        steps = _steps(episode)
        states = np.stack([
            _numpy(step["observation"]["state"]) for step in steps
        ]).astype(np.float32)
        if states.ndim != 2 or states.shape[1] < 7:
            raise ValueError("RS-CL requires seven Bridge proprioceptive channels.")
        if "proprio" not in self.dataset_statistics:
            raise ValueError("RS-CL requires the dataset's proprio Q99 statistics.")
        # SpatialVLA discards the first raw step and the final no-action step.
        proprio = normalize_q99(states[1:-1, :7], self.dataset_statistics["proprio"])
        if len(proprio) != len(prepared["actions"]) or not np.isfinite(proprio).all():
            raise ValueError("RS-CL proprioception must be finite and aligned with actions.")
        prepared["proprio"] = proprio
        return prepared

    def __call__(self, episode, trajectory, timestep):
        window = super().__call__(episode, trajectory, timestep)
        window["proprio"] = torch.from_numpy(episode["proprio"][timestep].copy()).float()
        return window
