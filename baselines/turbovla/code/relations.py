"""Training-only CSDR loss adapted to SpatialVLA action-token representations."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.nn.functional as F

@dataclass
class CSDRConfig:
    action_translation_scale: float = 0.502
    action_rotation_scale: float = 0.517
    action_gripper_scale: float = 0.676
    action_translation_weight: float = 1.0
    action_rotation_weight: float = 0.5
    action_gripper_weight: float = 0.25
    near_max_control_distance: float = 0.8
    far_min_control_distance: float = 1.0
    minimum_control_gap: float = 0.2
    order_margin: float = 0.1
    k_near: int = 4
    k_far: int = 8
    action_error_scale: float = 0.5
    confusion_scale: float = 0.1

def distributed_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)

def _check_equal_batch_size(tensor: torch.Tensor) -> None:
    if not distributed_is_ready():
        return
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)
    sizes = [int(size.item()) for size in sizes]
    if len(set(sizes)) != 1:
        raise RuntimeError(f'CSDR requires equal local batch sizes, got {sizes}.')

def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not distributed_is_ready():
        return tensor
    _check_equal_batch_size(tensor)
    from torch.distributed.nn.functional import all_gather
    return torch.cat(all_gather(tensor), dim=0)

def all_gather_without_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not distributed_is_ready():
        return tensor.detach()
    outputs = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(outputs, tensor.detach())
    return torch.cat(outputs, dim=0)

def _pairwise_cosine(features: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features.float(), dim=-1, eps=1e-06)
    return (1.0 - normalized @ normalized.transpose(0, 1)).clamp_min(0.0)

def _select(scores: torch.Tensor, candidates: torch.Tensor, count: int, largest: bool) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(max(int(count), 0), max(scores.shape[0] - 1, 0))
    if count == 0:
        shape = (scores.shape[0], 0)
        return (torch.empty(shape, dtype=torch.long, device=scores.device), torch.empty(shape, dtype=torch.bool, device=scores.device))
    fill = -torch.inf if largest else torch.inf
    indices = torch.topk(scores.masked_fill(~candidates, fill), count, dim=1, largest=largest).indices
    return (indices, candidates.gather(1, indices))
