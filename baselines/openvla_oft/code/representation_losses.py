"""Training-only representation regularizers for OpenVLA-OFT.

The losses in this file add no parameters and do not change inference.  They are
kept independent from the training script so their distributed behavior can be
tested without loading OpenVLA.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import torch
import torch.distributed as dist
import torch.nn.functional as F

def distributed_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)

def _check_equal_local_batch_size(tensor: torch.Tensor) -> None:
    """Fail clearly instead of hanging when ranks enter all_gather with different B."""
    if not distributed_is_ready():
        return
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)
    sizes = [int(item.item()) for item in sizes]
    if len(set(sizes)) != 1:
        raise RuntimeError(f'Representation all_gather requires equal local batch sizes, got {sizes}.')

def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Gather the first dimension across ranks while preserving autograd."""
    if not distributed_is_ready():
        return tensor
    _check_equal_local_batch_size(tensor)
    from torch.distributed.nn.functional import all_gather
    return torch.cat(all_gather(tensor), dim=0)

def all_gather_without_grad_many(tensors: Sequence[torch.Tensor], *, batch_size_already_checked: bool=False) -> tuple[torch.Tensor, ...]:
    """Coalesce small per-sample metadata into one exact, non-autograd collective."""
    if not tensors:
        return ()
    batch_size = tensors[0].shape[0]
    if batch_size < 1 or any((tensor.shape[0] != batch_size for tensor in tensors)):
        raise ValueError('All gathered metadata tensors must have the same non-zero first dimension.')
    if not distributed_is_ready():
        return tuple((tensor.detach() for tensor in tensors))
    if not batch_size_already_checked:
        _check_equal_local_batch_size(tensors[0])
    widths = [tensor.numel() // batch_size for tensor in tensors]
    shapes = [tensor.shape[1:] for tensor in tensors]
    dtypes = [tensor.dtype for tensor in tensors]
    packed = torch.cat([tensor.detach().reshape(batch_size, width).to(torch.float64) for tensor, width in zip(tensors, widths)], dim=1)
    outputs = [torch.empty_like(packed) for _ in range(dist.get_world_size())]
    dist.all_gather(outputs, packed)
    global_packed = torch.cat(outputs, dim=0)
    chunks = global_packed.split(widths, dim=1)
    return tuple((chunk.reshape(global_packed.shape[0], *shape).to(dtype) for chunk, shape, dtype in zip(chunks, shapes, dtypes)))

def flatten_action_queries(action_hidden_states: torch.Tensor) -> torch.Tensor:
    if action_hidden_states.ndim != 3:
        raise ValueError(f'Expected action_hidden_states with shape [batch, action_tokens, hidden_dim], got {tuple(action_hidden_states.shape)}.')
    return action_hidden_states.reshape(action_hidden_states.shape[0], -1)

def pairwise_cosine_distance(features: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features.float(), dim=-1, eps=1e-06)
    return (1.0 - normalized @ normalized.transpose(0, 1)).clamp_min(0.0)

def _pairwise_rms_distance(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.shape[-1] == 0:
        return values.new_zeros((values.shape[0], values.shape[0]))
    difference = values[:, None, :] - values[None, :, :]
    return difference.square().mean(dim=-1).clamp_min(0.0).sqrt()

def prompt_token_signature(input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return a fixed-size, padding-independent signature of prompt tokens.

    The four integer checksums make accidental equality extremely unlikely for
    the small instruction vocabulary used by LIBERO. They also avoid gathering
    variable-length token sequences across DDP ranks.
    """
    if input_ids.shape != labels.shape or input_ids.shape != attention_mask.shape:
        raise ValueError('input_ids, labels, and attention_mask must have identical shapes.')
    token_ids = input_ids.long()
    prompt_mask = (labels < 0) & attention_mask.bool()
    positions = torch.arange(1, token_ids.shape[1] + 1, device=token_ids.device, dtype=torch.long)
    masked_tokens = token_ids.masked_fill(~prompt_mask, 0)
    token_count = prompt_mask.long().sum(dim=1)
    token_sum = masked_tokens.sum(dim=1)
    weighted_sum = (masked_tokens * positions.unsqueeze(0)).sum(dim=1)
    squared_sum = masked_tokens.square().sum(dim=1)
    return torch.stack((token_count, token_sum, weighted_sum, squared_sum), dim=1).detach()

def linear_warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear warmup followed by a constant factor of one."""
    if warmup_steps < 0:
        raise ValueError('warmup_steps must be non-negative.')
    step = max(int(step), 0)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    return 1.0

def _normalize_per_dataset_pair_scale(distance: torch.Tensor, dataset_ids: torch.Tensor, scale_table: torch.Tensor, column: int, name: str) -> torch.Tensor:
    sample_scales = scale_table[:, column][dataset_ids]
    pair_scales = torch.sqrt(sample_scales[:, None] * sample_scales[None, :])
    return distance / pair_scales.clamp_min(1e-08)

def _validate_per_dataset_scale_inputs(dataset_ids: torch.Tensor, scale_table: torch.Tensor, batch_size: int) -> None:
    """Validate suite scales once instead of forcing 18 GPU synchronizations per micro-batch."""
    if scale_table.ndim != 2 or scale_table.shape[1] != 6:
        raise ValueError('scale_table must have shape [num_datasets, 6].')
    if dataset_ids.ndim != 1 or dataset_ids.shape[0] != batch_size:
        raise ValueError('dataset_ids must contain one ID per sample.')
    if int(dataset_ids.min().item()) < 0 or int(dataset_ids.max().item()) >= scale_table.shape[0]:
        raise ValueError('dataset_ids contain an index outside scale_table.')
    if bool((~torch.isfinite(scale_table) | (scale_table <= 0.0)).any().item()):
        raise ValueError('All per-dataset scale values must be finite and positive.')

def _per_dataset_state_distance(proprio: torch.Tensor, dataset_ids: torch.Tensor, scale_table: torch.Tensor, cfg: CSDRConfig) -> torch.Tensor:
    position = _normalize_per_dataset_pair_scale(_pairwise_rms_distance(proprio[:, :3]), dataset_ids, scale_table, 0, 'state_position_scale')
    rotation = _normalize_per_dataset_pair_scale(_pairwise_rms_distance(proprio[:, 3:6]), dataset_ids, scale_table, 1, 'state_rotation_scale')
    gripper = _normalize_per_dataset_pair_scale(_pairwise_rms_distance(proprio[:, 6:]), dataset_ids, scale_table, 2, 'state_gripper_scale')
    weight_sum = cfg.state_position_weight + cfg.state_rotation_weight + cfg.state_gripper_weight
    if weight_sum <= 0.0:
        return position.new_zeros(position.shape)
    return (cfg.state_position_weight * position + cfg.state_rotation_weight * rotation + cfg.state_gripper_weight * gripper) / weight_sum

def _per_dataset_action_distance(actions: torch.Tensor, dataset_ids: torch.Tensor, scale_table: torch.Tensor, cfg: CSDRConfig) -> torch.Tensor:
    translation = _normalize_per_dataset_pair_scale(_pairwise_rms_distance(actions[..., :3].flatten(1)), dataset_ids, scale_table, 3, 'action_translation_scale')
    rotation = _normalize_per_dataset_pair_scale(_pairwise_rms_distance(actions[..., 3:6].flatten(1)), dataset_ids, scale_table, 4, 'action_rotation_scale')
    gripper = _normalize_per_dataset_pair_scale(_pairwise_rms_distance(actions[..., 6:].flatten(1)), dataset_ids, scale_table, 5, 'action_gripper_scale')
    weight_sum = cfg.action_translation_weight + cfg.action_rotation_weight + cfg.action_gripper_weight
    if weight_sum <= 0.0:
        raise ValueError('At least one dense action distance weight must be positive.')
    return (cfg.action_translation_weight * translation + cfg.action_rotation_weight * rotation + cfg.action_gripper_weight * gripper) / weight_sum

def _select_control_neighbors(control_distance: torch.Tensor, valid_pairs: torch.Tensor, count: int, largest: bool) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = control_distance.shape[0]
    count = min(max(int(count), 0), max(batch_size - 1, 0))
    if count == 0:
        empty_indices = torch.empty((batch_size, 0), dtype=torch.long, device=control_distance.device)
        empty_valid = torch.empty((batch_size, 0), dtype=torch.bool, device=control_distance.device)
        return (empty_indices, empty_valid)
    fill_value = -torch.inf if largest else torch.inf
    scores = control_distance.masked_fill(~valid_pairs, fill_value)
    _, indices = torch.topk(scores, k=count, dim=1, largest=largest, sorted=False)
    selected_valid = valid_pairs.gather(1, indices)
    return (indices, selected_valid)

@dataclass
class CSDRConfig:
    state_weight: float = 0.2
    action_weight: float = 0.8
    state_position_weight: float = 1.0
    state_rotation_weight: float = 0.5
    state_gripper_weight: float = 0.25
    action_translation_weight: float = 1.0
    action_rotation_weight: float = 0.5
    action_gripper_weight: float = 0.25
    k_near: int = 4
    k_far: int = 8
    near_max_control_distance: float = 0.8
    far_min_control_distance: float = 1.0
    minimum_control_gap: float = 0.2
    order_margin: float = 0.1
    local_target_max: float = 0.2
    state_position_scale: float = 1.0
    state_rotation_scale: float = 1.0
    state_gripper_scale: float = 1.0
    action_translation_scale: float = 1.0
    action_rotation_scale: float = 1.0
    action_gripper_scale: float = 1.0
    local_max_control_distance: float = 0.4
    action_error_scale: float = 0.02
    confusion_scale: float = 0.1

def csdr_loss(action_hidden_states: torch.Tensor, proprio: torch.Tensor, actions: torch.Tensor, context_signatures: torch.Tensor, dataset_ids: torch.Tensor, scale_table: torch.Tensor, action_errors: torch.Tensor, cfg: CSDRConfig, valid_samples: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
    """Strict same-prompt ordering focused on severe representation confusion."""
    local_z = flatten_action_queries(action_hidden_states)
    if valid_samples is None:
        valid_samples = torch.ones(local_z.shape[0], dtype=torch.bool, device=local_z.device)
    global_z = all_gather_with_grad(local_z)
    global_proprio, global_actions, global_context, global_dataset_ids, global_errors, global_valid = all_gather_without_grad_many((proprio.detach().float(), actions.detach().float(), context_signatures.detach().long(), dataset_ids.detach().long(), action_errors.detach().float(), valid_samples.detach().bool()), batch_size_already_checked=True)
    zero = global_z.sum() * 0.0
    metric_names = ('csdr_order_loss', 'csdr_triplet_active_fraction', 'csdr_valid_anchor_fraction', 'csdr_mean_near_count', 'csdr_mean_far_count', 'csdr_same_context_far_fraction', 'csdr_mean_error_weight', 'csdr_error_weight_saturation_fraction', 'csdr_mean_confusion_weight', 'csdr_strong_confusion_fraction', 'csdr_global_unique_contexts')
    if global_z.shape[0] < 2 or int(global_valid.sum().item()) < 2:
        return {name: zero if name == 'csdr_order_loss' else zero.detach() for name in metric_names}
    if cfg.confusion_scale <= 0.0:
        raise ValueError('CSDR confusion_scale must be positive.')
    batch_size = global_z.shape[0]
    scale_table = scale_table.to(device=global_z.device, dtype=torch.float32)
    _validate_per_dataset_scale_inputs(global_dataset_ids, scale_table, batch_size)
    repr_distance = pairwise_cosine_distance(global_z)
    state_distance = _per_dataset_state_distance(global_proprio, global_dataset_ids, scale_table, cfg)
    action_distance = _per_dataset_action_distance(global_actions, global_dataset_ids, scale_table, cfg)
    source_weight_sum = cfg.state_weight + cfg.action_weight
    if source_weight_sum <= 0.0:
        raise ValueError('At least one of state_weight and action_weight must be positive.')
    control_distance = (cfg.state_weight * state_distance + cfg.action_weight * action_distance) / source_weight_sum
    off_diagonal = ~torch.eye(batch_size, device=global_z.device, dtype=torch.bool)
    valid_pairs = global_valid[:, None] & global_valid[None, :] & off_diagonal
    same_context = (global_context[:, None, :] == global_context[None, :, :]).all(dim=-1)
    same_dataset = global_dataset_ids[:, None] == global_dataset_ids[None, :]
    near_candidates = valid_pairs & same_context & same_dataset & (control_distance <= cfg.near_max_control_distance)
    near_indices, near_valid = _select_control_neighbors(control_distance, near_candidates, cfg.k_near, largest=False)
    near_membership = torch.zeros_like(valid_pairs)
    near_membership.scatter_(1, near_indices, near_valid)
    mutual_membership = near_membership & near_membership.transpose(0, 1)
    near_valid = near_valid & mutual_membership.gather(1, near_indices)
    far_candidates = valid_pairs & same_context & same_dataset & (control_distance >= cfg.far_min_control_distance)
    far_selection_score = repr_distance.detach()
    far_indices, far_valid = _select_control_neighbors(far_selection_score, far_candidates, cfg.k_far, largest=False)
    near_repr = repr_distance.gather(1, near_indices)
    far_repr = repr_distance.gather(1, far_indices)
    near_control = control_distance.gather(1, near_indices)
    far_control = control_distance.gather(1, far_indices)
    far_same_context = same_context.gather(1, far_indices) & far_valid
    triplet_valid = near_valid[:, :, None] & far_valid[:, None, :]
    triplet_valid = triplet_valid & (far_control[:, None, :] >= near_control[:, :, None] + cfg.minimum_control_gap)
    order_violation = F.relu(near_repr[:, :, None] + cfg.order_margin - far_repr[:, None, :])
    error_weights = (global_errors / float(cfg.action_error_scale)).clamp(0.0, 1.0)
    confusion_weights = (order_violation.detach() / float(cfg.confusion_scale)).clamp(0.0, 1.0)
    triplet_weights = error_weights[:, None, None] * confusion_weights * triplet_valid.float()
    triplet_denominator = triplet_valid.float().sum()
    if bool((triplet_denominator > 0).item()):
        order_loss = (order_violation * triplet_weights).sum() / triplet_denominator
        triplet_active = ((order_violation > 0) & triplet_valid).float().sum().detach() / triplet_denominator.detach()
        mean_confusion_weight = (confusion_weights * triplet_valid.float()).sum().detach() / triplet_denominator.detach()
        strong_confusion_fraction = ((confusion_weights >= 0.5) & triplet_valid).float().sum().detach() / triplet_denominator.detach()
    else:
        order_loss = zero
        triplet_active = zero.detach()
        mean_confusion_weight = zero.detach()
        strong_confusion_fraction = zero.detach()
    valid_sample_count = global_valid.float().sum().clamp_min(1.0)
    near_denominator = near_valid.float().sum()
    far_denominator = far_valid.float().sum()
    valid_anchors = triplet_valid.flatten(1).any(dim=1) & global_valid
    valid_error_weights = error_weights[global_valid]
    return {'csdr_order_loss': order_loss, 'csdr_triplet_active_fraction': triplet_active, 'csdr_valid_anchor_fraction': valid_anchors.float().sum().detach() / valid_sample_count.detach(), 'csdr_mean_near_count': near_valid.float().sum(dim=1).sum().detach() / valid_sample_count.detach(), 'csdr_mean_far_count': far_valid.float().sum(dim=1).sum().detach() / valid_sample_count.detach(), 'csdr_same_context_far_fraction': far_same_context.float().sum().detach() / far_denominator.detach().clamp_min(1.0), 'csdr_mean_error_weight': valid_error_weights.mean().detach(), 'csdr_error_weight_saturation_fraction': (valid_error_weights >= 1.0).float().mean().detach(), 'csdr_mean_confusion_weight': mean_confusion_weight, 'csdr_strong_confusion_fraction': strong_confusion_fraction, 'csdr_global_unique_contexts': zero.detach().new_tensor(float(torch.unique(global_context[global_valid], dim=0).shape[0]))}
