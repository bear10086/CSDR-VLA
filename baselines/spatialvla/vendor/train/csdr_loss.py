"""Training-only CSDR loss adapted to SpatialVLA action-token representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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
    minimum_valid_length: int = 4


def distributed_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _check_equal_batch_size(tensor: torch.Tensor) -> None:
    if not distributed_is_ready():
        return
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)
    sizes = [int(size.item()) for size in sizes]
    if len(set(sizes)) != 1:
        raise RuntimeError(f"CSDR requires equal local batch sizes, got {sizes}.")


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


def prompt_token_signature(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Build a padding-independent, fixed-width signature for an instruction."""
    prompt_mask = (labels < 0) & attention_mask.bool()
    tokens = input_ids.long().masked_fill(~prompt_mask, 0)
    positions = torch.arange(1, tokens.shape[1] + 1, device=tokens.device, dtype=torch.long)
    return torch.stack(
        (
            prompt_mask.long().sum(dim=1),
            tokens.sum(dim=1),
            (tokens * positions.unsqueeze(0)).sum(dim=1),
            tokens.square().sum(dim=1),
        ),
        dim=1,
    ).detach()


def _pairwise_rms(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    difference = values[:, None, :] - values[None, :, :]
    return difference.square().mean(dim=-1).clamp_min(0.0).sqrt()


def _pairwise_cosine(features: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features.float(), dim=-1, eps=1e-6)
    return (1.0 - normalized @ normalized.transpose(0, 1)).clamp_min(0.0)


def _select(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    count: int,
    largest: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(max(int(count), 0), max(scores.shape[0] - 1, 0))
    if count == 0:
        shape = (scores.shape[0], 0)
        return (
            torch.empty(shape, dtype=torch.long, device=scores.device),
            torch.empty(shape, dtype=torch.bool, device=scores.device),
        )
    fill = -torch.inf if largest else torch.inf
    indices = torch.topk(scores.masked_fill(~candidates, fill), count, dim=1, largest=largest).indices
    return indices, candidates.gather(1, indices)


def prefix_cosine(hidden, horizon):
    """Action steps contain contiguous groups of action prediction tokens."""
    if hidden.shape[1] % horizon:
        raise ValueError('Action token count must be divisible by action horizon')
    per_step = hidden.shape[1] // horizon
    if per_step != 3:
        raise ValueError('SpatialVLA requires translation, rotation and gripper tokens per step')
    return torch.stack([
        _pairwise_cosine(hidden[:, :length * per_step].flatten(1))
        for length in range(1, horizon + 1)
    ], dim=-1)


def prefix_control(actions, cfg):
    terms = []
    denom = torch.arange(1, actions.shape[1] + 1, device=actions.device).float()
    for component, scale, weight in [
        (actions[..., :3], cfg.action_translation_scale, cfg.action_translation_weight),
        (actions[..., 3:6], cfg.action_rotation_scale, cfg.action_rotation_weight),
        (actions[..., 6:], cfg.action_gripper_scale, cfg.action_gripper_weight),
    ]:
        squared = (component[:, None].float() - component[None].float()).square().mean(-1)
        terms.append((squared.cumsum(-1) / denom).clamp_min(0).sqrt() * weight / scale)
    return sum(terms) / (cfg.action_translation_weight + cfg.action_rotation_weight + cfg.action_gripper_weight)


def ranking(near_distance, far_distance, valid, errors, common, horizon, cfg):
    violation = F.relu(near_distance + cfg.order_margin - far_distance)
    error_weight = (errors / cfg.action_error_scale).clamp(0, 1).detach()
    confusion = (violation.detach() / cfg.confusion_scale).clamp(0, 1)
    weights = error_weight[:, None, None] * confusion * valid.float()
    denominator = valid.float().sum().clamp_min(1)
    length_weight = (common.float() / horizon).sqrt().detach()
    reference = (violation * weights).sum() / denominator
    loss = (violation * weights * length_weight).sum() / denominator
    return loss, reference.detach(), violation, confusion


def apply_order_budget(loss, reference, task_loss, ratio, schedule):
    budget = task_loss.detach().abs() * ratio * schedule
    scale = torch.minimum(loss.detach().new_tensor(1.0),
                          budget / reference.detach().abs().clamp_min(1e-8))
    return scale * loss, scale


def spatialvla_csdr_loss(action_hidden_states, actions, context_signatures, action_errors, cfg,
                         valid_samples=None, valid_lengths=None):
    if action_hidden_states.ndim != 3 or actions.ndim != 3:
        raise ValueError('Expected [batch, tokens, hidden] and [batch, horizon, action]')
    if valid_lengths is None or valid_lengths.shape != (actions.shape[0],):
        raise ValueError('CSDR requires one action_valid_length per window')
    horizon = actions.shape[1]
    z = all_gather_with_grad(action_hidden_states)
    a = all_gather_without_grad(actions.float())
    context = all_gather_without_grad(context_signatures.long())
    errors = all_gather_without_grad(action_errors.float())
    lengths = all_gather_without_grad(valid_lengths.long())
    if bool(((lengths < 1) | (lengths > horizon)).any()):
        raise ValueError('Invalid real action length')
    if valid_samples is None:
        valid_samples = torch.ones(len(actions), dtype=torch.bool, device=actions.device)
    global_valid = all_gather_without_grad(valid_samples.bool())
    global_valid &= lengths >= cfg.minimum_valid_length
    n = len(z)
    zero = z.sum() * 0
    if n < 3:
        return dict(csdr_order_loss=zero, csdr_budget_reference=zero.detach(),
                    csdr_valid_triplet_count=zero.detach(), csdr_valid_anchor_fraction=zero.detach())
    representations = prefix_cosine(z, horizon)
    controls = prefix_control(a, cfg)
    pair_h = torch.minimum(lengths[:, None], lengths[None, :])
    control = controls.gather(2, (pair_h - 1)[..., None]).squeeze(-1)
    distance = representations.gather(2, (pair_h - 1)[..., None]).squeeze(-1)
    candidates = (context[:, None, :] == context[None, :, :]).all(-1)
    candidates &= ~torch.eye(n, device=z.device, dtype=torch.bool)
    candidates &= global_valid[:, None] & global_valid[None, :]
    ni, nv = _select(control, candidates & (control <= cfg.near_max_control_distance), cfg.k_near, False)
    membership = torch.zeros_like(candidates).scatter(1, ni, nv)
    nv &= membership.T.gather(1, ni)
    fi, fv = _select(distance.detach(), candidates & (control >= cfg.far_min_control_distance), cfg.k_far, False)
    common = torch.minimum(lengths[:, None, None],
                           torch.minimum(lengths[ni][:, :, None], lengths[fi][:, None, :]))
    anchor = torch.arange(n, device=z.device)[:, None, None]
    hi = common - 1
    dn = controls[anchor, ni[:, :, None], hi]
    df = controls[anchor, fi[:, None, :], hi]
    valid = nv[:, :, None] & fv[:, None, :]
    valid &= common >= cfg.minimum_valid_length
    valid &= (dn <= cfg.near_max_control_distance) & (df >= cfg.far_min_control_distance)
    valid &= df >= dn + cfg.minimum_control_gap
    near_repr = representations[anchor, ni[:, :, None], hi]
    far_repr = representations[anchor, fi[:, None, :], hi]
    loss, reference, violation, confusion = ranking(near_repr, far_repr, valid, errors, common, horizon, cfg)
    count = valid.float().sum()
    denominator = count.clamp_min(1)
    return {
        'csdr_order_loss': loss,
        'csdr_budget_reference': reference,
        'csdr_valid_triplet_count': count.detach(),
        'csdr_valid_anchor_fraction': (valid.flatten(1).any(-1) & global_valid).float().mean().detach(),
        'csdr_triplet_active_fraction': (((violation > 0) & valid).float().sum() / denominator).detach(),
        'csdr_mean_near_count': nv.float().sum(1).mean().detach(),
        'csdr_mean_far_count': fv.float().sum(1).mean().detach(),
        'csdr_mean_near_control_distance': (dn.masked_fill(~valid, 0).sum() / denominator).detach(),
        'csdr_mean_far_control_distance': (df.masked_fill(~valid, 0).sum() / denominator).detach(),
        'csdr_mean_common_prefix': (common.masked_fill(~valid, 0).sum() / denominator).detach(),
        'csdr_mean_error_weight': (errors / cfg.action_error_scale).clamp(0, 1).mean().detach(),
        'csdr_mean_confusion_weight': ((confusion * valid).sum() / denominator).detach(),
        'csdr_global_unique_prompts': torch.unique(context, dim=0).shape[0] + zero.detach(),
        'csdr_prefix_aware': zero.detach() + 1,
        'csdr_short_window_fraction': (lengths < horizon).float().mean().detach(),
        'csdr_short_triplet_fraction': (((common < horizon) & valid).float().sum() / denominator).detach(),
        'csdr_excluded_short_window_fraction': (lengths < cfg.minimum_valid_length).float().mean().detach(),
    }
