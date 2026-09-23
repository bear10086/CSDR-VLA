"""Training-only ordering on genuine, common future-action prefixes."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from relations import (CSDRConfig as BaseConfig, all_gather_with_grad,
    all_gather_without_grad, _pairwise_cosine, _select)


@dataclass
class CSDRConfig(BaseConfig):
    minimum_valid_length: int = 4
    # Ordered rows for lengths 1..12, columns translation/rotation/gripper.
    prefix_scales: tuple | None = None


def prefix_control(actions, cfg):
    """[B,B,H], each entry uses only the first h real-or-candidate actions."""
    horizon = actions.shape[1]
    default = [cfg.action_translation_scale, cfg.action_rotation_scale, cfg.action_gripper_scale]
    scales = actions.new_tensor(cfg.prefix_scales or [default] * horizon).float()
    assert scales.shape == (horizon, 3)
    denom = torch.arange(1, horizon + 1, device=actions.device).float()
    weights = [cfg.action_translation_weight, cfg.action_rotation_weight, cfg.action_gripper_weight]
    terms = []
    for c, part in enumerate((slice(0, 3), slice(3, 6), slice(6, 7))):
        x = actions[..., part].float()
        mse = (x[:, None] - x[None, :]).square().mean(-1)
        rms = (mse.cumsum(-1) / denom).clamp_min(0).sqrt()
        terms.append(weights[c] * rms / scales[:, c].clamp_min(1e-6))
    return sum(terms) / sum(weights)


def common_triplet_controls(prefix, lengths, near, far, cfg):
    """Revalidate BOTH edges at the SAME horizon, not at two pairwise horizons."""
    common = torch.minimum(torch.minimum(lengths[:, None, None], lengths[near][:, :, None]),
                           lengths[far][:, None, :])
    anchor = torch.arange(len(lengths), device=lengths.device)[:, None, None]
    h = (common - 1).clamp_min(0)
    dn = prefix[anchor, near[:, :, None], h]
    df = prefix[anchor, far[:, None, :], h]
    legal = ((common >= cfg.minimum_valid_length) & (dn <= cfg.near_max_control_distance)
             & (df >= cfg.far_min_control_distance) & (df >= dn + cfg.minimum_control_gap))
    return dn, df, common, legal


def spatialvla_csdr_loss(action_hidden_states, actions, context_signatures, action_errors,
                        cfg, valid_samples=None, compute_metrics=True, action_valid_lengths=None):
    b, horizon = actions.shape[:2]
    assert action_hidden_states.ndim == 3 and action_hidden_states.shape[:2] == (b, horizon)
    if action_valid_lengths is None:
        raise ValueError('True future-action lengths are mandatory; never infer them from padded values.')
    z = all_gather_with_grad(action_hidden_states.reshape(b, -1))
    a = all_gather_without_grad(actions.float())
    prompt = all_gather_without_grad(context_signatures.long())
    errors = all_gather_without_grad(action_errors.float())
    valid = all_gather_without_grad(torch.ones(b, device=z.device, dtype=torch.bool)
                                   if valid_samples is None else valid_samples.bool())
    lengths = all_gather_without_grad(action_valid_lengths.long()).clamp(0, horizon)
    valid = valid & (lengths >= cfg.minimum_valid_length)
    from representation_prefix import prefix_cosine
    repr_prefix = prefix_cosine(z.reshape(len(z), horizon, -1))
    pair_h_repr = torch.minimum(lengths[:, None], lengths[None, :])
    distance = repr_prefix.gather(2, (pair_h_repr - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    prefix = prefix_control(a, cfg)
    pair_h = torch.minimum(lengths[:, None], lengths[None, :])
    control = prefix.gather(2, (pair_h - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    candidates = ((prompt[:, None] == prompt[None, :]).all(-1)
                  & ~torch.eye(len(z), device=z.device, dtype=torch.bool)
                  & valid[:, None] & valid[None, :])
    near, nv = _select(control, candidates & (control <= cfg.near_max_control_distance), cfg.k_near, False)
    members = torch.zeros_like(candidates).scatter(1, near, nv)
    nv = nv & members.T.gather(1, near)
    far, fv = _select(distance.detach(), candidates & (control >= cfg.far_min_control_distance), cfg.k_far, False)
    dn, df, common, legal = common_triplet_controls(prefix, lengths, near, far, cfg)
    proposed = nv[:, :, None] & fv[:, None, :]
    triplets = proposed & legal
    anchor = torch.arange(len(z), device=z.device)[:, None, None]
    hi = (common - 1).clamp_min(0)
    dn_repr = repr_prefix[anchor, near[:, :, None], hi]
    df_repr = repr_prefix[anchor, far[:, None, :], hi]
    violation = F.relu(dn_repr + cfg.order_margin - df_repr)
    error_weight = (errors / cfg.action_error_scale).clamp(0, 1)
    confusion = (violation.detach() / cfg.confusion_scale).clamp(0, 1)
    weights = error_weight[:, None, None] * confusion * triplets.float()
    length_weight = (common.float() / horizon).sqrt().detach()
    denom = triplets.float().sum().clamp_min(1)
    reference = (violation * weights).sum() / denom
    loss = (violation * weights * length_weight).sum() / denom
    # Use UNWEIGHTED reference for the scalar budget: otherwise it cancels length downweighting.
    result = {'csdr_order_loss': loss, 'csdr_budget_reference': reference.detach()}
    if not compute_metrics:
        return result
    anchors = triplets.flatten(1).any(1) & valid
    tail = valid & (lengths < horizon)
    result.update({
        'csdr_valid_anchor_fraction': anchors.float().mean(),
        'csdr_triplet_active_fraction': ((violation > 0) & triplets).sum() / denom,
        'csdr_mean_near_count': nv.float().sum(1).mean(),
        'csdr_mean_far_count': fv.float().sum(1).mean(),
        'csdr_mean_near_control_distance': dn.masked_fill(~triplets, 0).sum() / denom,
        'csdr_mean_far_control_distance': df.masked_fill(~triplets, 0).sum() / denom,
        'csdr_mean_error_weight': error_weight.mean(),
        'csdr_mean_confusion_weight': (confusion * triplets).sum() / denom,
        'csdr_global_unique_prompts': z.new_tensor(torch.unique(prompt, dim=0).shape[0]),
        'csdr_tail_eligible_count': tail.float().sum(),
        'csdr_tail_valid_anchor_count': (tail & anchors).float().sum(),
        'csdr_tail_anchor_fraction': (tail & anchors).sum() / tail.sum().clamp_min(1),
        'csdr_triplet_count': triplets.float().sum(),
        'csdr_common_horizon': (common * triplets).sum() / denom,
        'csdr_length_weight': (length_weight * triplets).sum() / denom,
        'csdr_common_rejection_fraction': (proposed & ~legal).sum() / proposed.sum().clamp_min(1),
    })
    return {k: v if k == 'csdr_order_loss' else v.detach() for k, v in result.items()}
