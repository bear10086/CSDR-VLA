"""RS-CL-inspired auxiliary loss for SpatialVLA.

The original RS-CL appends a learnable summary token to the VLM output and
uses view-cutoff augmentation. Bridge SpatialVLA provides one image view and
its native action path is already defined, so this reproduction uses masked
mean pooling of the final hidden sequence, a learnable summary offset, a
two-layer projector, and feature dropout as the representation-level view.
The proprioceptive soft-label objective follows the published equations.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from train.csdr_loss import all_gather_with_grad, all_gather_without_grad


def _base_model(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def rscl_embedding_loss(z, z_aug, proprio, beta=1.0, temperature=0.2):
    """Compute the published weighted InfoNCE objective from two projected views."""
    z = all_gather_with_grad(z)
    z_aug = all_gather_with_grad(z_aug)
    q = all_gather_without_grad(proprio.float())
    distances = torch.cdist(q, q, p=2)
    soft_targets = F.softmax(-distances / float(beta), dim=-1)
    logits = z @ z_aug.transpose(0, 1) / float(temperature)
    loss = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    return {
        "rscl_loss": loss,
        "rscl_mean_state_distance": distances.mean().detach(),
        "rscl_embedding_norm": z.norm(dim=-1).mean().detach(),
        "rscl_global_batch": loss.detach().new_tensor(float(z.shape[0])),
    }


def rscl_loss(hidden_states, attention_mask, proprio, model, beta=1.0, temperature=0.2):
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden states [B,L,D], got {tuple(hidden_states.shape)}")
    base = _base_model(model)
    required = ("rscl_summary_token", "rscl_adapter", "rscl_projector", "rscl_dropout")
    if not all(hasattr(base, name) for name in required):
        raise RuntimeError("RS-CL modules are missing from the model.")

    mask = attention_mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    pooled = (hidden_states * mask).sum(dim=1) / denom
    pooled = pooled.float()
    summary = base.rscl_adapter(pooled + base.rscl_summary_token.float().unsqueeze(0))
    summary_aug = base.rscl_dropout(summary)
    projected = base.rscl_projector(torch.cat([summary, summary_aug], dim=0))
    z, z_aug = projected.chunk(2, dim=0)
    z = F.normalize(z, dim=-1, eps=1e-6)
    z_aug = F.normalize(z_aug, dim=-1, eps=1e-6)
    return rscl_embedding_loss(z, z_aug, proprio, beta=beta, temperature=temperature)
