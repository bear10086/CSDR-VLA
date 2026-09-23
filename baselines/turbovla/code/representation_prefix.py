"""Cosine distance over flattened action-query prefixes, with no padded-token gradient."""
import torch


def prefix_cosine(hidden):
    # Native interaction is FP32; explicitly avoid BF16 autocast reducing prefix dot precision.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        z = hidden.float()
        dot = torch.einsum('ihd,jhd->ijh', z, z).cumsum(-1)
        norm = z.square().sum(-1).cumsum(-1).clamp_min(1e-12).sqrt()
        return (1 - dot / (norm[:, None] * norm[None, :])).clamp_min(0)
