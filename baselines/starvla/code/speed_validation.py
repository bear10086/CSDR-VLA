"""Strict equality fingerprints for automatic speed experiments."""
import hashlib
import pickle
import torch

def tree_digest(value):
    digest = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            digest.update(str((tuple(x.shape), str(x.dtype))).encode())
            raw = x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
        elif isinstance(x, dict):
            for k, v in x.items():
                digest.update(str(k).encode())
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        else:
            digest.update(pickle.dumps(x, protocol=4))
    visit(value)
    return digest.hexdigest()

def model_digest(model, gradients=False):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        value = param.grad if gradients else param
        digest.update(name.encode())
        digest.update(tree_digest(value).encode())
    return digest.hexdigest()
