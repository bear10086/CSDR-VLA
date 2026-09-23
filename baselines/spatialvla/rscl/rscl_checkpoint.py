"""Save auxiliary trainables separately from inference-only PEFT adapters."""
import json
from pathlib import Path
import torch
from transformers import TrainerCallback


def auxiliary_parameters(model):
    return {name: value for name, value in model.named_parameters() if "rscl_" in name}


def save_auxiliary(model, checkpoint):
    values = {name: p.detach().cpu() for name, p in auxiliary_parameters(model).items()}
    path = Path(checkpoint) / "rscl_auxiliary.pt"
    torch.save(values, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def load_auxiliary(model, checkpoint):
    path = Path(checkpoint) / "rscl_auxiliary.pt"
    saved = torch.load(path, map_location="cpu")
    params = auxiliary_parameters(model)
    if set(saved) != set(params):
        raise RuntimeError(f"Auxiliary checkpoint mismatch: {set(saved) ^ set(params)}")
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(saved[name].to(device=p.device, dtype=p.dtype))


class AuxiliaryStateCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.params = auxiliary_parameters(model)
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert self.params and all(p.requires_grad and id(p) in optimizer_ids for p in self.params.values())
        self.before = {name: p.detach().float().cpu().clone() for name, p in self.params.items()}
        self.grads = {}
        self.handles = []
        for name, p in self.params.items():
            def record(grad, name=name):
                self.grads[name] = grad.detach().float().norm()
                return grad
            self.handles.append(p.register_hook(record))
        self.checked = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.checked:
            return
        record = {}
        for name, p in self.params.items():
            delta = (p.detach().float().cpu() - self.before[name]).abs().max().item()
            grad = float(self.grads[name].item()) if name in self.grads else 0.0
            assert delta > 0 and grad > 0 and torch.isfinite(torch.tensor(grad)), (name, delta, grad)
            record[name] = {"max_update": delta, "gradient_norm": grad, "trainable": True}
        if state.is_world_process_zero:
            (Path(args.output_dir) / "auxiliary_update_verified.json").write_text(json.dumps(record, indent=2))
        for handle in self.handles:
            handle.remove()
        self.before.clear()
        self.checked = True

    def on_save(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero:
            save_auxiliary(model, Path(args.output_dir) / f"checkpoint-{state.global_step}")
