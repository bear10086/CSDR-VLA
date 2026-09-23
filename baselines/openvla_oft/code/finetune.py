"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""
import hashlib
from checkpoint_utils import checkpoint_identity, fingerprint_files, cursor_after_updates
import itertools
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb
from experiments.robot.openvla_utils import check_model_logic_mismatch, model_is_on_hf_hub, update_auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import compute_actions_l1_loss, compute_token_accuracy, get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from representation_losses import CSDRConfig, csdr_loss, linear_warmup_factor, prompt_token_signature
from cohort_planning.baseline_adapters import LiberoOpenVLAWindowTransform
from cohort_planning.planned_dataset import PlannedCollator, PlannedRLDSDataset
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@dataclass
class FinetuneConfig:
    vla_path: str = 'openvla/openvla-7b'
    data_root_dir: Path = Path('datasets/rlds')
    dataset_name: str = 'aloha_scoop_x_into_bowl'
    run_root_dir: Path = Path('runs')
    shuffle_buffer_size: int = 100000
    planned_cohort_data_dir: Optional[Path] = None
    planned_epoch_count: int = 0
    planned_episode_cache_size: int = 16
    num_diffusion_steps_train: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    batch_size: int = 8
    learning_rate: float = 0.0005
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100000
    grad_accumulation_steps: int = 1
    max_steps: int = 200000
    use_val_set: bool = False
    val_freq: int = 10000
    val_time_limit: int = 180
    save_freq: int = 10000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    diffusion_sample_freq: int = 50
    warm_start_components: bool = False
    component_checkpoint_dir: Optional[Path] = None
    component_checkpoint_step: Optional[int] = None
    log_step_offset: int = 0
    skip_checkpoint_sync: bool = False
    resume_training_state: bool = False
    training_state_path: Optional[Path] = None
    resume_lora_adapter_dir: Optional[Path] = None
    representation_layer: int = -1
    csdr_order_weight: float = 0.0
    csdr_order_to_task_ratio: float = 0.005
    csdr_state_weight: float = 0.2
    csdr_action_weight: float = 0.8
    csdr_action_translation_weight: float = 1.0
    csdr_action_rotation_weight: float = 0.5
    csdr_action_gripper_weight: float = 0.25
    csdr_k_near: int = 4
    csdr_k_far: int = 8
    csdr_near_max_control_distance: float = 0.8
    csdr_far_min_control_distance: float = 1.0
    csdr_minimum_control_gap: float = 0.2
    csdr_order_margin: float = 0.1
    csdr_action_error_scale: float = 0.02
    csdr_confusion_scale: float = 0.1
    csdr_fixed_scale_path: Optional[Path] = None
    csdr_synchronized_prompt_batches: bool = False
    csdr_prompt_cohorts: int = 4
    csdr_ready_prompt_count: int = 16
    csdr_warmup_steps: int = 300
    state_position_weight: float = 1.0
    state_rotation_weight: float = 0.5
    state_gripper_weight: float = 0.25
    diffusion_regularizer_max_t_fraction: float = 0.5
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: str = 'your-wandb-project'
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10
    seed: int = 7
CONTROL_SCALE_FIELDS = ('state_position', 'state_rotation', 'state_gripper', 'action_translation', 'action_rotation', 'action_gripper')

def load_csdr_fixed_scales(cfg: FinetuneConfig) -> None:
    if cfg.csdr_fixed_scale_path is None:
        raise ValueError('CSDR requires --csdr_fixed_scale_path from the per-dataset scale scan.')
    scale_path = Path(cfg.csdr_fixed_scale_path)
    if not scale_path.is_file():
        raise FileNotFoundError(f'CSDR fixed-scale file does not exist: {scale_path}')
    with scale_path.open(encoding='utf-8') as stream:
        payload = json.load(stream)
    scales_by_dataset = payload.get('scales_by_dataset')
    if not isinstance(scales_by_dataset, dict) or not scales_by_dataset:
        raise ValueError(f"Missing non-empty 'scales_by_dataset' object in {scale_path}.")
    dataset_names = sorted(scales_by_dataset)
    scale_table = []
    for dataset_name in dataset_names:
        scales = scales_by_dataset[dataset_name]
        row = []
        for field_name in CONTROL_SCALE_FIELDS:
            value = float(scales.get(field_name, 0.0))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'Invalid {dataset_name} fixed scale {field_name}={value} in {scale_path}.')
            row.append(value)
        scale_table.append(row)
    cfg.csdr_dataset_names = dataset_names
    cfg.csdr_scale_table = scale_table
    cfg.csdr_dataset_lookup = {name: index for index, name in enumerate(dataset_names)}
    cfg.csdr_scale_tensor = None
    print(f'Loaded CSDR per-dataset scales from {scale_path}: datasets={dataset_names}')

def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        run_id = cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.vla_path.split('/')[-1]
        if 'chkpt' in run_id.split('--')[-1]:
            run_id = '--'.join(run_id.split('--')[:-1])
    else:
        run_id = f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr-{cfg.learning_rate}"
        if cfg.use_lora:
            run_id += f'+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}'
        if cfg.image_aug:
            run_id += '--image_aug'
        if cfg.run_id_note is not None:
            run_id += f'--{cfg.run_id_note}'
    return run_id

def load_checkpoint(module_name: str, path: str, step: int, device: str='cpu') -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f'{module_name}--{step}_checkpoint.pt')
    print(f'Loading checkpoint: {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool=False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)

def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum((p.numel() for p in module.parameters() if p.requires_grad))
    print(f'# trainable params in {name}: {num_params}')

def get_direction_token_ids(tokenizer) -> Tuple[int, ...]:
    """Resolve the seven motion words used by the no-new-head semantic loss."""
    words = ('forward', 'backward', 'left', 'right', 'up', 'down', 'still')
    token_ids = []
    for word in words:
        encoded = tokenizer.encode(f' {word}', add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Tokenizer produced no token for motion word '{word}'.")
        if len(encoded) != 1:
            print(f"Warning: motion word '{word}' maps to {len(encoded)} tokens; using its final token for the auxiliary vocabulary classifier.")
        token_ids.append(encoded[-1])
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f'Motion words did not map to seven unique vocabulary ids: {token_ids}')
    print(f'Motion semantic token ids: {dict(zip(words, token_ids))}')
    return tuple(token_ids)

def init_module(module_class: Type[nn.Module], module_name: str, cfg: FinetuneConfig, device_id: int, module_args: dict, to_bf16: bool=False, find_unused_params: bool=False) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)
    should_load = cfg.resume or cfg.warm_start_components
    if should_load:
        checkpoint_dir = cfg.component_checkpoint_dir or Path(cfg.vla_path)
        checkpoint_step = cfg.component_checkpoint_step if cfg.warm_start_components else cfg.resume_step
        if checkpoint_step is None:
            raise ValueError(f'Missing checkpoint step while loading {module_name}.')
        state_dict = load_checkpoint(module_name, str(checkpoint_dir), checkpoint_step)
        module.load_state_dict(state_dict)
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return wrap_ddp(module, device_id, find_unused_params)

def masked_ddp_mean(per_sample: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    """Global masked mean with the correct DDP gradient scaling."""
    sample_mask = sample_mask.to(device=per_sample.device, dtype=per_sample.dtype)
    local_sum = (per_sample * sample_mask).sum()
    local_count = sample_mask.sum().detach()
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        global_count = local_count.clone()
        global_value = local_sum.detach().clone()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_value, op=dist.ReduceOp.SUM)
        if not bool((global_count > 0).item()):
            raise RuntimeError('A planned batch cannot contain only padding samples.')
        gradient_value = local_sum * world_size / global_count
        reported_value = global_value / global_count
        return gradient_value + (reported_value - gradient_value.detach())
    return local_sum / local_count.clamp_min(1.0)

def run_forward_pass(vla, action_head, noisy_action_projector, proprio_projector, batch, action_tokenizer, device_id, use_l1_regression, use_diffusion, use_proprio, use_film, num_patches, compute_diffusion_l1=False, num_diffusion_steps_train=None, representation_cfg: Optional[FinetuneConfig]=None, direction_token_ids: Optional[Tuple[int, ...]]=None, apply_regularizers: bool=True, regularizer_step: int=0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}
    ground_truth_actions = batch['actions'].to(device_id).to(torch.bfloat16)
    sample_loss_mask = batch.get('sample_loss_mask')
    if sample_loss_mask is None:
        sample_loss_mask = torch.ones(ground_truth_actions.shape[0], device=device_id)
    else:
        sample_loss_mask = sample_loss_mask.to(device_id, dtype=torch.float32)
    noise, noisy_actions, diffusion_timestep_embeddings = (None, None, None)
    diffusion_timesteps = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(input_ids=batch['input_ids'].to(device_id), attention_mask=batch['attention_mask'].to(device_id), pixel_values=batch['pixel_values'].to(torch.bfloat16).to(device_id), labels=batch['labels'], output_hidden_states=True, proprio=batch['proprio'] if use_proprio else None, proprio_projector=proprio_projector if use_proprio else None, noisy_actions=None, noisy_action_projector=None, diffusion_timestep_embeddings=None, use_film=use_film)
    ground_truth_token_ids = batch['labels'][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    if not True:
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
        curr_action_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
        next_actions_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
        next_actions_l1_loss = compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask)
        metrics.update({'loss_value': loss.item(), 'curr_action_accuracy': curr_action_accuracy.item(), 'curr_action_l1_loss': curr_action_l1_loss.item(), 'next_actions_accuracy': next_actions_accuracy.item(), 'next_actions_l1_loss': next_actions_l1_loss.item()})
    else:
        if representation_cfg is None:
            representation_layer = -1
        else:
            representation_layer = representation_cfg.representation_layer
        if not -len(output.hidden_states) <= representation_layer < len(output.hidden_states):
            raise ValueError(f'representation_layer={representation_layer} is invalid for {len(output.hidden_states)} returned hidden-state tensors.')
        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch['input_ids'].shape[0]
        actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
        regularizer_layer_states = output.hidden_states[representation_layer]
        regularizer_text_states = regularizer_layer_states[:, num_patches:-1]
        regularizer_action_hidden_states = regularizer_text_states[current_action_mask | next_actions_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
        predicted_actions = action_head(actions_hidden_states)
        per_sample_task_loss = (predicted_actions.float() - ground_truth_actions.float()).abs().flatten(1).mean(dim=1)
        loss = masked_ddp_mean(per_sample_task_loss, sample_loss_mask)
        per_sample_action_error = (predicted_actions.detach().float() - ground_truth_actions.detach().float()).abs().flatten(1).mean(dim=1)
        metrics.update({'loss_value': loss.item()})
        should_log_l1_loss = not False or False
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = masked_ddp_mean((ground_truth_curr_action.float() - predicted_curr_action.float()).abs().flatten(1).mean(dim=1), sample_loss_mask)
            next_actions_l1_loss = masked_ddp_mean((ground_truth_next_actions.float() - predicted_next_actions.float()).abs().flatten(1).mean(dim=1), sample_loss_mask)
            metrics.update({'curr_action_l1_loss': curr_action_l1_loss.item(), 'next_actions_l1_loss': next_actions_l1_loss.item()})
    if apply_regularizers and representation_cfg is not None:
        task_loss = loss
        metrics['task_loss'] = task_loss.detach().float().item()
        total_loss = task_loss
        valid_samples = torch.ones(batch_size, device=device_id, dtype=torch.bool)
        if 'csdr_sample_mask' in batch:
            valid_samples &= batch['csdr_sample_mask'].to(device_id, dtype=torch.bool)
        metrics['regularizer_valid_fraction'] = valid_samples.float().mean().detach().item()
        uses_csdr = representation_cfg.csdr_order_weight != 0.0
        if uses_csdr:
            if batch.get('proprio') is None:
                raise ValueError('CSDR requires --use_proprio True.')
            if not hasattr(representation_cfg, 'csdr_dataset_names'):
                raise ValueError('CSDR per-dataset scales were not loaded.')
            if 'dataset_names' not in batch:
                raise ValueError('CSDR requires dataset_names from the RLDS collator.')
            schedule_factor = linear_warmup_factor(regularizer_step, representation_cfg.csdr_warmup_steps)
            metrics['csdr_schedule_factor'] = schedule_factor
            csdr_config = CSDRConfig(state_weight=representation_cfg.csdr_state_weight, action_weight=representation_cfg.csdr_action_weight, state_position_weight=representation_cfg.state_position_weight, state_rotation_weight=representation_cfg.state_rotation_weight, state_gripper_weight=representation_cfg.state_gripper_weight, action_translation_weight=representation_cfg.csdr_action_translation_weight, action_rotation_weight=representation_cfg.csdr_action_rotation_weight, action_gripper_weight=representation_cfg.csdr_action_gripper_weight, k_near=representation_cfg.csdr_k_near, k_far=representation_cfg.csdr_k_far, near_max_control_distance=representation_cfg.csdr_near_max_control_distance, far_min_control_distance=representation_cfg.csdr_far_min_control_distance, minimum_control_gap=representation_cfg.csdr_minimum_control_gap, order_margin=representation_cfg.csdr_order_margin, action_error_scale=representation_cfg.csdr_action_error_scale, confusion_scale=representation_cfg.csdr_confusion_scale)
            if 'canonical_prompt_ids' in batch:
                context_signatures = batch['canonical_prompt_ids'].to(device_id, dtype=torch.long).unsqueeze(1)
            else:
                context_signatures = prompt_token_signature(input_ids=batch['input_ids'][:, 1:].to(device_id), labels=ground_truth_token_ids, attention_mask=batch['attention_mask'][:, 1:].to(device_id))
            dataset_lookup = representation_cfg.csdr_dataset_lookup
            decoded_dataset_names = [value.decode('utf-8') if isinstance(value, bytes) else str(value) for value in batch['dataset_names']]
            try:
                dataset_ids = torch.tensor([dataset_lookup[name] for name in decoded_dataset_names], device=device_id, dtype=torch.long)
            except KeyError as error:
                raise ValueError(f'CSDR has no fixed scales for dataset {error.args[0]!r}.') from error
            scale_table = representation_cfg.csdr_scale_tensor
            if scale_table is None or scale_table.device.index != device_id:
                scale_table = torch.tensor(representation_cfg.csdr_scale_table, device=device_id, dtype=torch.float32)
                representation_cfg.csdr_scale_tensor = scale_table
            ordinal = csdr_loss(action_hidden_states=regularizer_action_hidden_states, proprio=batch['proprio'].to(device_id), actions=ground_truth_actions, context_signatures=context_signatures, dataset_ids=dataset_ids, scale_table=scale_table, action_errors=per_sample_action_error, cfg=csdr_config, valid_samples=valid_samples)
            raw_order = representation_cfg.csdr_order_weight * ordinal['csdr_order_loss']
            order_budget = task_loss.detach().abs() * representation_cfg.csdr_order_to_task_ratio * schedule_factor
            order_scale = torch.minimum(raw_order.detach().new_tensor(1.0), order_budget / raw_order.detach().abs().clamp_min(1e-08))
            order_contribution_v7 = order_scale * raw_order
            total_loss = total_loss + order_contribution_v7
            metrics['csdr_order_contribution'] = order_contribution_v7.detach().item()
            metrics['csdr_order_scale'] = order_scale.detach().item()
            metrics['csdr_local_unique_contexts'] = float(torch.unique(context_signatures, dim=0).shape[0])
            for name, value in ordinal.items():
                metrics[name] = value.detach()
        loss = total_loss
        metrics['loss_value'] = total_loss.detach().float().item()
        metrics['regularizer_contribution'] = (total_loss - task_loss).detach().float().item()
        metrics['regularizer_to_task_ratio'] = ((total_loss - task_loss).detach().abs() / task_loss.detach().abs().clamp_min(1e-08)).float().item()
    return (loss, metrics)

def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics

def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        if name == 'loss_value':
            log_dict[f'{prefix}/Loss'] = value
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)

def save_training_checkpoint(cfg, run_dir, log_step, vla, processor, proprio_projector, noisy_action_projector, action_head, optimizer, scheduler, completed_training_steps, train_dataset, distributed_state) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        optimizer (Optimizer): Optimizer whose state enables full-state continuation.
        scheduler (LRScheduler): Learning-rate scheduler paired with the optimizer.
        completed_training_steps (int): Optimizer steps completed within the representation run.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = 'latest_checkpoint.pt'
    else:
        checkpoint_dir = Path(str(run_dir) + f'--{log_step}_chkpt')
        checkpoint_name_suffix = f'{log_step}_checkpoint.pt'
    adapter_dir = checkpoint_dir / 'lora_adapter'
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f'Saving Model Checkpoint for Step {log_step}')
    dist.barrier()
    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f'proprio_projector--{checkpoint_name_suffix}')
        if action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f'action_head--{checkpoint_name_suffix}')
        torch.save({'schema_version': 2, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'completed_training_steps': int(completed_training_steps), 'log_step': int(log_step), 'gradient_accumulation_steps': int(cfg.grad_accumulation_steps), 'world_size': int(distributed_state.num_processes), 'base_content_sha256': cfg._base_content_sha256, 'batch_size': cfg.batch_size, 'plan_fingerprint': getattr(train_dataset, 'plan_fingerprint', None), 'next_micro_batch': getattr(train_dataset, 'consumed_micro_batches', None)}, checkpoint_dir / f'training_state--{checkpoint_name_suffix}')
        if cfg.use_film:
            torch.save(vla.module.vision_backbone.state_dict(), checkpoint_dir / f'vision_backbone--{checkpoint_name_suffix}')
    dist.barrier()
    saved_rng = {'python_random_state': random.getstate(), 'numpy_random_state': np.random.get_state(), 'torch_cpu_rng_state': torch.get_rng_state(), 'torch_cuda_rng_state': torch.cuda.get_rng_state()}
    torch.save(saved_rng, checkpoint_dir / f'rng_state_rank{distributed_state.process_index}--{checkpoint_name_suffix}')
    dist.barrier()
    if cfg.use_lora and cfg.merge_lora_during_training:
        if distributed_state.is_main_process:
            base_vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
            merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
            merged_vla = merged_vla.merge_and_unload()
            merged_vla.save_pretrained(checkpoint_dir)
            print(f'Saved merged model for Step {log_step} at: {checkpoint_dir}')
            del merged_vla, base_vla
        dist.barrier()
    # Model materialization for export must not change the training RNG stream.
    random.setstate(saved_rng['python_random_state'])
    np.random.set_state(saved_rng['numpy_random_state'])
    torch.set_rng_state(saved_rng['torch_cpu_rng_state'])
    torch.cuda.set_rng_state(saved_rng['torch_cuda_rng_state'])

def _torch_load_full(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_optimizer_scheduler_state(cfg: FinetuneConfig, optimizer, scheduler, distributed_state: PartialState) -> int:
    if not cfg.resume_training_state:
        return 0
    state_path = Path(cfg.training_state_path)
    if not state_path.is_file():
        raise FileNotFoundError(f'Training-state checkpoint does not exist: {state_path}')
    payload = _torch_load_full(state_path, map_location='cpu')
    if int(payload.get('schema_version', 0)) != 2:
        raise ValueError('Exact resume requires a version-2 checkpoint with base identity and data cursor; legacy checkpoints cannot be silently resumed.')
    if int(payload.get('world_size', -1)) != distributed_state.num_processes:
        raise ValueError('Full-state continuation requires the same DDP world size as the saved run.')
    if int(payload.get('gradient_accumulation_steps', -1)) != cfg.grad_accumulation_steps:
        raise ValueError('Full-state continuation requires the same gradient accumulation setting.')
    if payload.get('batch_size') != cfg.batch_size:
        raise ValueError('Resume requires the same per-device batch size.')
    cfg._resume_payload = payload
    completed_steps = int(payload.get('completed_training_steps', -1))
    if not 0 <= completed_steps < cfg.max_steps:
        raise ValueError(f'Saved completed_training_steps={completed_steps} must be in [0, max_steps={cfg.max_steps}).')
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    print(f'Restored optimizer and scheduler from {state_path} at added step {completed_steps}.')
    return completed_steps

def restore_rank_rng_state(cfg: FinetuneConfig, distributed_state: PartialState) -> None:
    if not cfg.resume_training_state:
        return
    state_path = Path(cfg.training_state_path)
    rng_name = state_path.name.replace('training_state--', f'rng_state_rank{distributed_state.process_index}--', 1)
    rng_path = state_path.with_name(rng_name)
    if not rng_path.is_file():
        raise FileNotFoundError(f'Per-rank RNG state does not exist: {rng_path}')
    payload = _torch_load_full(rng_path, map_location='cpu')
    random.setstate(payload['python_random_state'])
    np.random.set_state(payload['numpy_random_state'])
    torch.set_rng_state(payload['torch_cpu_rng_state'])
    torch.cuda.set_rng_state(payload['torch_cuda_rng_state'])
    print(f'Restored RNG state for rank {distributed_state.process_index} from {rng_path}.')

def run_validation(vla, action_head, noisy_action_projector, proprio_projector, val_dataloader, action_tokenizer, device_id, cfg, num_patches, log_step, distributed_state, val_time_limit) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0
    all_val_metrics = []
    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = run_forward_pass(vla=vla, action_head=action_head, noisy_action_projector=noisy_action_projector, proprio_projector=proprio_projector, batch=batch, action_tokenizer=action_tokenizer, device_id=device_id, use_l1_regression=True, use_diffusion=False, use_proprio=cfg.use_proprio, use_film=cfg.use_film, num_patches=num_patches, compute_diffusion_l1=True, num_diffusion_steps_train=None)
            metrics['loss'] = metrics['loss_value']
            all_val_metrics.append(metrics)
            val_batches_count += 1
            if time.time() - val_start_time > val_time_limit:
                break
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)
    avg_val_metrics['val_batches_count'] = val_batches_count
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, 'VLA Val', log_step, wandb)

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, 'Only LoRA fine-tuning is supported. Please set --use_lora=True!'
    assert not False, 'Cannot do both L1 regression and diffusion. Please pick one of them!'
    if cfg.warm_start_components:
        assert cfg.component_checkpoint_step is not None, '--component_checkpoint_step is required with --warm_start_components True.'
    if cfg.resume_training_state:
        if cfg.training_state_path is None or cfg.resume_lora_adapter_dir is None:
            raise ValueError('--resume_training_state True requires --training_state_path and --resume_lora_adapter_dir.')
        if not cfg.warm_start_components:
            raise ValueError('Full-state continuation also requires --warm_start_components True.')
    if not 0.0 <= cfg.diffusion_regularizer_max_t_fraction <= 1.0:
        raise ValueError('diffusion_regularizer_max_t_fraction must be between 0 and 1.')
    if cfg.csdr_order_weight != 0.0:
        assert cfg.use_proprio, 'CSDR requires --use_proprio True.'
        if cfg.csdr_order_weight < 0.0:
            raise ValueError('CSDR order weight must be non-negative.')
        if cfg.csdr_k_near < 1 or cfg.csdr_k_far < 1:
            raise ValueError('CSDR requires positive k_near and k_far.')
        if cfg.csdr_near_max_control_distance <= 0.0:
            raise ValueError('CSDR near distance must be positive.')
        if cfg.csdr_far_min_control_distance <= cfg.csdr_near_max_control_distance:
            raise ValueError('CSDR far distance must be greater than its near distance.')
        if cfg.csdr_action_error_scale <= 0.0 or cfg.csdr_confusion_scale <= 0.0:
            raise ValueError('CSDR action error and confusion scales must be positive.')
        if not 0.0 <= cfg.csdr_order_to_task_ratio <= 0.05:
            raise ValueError('CSDR order ratio must be between 0 and 0.05.')
        if not False and cfg.planned_cohort_data_dir is None:
            raise ValueError('CSDR requires synchronized prompt batches or --planned_cohort_data_dir.')
        if cfg.csdr_prompt_cohorts < 1:
            raise ValueError('CSDR prompt cohorts must be positive.')
        if cfg.csdr_ready_prompt_count < cfg.csdr_prompt_cohorts:
            raise ValueError('CSDR ready prompt count must cover every cohort.')
        linear_warmup_factor(0, cfg.csdr_warmup_steps)
        load_csdr_fixed_scales(cfg)
    cfg.vla_path = cfg.vla_path.rstrip('/')
    print(f'Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`')
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    process_seed = cfg.seed + distributed_state.process_index
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.random.default_generator.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.wandb_run_name or f'ft+{run_id}', group=cfg.wandb_group, config={'objective': 'l1', 'dataset': cfg.dataset_name, 'representation_layer': cfg.representation_layer, 'csdr_order_weight': cfg.csdr_order_weight, 'csdr_order_to_task_ratio': cfg.csdr_order_to_task_ratio, 'csdr_k_near': cfg.csdr_k_near, 'csdr_k_far': cfg.csdr_k_far, 'csdr_near_max_control_distance': cfg.csdr_near_max_control_distance, 'csdr_far_min_control_distance': cfg.csdr_far_min_control_distance, 'csdr_action_error_scale': cfg.csdr_action_error_scale, 'csdr_confusion_scale': cfg.csdr_confusion_scale, 'csdr_synchronized_prompt_batches': False, 'csdr_prompt_cohorts': cfg.csdr_prompt_cohorts, 'csdr_ready_prompt_count': cfg.csdr_ready_prompt_count, 'csdr_warmup_steps': cfg.csdr_warmup_steps, 'global_world_size': distributed_state.num_processes, 'per_device_batch': cfg.batch_size, 'gradient_accumulation': cfg.grad_accumulation_steps, 'seed': cfg.seed})
    print(f'Action chunk length: {NUM_ACTIONS_CHUNK}')
    original_vla_path = cfg.vla_path
    if model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path
        if cfg.warm_start_components and (cfg.component_checkpoint_dir is None or str(cfg.component_checkpoint_dir).rstrip('/') == original_vla_path.rstrip('/')):
            cfg.component_checkpoint_dir = Path(vla_download_path)
    else:
        AutoConfig.register('openvla', OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    if distributed_state.is_main_process and (not cfg.skip_checkpoint_sync):
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    cfg._base_content_sha256 = checkpoint_identity(cfg.vla_path)['content_sha256']
    if cfg.resume_training_state:
        resume_payload = _torch_load_full(Path(cfg.training_state_path), map_location='cpu')
        if resume_payload.get('schema_version') != 2 or resume_payload.get('base_content_sha256') != cfg._base_content_sha256:
            raise ValueError('Resume requires the original unmerged base and a version-2 training checkpoint with matching base identity.')
    vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    if cfg.use_lora:
        if cfg.resume_training_state:
            adapter_dir = Path(cfg.resume_lora_adapter_dir)
            if not adapter_dir.is_dir():
                raise FileNotFoundError(f'Resume LoRA adapter directory does not exist: {adapter_dir}')
            vla = PeftModel.from_pretrained(vla, adapter_dir, is_trainable=True)
            print(f'Restored trainable LoRA adapter from {adapter_dir}')
        else:
            lora_config = LoraConfig(r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=cfg.lora_dropout, target_modules='all-linear', init_lora_weights='gaussian')
            vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
    vla = vla.to(device_id)
    if cfg.use_film:
        count_parameters(vla.vision_backbone, 'vla.vision_backbone (original)')
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(vision_backbone=vla.model.vision_backbone, llm_dim=vla.llm_dim)
        count_parameters(vla.vision_backbone, 'vla.vision_backbone (post-wrap)')
        if cfg.resume or cfg.warm_start_components:
            checkpoint_dir = cfg.component_checkpoint_dir or Path(cfg.vla_path)
            checkpoint_step = cfg.component_checkpoint_step if cfg.warm_start_components else cfg.resume_step
            state_dict = load_checkpoint('vision_backbone', str(checkpoint_dir), checkpoint_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)
    vla = wrap_ddp(vla, device_id, find_unused=True)
    if cfg.use_proprio:
        proprio_projector = init_module(ProprioProjector, 'proprio_projector', cfg, device_id, {'llm_dim': vla.module.llm_dim, 'proprio_dim': PROPRIO_DIM})
    action_head = init_module(L1RegressionActionHead, 'action_head', cfg, device_id, {'input_dim': vla.module.llm_dim, 'hidden_dim': vla.module.llm_dim, 'action_dim': ACTION_DIM}, to_bf16=True)
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f'# total trainable params: {sum((p.numel() for p in trainable_params))}')
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]['lr']
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)
    restored_training_steps = load_optimizer_scheduler_state(cfg, optimizer, scheduler, distributed_state)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    direction_token_ids = get_direction_token_ids(processor.tokenizer) if 0.0 != 0.0 else None
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(action_tokenizer, processor.tokenizer, image_transform=processor.image_processor.apply_transform, prompt_builder_fn=PurePromptBuilder, use_wrist_image=use_wrist_image, use_proprio=cfg.use_proprio)
    if cfg.planned_cohort_data_dir is not None:
        planned_dir = Path(cfg.planned_cohort_data_dir)
        plan_paths = sorted((planned_dir / 'plans').glob('epoch_seed*.npz'), key=lambda path: int(path.stem.split('seed')[-1]))
        if cfg.planned_epoch_count > 0:
            plan_paths = plan_paths[:cfg.planned_epoch_count]
        dataset_names = ('libero_spatial_no_noops', 'libero_object_no_noops', 'libero_goal_no_noops', 'libero_10_no_noops')
        dataset_sources = {name: (Path(cfg.data_root_dir) / name / '1.0.0', planned_dir / f'{name}_tfrecord_index.json') for name in dataset_names}
        planned_transform = LiberoOpenVLAWindowTransform(batch_transform=batch_transform, statistics_path=Path(cfg.vla_path) / 'dataset_statistics.json', resize_size=tuple(vla.module.config.image_sizes), image_aug=cfg.image_aug)
        train_dataset = PlannedRLDSDataset(manifest_path=planned_dir / 'trajectories.jsonl', plan_paths=plan_paths, dataset_sources=dataset_sources, rank=distributed_state.process_index, episode_transform=planned_transform.prepare_episode, window_transform=planned_transform, episode_cache_size=cfg.planned_episode_cache_size)
        train_dataset.plan_fingerprint = fingerprint_files([planned_dir / 'trajectories.jsonl', *plan_paths])
        if distributed_state.is_main_process:
            print(f'Enabled full-coverage planned LIBERO data: epochs={len(plan_paths)}, local_samples={len(train_dataset)}')
    else:
        train_dataset = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, batch_transform, resize_resolution=tuple(vla.module.config.image_sizes), shuffle_buffer_size=cfg.shuffle_buffer_size, image_aug=cfg.image_aug)
    if cfg.use_val_set:
        val_dataset = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, batch_transform, resize_resolution=tuple(vla.module.config.image_sizes), shuffle_buffer_size=cfg.shuffle_buffer_size // 10, image_aug=cfg.image_aug, train=False)
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side='right')
    if cfg.planned_cohort_data_dir is not None:
        collator = PlannedCollator(collator)
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0, generator=torch.Generator().manual_seed(cfg.seed))
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=None, collate_fn=collator, num_workers=0)
    restore_rank_rng_state(cfg, distributed_state)
    recent_metrics = {'loss_value': deque(maxlen=cfg.grad_accumulation_steps), 'curr_action_accuracy': deque(maxlen=cfg.grad_accumulation_steps), 'curr_action_l1_loss': deque(maxlen=cfg.grad_accumulation_steps), 'next_actions_accuracy': deque(maxlen=cfg.grad_accumulation_steps), 'next_actions_l1_loss': deque(maxlen=cfg.grad_accumulation_steps)}
    total_micro_batches = len(dataloader)
    micro_batches_per_epoch = list(getattr(train_dataset, 'micro_batches_per_plan', [total_micro_batches]))
    if sum(micro_batches_per_epoch) != total_micro_batches:
        raise ValueError(f'Planned epoch boundaries do not match the DataLoader length: boundaries={micro_batches_per_epoch}, total={total_micro_batches}.')
    resume_micro = 0
    if cfg.resume_training_state:
        payload = cfg._resume_payload
        if cfg.planned_cohort_data_dir is None or payload.get('plan_fingerprint') != train_dataset.plan_fingerprint:
            raise ValueError('Exact resume requires the unchanged full-coverage data plan.')
        resume_micro = payload.get('next_micro_batch')
        expected = cursor_after_updates(micro_batches_per_epoch, cfg.grad_accumulation_steps, restored_training_steps)
        if resume_micro != expected:
            raise ValueError(f'Resume cursor disagrees with completed updates: {resume_micro} != {expected}')
        train_dataset.start_micro = resume_micro
        train_dataset.consumed_micro_batches = resume_micro
    epoch_index = 0
    epoch_start_batch = 0
    epoch_end_batch = micro_batches_per_epoch[0]
    completed_steps = restored_training_steps
    last_saved_steps = restored_training_steps
    with tqdm.tqdm(total=cfg.max_steps, initial=restored_training_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        previous_iteration_end = time.perf_counter()
        for batch_idx, batch in enumerate(dataloader, start=resume_micro):
            while batch_idx >= epoch_end_batch:
                epoch_index += 1
                epoch_start_batch = epoch_end_batch
                epoch_end_batch += micro_batches_per_epoch[epoch_index]
            micro_step_in_epoch = batch_idx - epoch_start_batch
            epoch_micro_batches = micro_batches_per_epoch[epoch_index]
            iteration_start = time.perf_counter()
            data_wait_seconds = iteration_start - previous_iteration_end
            total_micro_step = completed_steps * cfg.grad_accumulation_steps + micro_step_in_epoch % cfg.grad_accumulation_steps
            compute_diffusion_l1 = False
            loss, metrics = run_forward_pass(vla=vla, action_head=action_head, noisy_action_projector=None, proprio_projector=proprio_projector if cfg.use_proprio else None, batch=batch, action_tokenizer=action_tokenizer, device_id=device_id, use_l1_regression=True, use_diffusion=False, use_proprio=cfg.use_proprio, use_film=cfg.use_film, num_patches=NUM_PATCHES, compute_diffusion_l1=compute_diffusion_l1, num_diffusion_steps_train=None, representation_cfg=cfg, direction_token_ids=direction_token_ids, apply_regularizers=True, regularizer_step=completed_steps)
            accumulation_group_start = micro_step_in_epoch // cfg.grad_accumulation_steps * cfg.grad_accumulation_steps
            accumulation_group_size = min(cfg.grad_accumulation_steps, epoch_micro_batches - accumulation_group_start)
            normalized_loss = loss / accumulation_group_size
            normalized_loss.backward()
            metrics['perf_data_wait_seconds'] = data_wait_seconds
            metrics['perf_forward_backward_seconds'] = time.perf_counter() - iteration_start
            for metric_name, value in metrics.items():
                if metric_name not in recent_metrics:
                    recent_metrics[metric_name] = deque(maxlen=cfg.grad_accumulation_steps)
                recent_metrics[metric_name].append(value)
            gradient_step_idx = completed_steps
            did_optimizer_step = (micro_step_in_epoch + 1) % cfg.grad_accumulation_steps == 0 or micro_step_in_epoch + 1 == epoch_micro_batches
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)
            legacy_resume_offset = cfg.resume_step if cfg.resume and cfg.resume_step is not None else 0
            log_step = cfg.log_step_offset + legacy_resume_offset + gradient_step_idx
            if distributed_state.is_main_process and did_optimizer_step and (log_step % cfg.wandb_log_freq == 0):
                log_metrics_to_wandb(smoothened_metrics, 'VLA Train', log_step, wandb)
                repr_summary = ' '.join((f'{name}={smoothened_metrics[name]:.6f}' for name in ('task_loss', 'loss_value', 'control_order_contribution', 'local_geometry_contribution', 'csdr_order_contribution', 'csdr_schedule_factor', 'csdr_order_scale', 'csdr_order_loss', 'csdr_valid_anchor_fraction', 'csdr_triplet_active_fraction', 'csdr_mean_near_count', 'csdr_mean_far_count', 'csdr_same_context_far_fraction', 'csdr_mean_error_weight', 'csdr_error_weight_saturation_fraction', 'csdr_mean_confusion_weight', 'csdr_strong_confusion_fraction', 'csdr_local_unique_contexts', 'csdr_global_unique_contexts', 'regularizer_to_task_ratio', 'selected_far_pair_fraction', 'selected_near_pair_fraction', 'triplet_active_fraction') if name in smoothened_metrics))
                if repr_summary:
                    print(f'[repr_metrics] step={log_step} {repr_summary}', flush=True)
                print(f"[perf_metrics] step={log_step} data_wait_seconds={smoothened_metrics['perf_data_wait_seconds']:.4f} forward_backward_seconds={smoothened_metrics['perf_forward_backward_seconds']:.4f}", flush=True)
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            if distributed_state.is_main_process and did_optimizer_step and (gradient_step_idx % cfg.wandb_log_freq == 0):
                wandb.log({'VLA Train/Learning Rate': scheduler.get_last_lr()[0]}, step=log_step)
            if did_optimizer_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()
                completed_steps += 1
                train_dataset.consumed_micro_batches = batch_idx + 1
            completed_log_step = cfg.log_step_offset + legacy_resume_offset + completed_steps
            if did_optimizer_step and completed_steps > 0 and (completed_steps % cfg.save_freq == 0):
                save_training_checkpoint(cfg=cfg, run_dir=run_dir, log_step=completed_log_step, vla=vla, processor=processor, proprio_projector=proprio_projector if cfg.use_proprio else None, noisy_action_projector=None, action_head=action_head, optimizer=optimizer, scheduler=scheduler, completed_training_steps=completed_steps, train_dataset=train_dataset, distributed_state=distributed_state)
                last_saved_steps = completed_steps
            if cfg.use_val_set and did_optimizer_step and (completed_steps > 0) and (completed_steps % cfg.val_freq == 0):
                run_validation(vla=vla, action_head=action_head, noisy_action_projector=None, proprio_projector=proprio_projector if cfg.use_proprio else None, val_dataloader=val_dataloader, action_tokenizer=action_tokenizer, device_id=device_id, cfg=cfg, num_patches=NUM_PATCHES, log_step=completed_log_step, distributed_state=distributed_state, val_time_limit=cfg.val_time_limit)
                vla.train()
            previous_iteration_end = time.perf_counter()
            if did_optimizer_step and completed_steps >= cfg.max_steps:
                print(f'Completed {cfg.max_steps} optimizer steps (logged as step {completed_log_step}). Stopping training...')
                break
        if completed_steps > restored_training_steps and completed_steps != last_saved_steps:
            final_log_step = cfg.log_step_offset + legacy_resume_offset + completed_steps
            save_training_checkpoint(cfg=cfg, run_dir=run_dir, log_step=final_log_step, vla=vla, processor=processor, proprio_projector=proprio_projector if cfg.use_proprio else None, noisy_action_projector=None, action_head=action_head, optimizer=optimizer, scheduler=scheduler, completed_training_steps=completed_steps, train_dataset=train_dataset, distributed_state=distributed_state)
if __name__ == '__main__':
    finetune()
