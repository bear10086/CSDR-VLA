import os
import torch
import numpy as np
import torch.nn as nn
import datasets
from torch.utils.data import DataLoader
import transformers
from transformers import logging, TrainerCallback, Trainer
from transformers.trainer import LengthGroupedSampler, RandomSampler, has_length, is_datasets_available, seed_worker, _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_pt_utils import logger
from typing import List, Optional
from torch.utils.data import Dataset, Sampler

from train.csdr_loss import CSDRConfig, prompt_token_signature, spatialvla_csdr_loss, apply_order_budget

logger = logging.get_logger(__name__)

IGNORE_INDEX = -100

# data patch
def concat_pad_data_collator(features, pad_id=0):
    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        # handel temp_token_type_ids for gemma
        temp_token_type_ids = torch.LongTensor([0] * max_item_length) # pad with 0 to indicate first scentence
        temp_token_type_ids[:feat['token_type_ids'].shape[0]] = feat['token_type_ids']
        feat['token_type_ids'] = temp_token_type_ids

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch

# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L38
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')

    return chunks

# copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L88
def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L99
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            if (
                    not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                'If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]...'
            )
            lengths = lengths.tolist()
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

# patch trainer
def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    # Build the sampler.
    if self.args.group_by_length:
        lengths = []
        for dataset in self.train_dataset.datasets:
            lengths = lengths + dataset.length
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            # self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        return RandomSampler(self.train_dataset)

def replace_train_sampler():
    transformers.Trainer._get_train_sampler = _get_train_sampler
    print('Replace train sampler!!')

def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
    else:
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    if train_dataset.use_raw_dataloader:
        return DataLoader(train_dataset, **dataloader_params)
    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

def replace_train_dataloader():
    transformers.Trainer.get_train_dataloader = get_train_dataloader
    print("Replace train dataloader!!")

def _action_mask(labels, action_tokenizer):
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for tokenizer in (
        action_tokenizer.translation_tokenizer,
        action_tokenizer.rotation_tokenizer,
        action_tokenizer.gripper_tokenizer,
    ):
        mask |= (labels >= tokenizer.token_start_idx) & (labels <= tokenizer.token_end_idx)
    return mask


def configure_csdr_trainer(trainer, cfg: CSDRConfig, order_to_task_ratio: float, warmup_steps: int):
    """Attach a last-layer hook without materializing hidden states from every layer."""
    trainer.csdr_config = cfg
    trainer.csdr_order_to_task_ratio = float(order_to_task_ratio)
    trainer.csdr_warmup_steps = int(warmup_steps)
    trainer._csdr_hidden_cache = {}
    trainer._csdr_last_logged_step = -1
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    candidates = [module for name, module in unwrapped.named_modules() if name.endswith("language_model.model.norm")]
    if len(candidates) != 1:
        names = [name for name, _ in unwrapped.named_modules() if name.endswith(".norm")]
        raise RuntimeError(f"Expected one language model final norm, found {len(candidates)}; norms={names[-10:]}")

    def capture_last_hidden(_module, _inputs, output):
        trainer._csdr_hidden_cache["last_hidden"] = output

    trainer._csdr_hook_handle = candidates[0].register_forward_hook(capture_last_hidden)


def _masked_ddp_mean(per_sample, sample_mask):
    sample_mask = sample_mask.to(device=per_sample.device, dtype=per_sample.dtype)
    local_sum = (per_sample * sample_mask).sum()
    local_count = sample_mask.sum().detach()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        global_count = local_count.clone()
        global_value = local_sum.detach().clone()
        torch.distributed.all_reduce(global_count)
        torch.distributed.all_reduce(global_value)
        if not bool((global_count > 0).item()):
            raise RuntimeError("A planned batch cannot contain only padding samples.")
        gradient_value = local_sum * world_size / global_count
        reported_value = global_value / global_count
        return gradient_value + (reported_value - gradient_value.detach())
    return local_sum / local_count.clamp_min(1.0)


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    sample_loss_mask = inputs.pop("sample_loss_mask", None)
    csdr_sample_mask = inputs.pop("csdr_sample_mask", None)
    action_valid_lengths = inputs.pop("action_valid_length", None)
    canonical_prompt_ids = inputs.pop("canonical_prompt_ids", None)
    if canonical_prompt_ids is None:
        canonical_prompt_ids = inputs.pop("canonical_prompt_id", None)
    outputs = model(**inputs)
    task_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    unwrapped_model = self.accelerator.unwrap_model(model)
    action_tokenizer = unwrapped_model.action_tokenizer
    labels = inputs["labels"]
    shift_labels = labels[..., 1:].contiguous()
    action_mask = _action_mask(shift_labels, action_tokenizer)
    predicted_ids = outputs["logits"][..., :-1, :].detach().argmax(dim=-1)
    counts = action_mask.sum(dim=1)
    if int(counts.min().item()) == 0 or not bool((counts == counts[0]).all().item()):
        raise RuntimeError(f"Every sample must contain the same non-zero action-token count, got {counts.tolist()}.")
    per_sample_action_error = (
        ((predicted_ids != shift_labels) & action_mask).float().sum(dim=1) / counts.float()
    ).detach()
    if sample_loss_mask is None:
        sample_loss_mask = torch.ones_like(per_sample_action_error)
    else:
        sample_loss_mask = sample_loss_mask.to(per_sample_action_error.device, dtype=torch.float32)
        shift_logits = outputs["logits"][..., :-1, :].contiguous().float()
        token_losses = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shift_labels)
        supervised_counts = (shift_labels != IGNORE_INDEX).sum(dim=1).clamp_min(1)
        per_sample_task_loss = token_losses.sum(dim=1) / supervised_counts.float()
        task_loss = _masked_ddp_mean(per_sample_task_loss, sample_loss_mask)
    loss = task_loss

    metrics = {
        "task_loss": task_loss.detach(),
        "action_accuracy": (
            1.0 - _masked_ddp_mean(per_sample_action_error, sample_loss_mask)
        ).detach(),
    }
    if hasattr(self, "csdr_config"):
        if "last_hidden" not in self._csdr_hidden_cache:
            raise RuntimeError("CSDR last-hidden hook did not run.")
        last_hidden = self._csdr_hidden_cache.pop("last_hidden")
        shifted_hidden = last_hidden[..., :-1, :]
        token_count = int(counts[0].item())
        action_hidden = shifted_hidden[action_mask].reshape(
            shifted_hidden.shape[0], token_count, shifted_hidden.shape[-1]
        )
        if canonical_prompt_ids is not None:
            context = canonical_prompt_ids.to(labels.device, dtype=torch.long).unsqueeze(1)
        else:
            context = prompt_token_signature(inputs["input_ids"], labels, inputs["attention_mask"])
        if csdr_sample_mask is None:
            csdr_sample_mask = torch.ones_like(per_sample_action_error, dtype=torch.bool)
        else:
            csdr_sample_mask = csdr_sample_mask.to(labels.device, dtype=torch.bool)
        csdr = spatialvla_csdr_loss(
            action_hidden_states=action_hidden,
            actions=inputs["actions"],
            context_signatures=context,
            action_errors=per_sample_action_error,
            cfg=self.csdr_config,
            valid_samples=csdr_sample_mask,
            valid_lengths=action_valid_lengths,
        )
        if self.csdr_warmup_steps > 0:
            schedule = min((int(self.state.global_step) + 1) / self.csdr_warmup_steps, 1.0)
        else:
            schedule = 1.0
        raw_order = csdr["csdr_order_loss"]
        contribution, order_scale = apply_order_budget(
            raw_order, csdr["csdr_budget_reference"], task_loss,
            self.csdr_order_to_task_ratio, schedule,
        )
        loss = task_loss + contribution
        metrics.update({name: value.detach() for name, value in csdr.items()})
        metrics.update(
            {
                "csdr_contribution": contribution.detach(),
                "csdr_order_scale": order_scale.detach(),
                "csdr_schedule": contribution.detach().new_tensor(schedule),
                "csdr_to_task_ratio": contribution.detach().abs() / task_loss.detach().abs().clamp_min(1e-8),
            }
        )

    step = int(self.state.global_step)
    logging_steps = max(int(self.args.logging_steps), 1)
    if step != getattr(self, "_csdr_last_logged_step", -1) and step % logging_steps == 0:
        self._csdr_last_logged_step = step
        self.log({name: float(value.float().item()) for name, value in metrics.items()})

    return (loss, outputs) if return_outputs else loss

def replace_compute_loss():
    transformers.Trainer.compute_loss = compute_loss
    print("Replace compute_loss!!")

class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            output_dir = args.output_dir
            if state.global_step > 0:
                output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.processor.save_pretrained(output_dir)
        return control

class ProfilerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=4),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_output")
        )
        self.profiler.__enter__()

    def training_step(self, model, inputs):
        output = super().training_step(model, inputs)
        self.profiler.step()
        return output

    def __del__(self):
        self.profiler.__exit__(None, None, None)
