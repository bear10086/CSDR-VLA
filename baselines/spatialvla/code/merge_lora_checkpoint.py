#!/usr/bin/env python3
"""Merge a trained SpatialVLA LoRA checkpoint for unchanged official evaluation."""

import argparse
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModel, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.adapter, "adapter_config.json")):
        raise FileNotFoundError(f"Not a PEFT checkpoint: {args.adapter}")
    os.makedirs(args.output, exist_ok=True)

    processor_source = args.adapter if os.path.isfile(
        os.path.join(args.adapter, "processor_config.json")
    ) else args.base_model
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(args.output)

    # AutoModel resolves these files locally through config.json's auto_map.
    # save_pretrained does not copy implementation modules from a local model.
    for filename in ("modeling_spatialvla.py", "modeling_gemma2.py"):
        source = os.path.join(args.base_model, filename)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Missing custom model implementation: {source}")
        shutil.copy2(source, os.path.join(args.output, filename))


if __name__ == "__main__":
    main()
