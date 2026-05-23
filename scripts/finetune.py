#!/usr/bin/env python3
"""
finetune.py
===========
Fine-tune a Vision-Language Model (VLM) on the Schrödinger PDF parameter
extraction task using Unsloth's FastVisionModel with QLoRA.

The model learns: PNG image of |ψ(x)|² → JSON of Schrödinger parameters.

Requirements:
  - NVIDIA GPU with ≥10 GB VRAM (for default Qwen2.5-VL-7B @ 4-bit)
  - OR Google Colab A100/T4
  - OR Apple Silicon Mac (M1/M2/M3) — change load_in_4bit: false in config

Windows note:
  dataloader_num_workers is automatically set to 0 on Windows because the
  Qwen2.5-VL processor cannot be pickled across subprocess boundaries
  (multiprocessing 'spawn' start method). This is safe — it just means
  data loading runs in the main process rather than background workers.

Usage:
  python scripts/finetune.py --config configs/finetune_config.yaml
  python scripts/finetune.py --config configs/finetune_config.yaml --resume_from_checkpoint ./output/adapter/checkpoint-100
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import yaml

# ── Windows multiprocessing guard ────────────────────────────────────────────
# MUST be at module top-level (outside main/any function) so that the
# re-spawned worker processes Windows creates don't re-execute training code.
# https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
IS_WINDOWS = platform.system() == "Windows"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(jsonl_path: str, tokenizer, max_seq_length: int):
    """Load JSONL conversations, apply chat template, return HF Dataset."""
    from datasets import Dataset
    from PIL import Image

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    def process(record):
        images = [Image.open(p).convert("RGB") for p in record["images"]]
        return {"messages": record["messages"], "images": images}

    processed = [process(r) for r in records]
    return Dataset.from_list(processed)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLM with Unsloth QLoRA.")
    parser.add_argument("--config", required=True, help="Path to finetune_config.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Path to a checkpoint directory to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Imports (deferred so --help works without GPU deps) ──────────────────
    try:
        from unsloth import FastVisionModel, is_bf16_supported
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig
        import torch
    except ImportError as e:
        print(f"\nERROR: Missing dependency: {e}")
        print("Install with: pip install unsloth[colab-new] trl")
        sys.exit(1)

    lora_cfg  = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg  = cfg["data"]

    # ── Resolve dataloader workers ────────────────────────────────────────────
    # Windows uses 'spawn' to create worker processes, which requires pickling
    # the entire DataLoader state including the processor/tokenizer.
    # Qwen2.5-VL's processor fails this pickle round-trip → EOFError / PicklingError.
    # Fix: force num_workers=0 on Windows (data loaded in main process, no perf loss
    # on typical dataset sizes; training is GPU-bound anyway).
    if IS_WINDOWS:
        num_workers = 0
        print("  [Windows] dataloader_num_workers forced to 0 (avoids processor pickle error)")
    else:
        num_workers = train_cfg.get("dataloader_num_workers", 0)

    print(f"\n── Loading base model: {cfg['base_model']} ─────────────────────")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name                 = cfg["base_model"],
        max_seq_length             = cfg["max_seq_length"],
        load_in_4bit               = cfg["load_in_4bit"],
        use_gradient_checkpointing = train_cfg.get("gradient_checkpointing", "unsloth"),
    )

    print(f"\n── Applying LoRA adapter (r={lora_cfg['r']}, α={lora_cfg['alpha']}) ─")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = lora_cfg.get("finetune_vision_layers", True),
        finetune_language_layers   = lora_cfg.get("finetune_language_layers", True),
        finetune_attention_modules = lora_cfg.get("finetune_attention_modules", True),
        finetune_mlp_modules       = lora_cfg.get("finetune_mlp_modules", True),
        r              = lora_cfg["r"],
        lora_alpha     = lora_cfg["alpha"],
        lora_dropout   = lora_cfg["dropout"],
        bias           = "none",
        random_state   = train_cfg.get("seed", 42),
        use_rslora     = False,
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"\n── Loading datasets ─────────────────────────────────────────────")
    train_dataset = build_dataset(data_cfg["train_file"], tokenizer, cfg["max_seq_length"])
    val_dataset   = build_dataset(data_cfg["val_file"],   tokenizer, cfg["max_seq_length"])
    print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # ── Formatting function for SFTTrainer ───────────────────────────────────
    def formatting_func(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return texts

    # ── Training arguments ───────────────────────────────────────────────────
    use_bf16 = train_cfg.get("bf16", False) and is_bf16_supported()
    use_fp16 = train_cfg.get("fp16", False) and not use_bf16

    sft_config = SFTConfig(
        output_dir                  = train_cfg["output_dir"],
        num_train_epochs            = train_cfg["num_train_epochs"],
        per_device_train_batch_size = train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size  = train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"],
        learning_rate               = train_cfg["learning_rate"],
        lr_scheduler_type           = train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio                = train_cfg.get("warmup_ratio", 0.05),
        weight_decay                = train_cfg.get("weight_decay", 0.01),
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        logging_steps               = train_cfg.get("logging_steps", 10),
        eval_strategy               = "steps",
        eval_steps                  = train_cfg.get("eval_steps", 50),
        save_strategy               = "steps",
        save_steps                  = train_cfg.get("save_steps", 100),
        save_total_limit            = train_cfg.get("save_total_limit", 3),
        load_best_model_at_end      = train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model       = train_cfg.get("metric_for_best_model", "eval_loss"),
        report_to                   = train_cfg.get("report_to", "none"),
        seed                        = train_cfg.get("seed", 42),
        # ── Windows-safe dataloader settings ─────────────────────────────────
        dataloader_num_workers      = num_workers,
        dataloader_persistent_workers = False,  # must be False when num_workers=0
        remove_unused_columns       = False,    # required for vision datasets
        dataset_text_field          = "",       # we use a formatting_func
        dataset_kwargs              = {"skip_prepare_dataset": True},
        max_seq_length              = cfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = UnslothVisionDataCollator(model, tokenizer),
        formatting_func = formatting_func,
        args            = sft_config,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n── Starting training ────────────────────────────────────────────")
    trainer_stats = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    print(f"\n── Training complete ────────────────────────────────────────────")
    print(f"   Runtime   : {trainer_stats.metrics.get('train_runtime', 0):.0f}s")

    # ── Save LoRA adapter to a DEDICATED subdirectory ─────────────────────────
    # IMPORTANT: SFTTrainer saves full model checkpoints into output_dir at
    # each save_steps. If export_gguf.py reloads from output_dir it sees a
    # plain merged model (0 LoRA layers) and crashes with:
    #   "# of LoRAs = N does not match # of saved modules = 0"
    #
    # Fix: save the LoRA adapter (adapter_config.json + adapter_model.safetensors)
    # to a separate lora_adapter/ subdir that SFTTrainer never touches.
    # export_gguf.py loads from this path to get a proper PeftModel.
    output_dir   = Path(train_cfg["output_dir"])
    adapter_path = output_dir / "lora_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(adapter_path))      # saves adapter_config.json +
    tokenizer.save_pretrained(str(adapter_path))  # adapter_model.safetensors
    print(f"   LoRA adapter saved → {adapter_path.resolve()}")
    print(f"   (Saved separately from SFTTrainer checkpoints to avoid LoRA=0 error)")
    print(f"\n   Next step: python scripts/export_gguf.py --config configs/finetune_config.yaml")


# ── Windows spawn entry-point guard ──────────────────────────────────────────
# On Windows, multiprocessing uses 'spawn': every worker process imports this
# module from scratch. Without this guard the workers would call main() again,
# causing the EOFError / PicklingError seen in the traceback.
# With num_workers=0 this guard is technically redundant, but it is cheap
# insurance against accidental config changes.
if __name__ == "__main__":
    main()
