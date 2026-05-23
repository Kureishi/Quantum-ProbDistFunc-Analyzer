#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Convert the probability-distribution PNG dataset (with .json sidecars) into
the multi-turn conversation format expected by Unsloth's SFTTrainer for
vision fine-tuning.

Each training example becomes:
  [system]  → task description
  [user]    → <image> + "Extract the Schrödinger equation parameters..."
  [assistant] → JSON object with potential_type, n, energy_au, potential_params

Outputs three JSONL files:
  data/train.jsonl
  data/val.jsonl
  data/test.jsonl

Usage:
  python scripts/prepare_dataset.py --dataset_dir ./dataset --output_dir ./data
  python scripts/prepare_dataset.py --dataset_dir ./dataset --output_dir ./data \\
      --val_split 0.1 --test_split 0.1 --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a quantum mechanics analysis assistant. "
    "Given an image of a probability density function |ψ(x)|² plot for a solution "
    "to the time-independent Schrödinger equation, extract and return the physical "
    "parameters as a JSON object. "
    "Always respond with valid JSON only — no explanation, no markdown fences."
)

USER_PROMPT = (
    "Analyse this |ψ(x)|² probability density function plot and return the "
    "Schrödinger equation parameters as JSON with these exact keys:\n"
    "  potential_type   (string: infinite_square_well | harmonic_oscillator | "
    "finite_square_well | hydrogen_radial | double_well)\n"
    "  quantum_number_n (integer)\n"
    "  mass_au          (float, in atomic units)\n"
    "  energy_au        (float, in atomic units)\n"
    "  potential_params (object with potential-specific keys)\n\n"
    "Respond with JSON only."
)


def build_assistant_response(meta: dict) -> str:
    """Build the target JSON string the model should output."""
    output = {
        "potential_type":   meta["potential_type"],
        "quantum_number_n": meta["quantum_number_n"],
        "mass_au":          round(meta["mass_au"], 4),
        "energy_au":        round(meta["energy_au"], 6),
        "potential_params": meta["potential_params"],
    }
    return json.dumps(output, separators=(", ", ": "))


def load_samples(dataset_dir: Path) -> list[dict]:
    """Load all samples from the dataset directory using .json sidecars."""
    samples = []
    json_files = sorted(dataset_dir.glob("pdf_*.json"))

    if not json_files:
        print(f"ERROR: No pdf_*.json files found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    skipped = 0
    for jf in json_files:
        meta = json.loads(jf.read_text(encoding="utf-8"))
        img_path = dataset_dir / meta["image_file"]
        if not img_path.exists():
            print(f"  WARNING: image not found, skipping: {img_path}")
            skipped += 1
            continue
        samples.append({
            "image_path": str(img_path.resolve()),
            "meta":       meta,
        })

    print(f"  Loaded {len(samples)} samples ({skipped} skipped).")
    return samples


def sample_to_conversation(sample: dict) -> dict:
    """
    Convert one sample to Unsloth's vision conversation format.

    Unsloth expects:
      {
        "messages": [
          {"role": "system",    "content": [{"type": "text", "text": "..."}]},
          {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
          {"role": "assistant", "content": [{"type": "text", "text": "<JSON>"}]}
        ],
        "images": ["<absolute_path_or_PIL_image>"]
      }
    """
    assistant_json = build_assistant_response(sample["meta"])

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},   # placeholder — Unsloth matches with images[]
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_json}],
            },
        ],
        "images": [sample["image_path"]],
    }


def split_dataset(samples: list, val_frac: float, test_frac: float, seed: int):
    rng = random.Random(seed)
    data = list(samples)
    rng.shuffle(data)
    n = len(data)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    test  = data[:n_test]
    val   = data[n_test:n_test + n_val]
    train = data[n_test + n_val:]
    return train, val, test


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>4} records → {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset.")
    parser.add_argument("--dataset_dir", required=True,
                        help="Directory containing pdf_*.png + pdf_*.json files")
    parser.add_argument("--output_dir",  default="./data",
                        help="Where to write train/val/test JSONL files")
    parser.add_argument("--val_split",   type=float, default=0.1)
    parser.add_argument("--test_split",  type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)

    print(f"\n── Loading samples from {dataset_dir} ──────────────────────────")
    samples = load_samples(dataset_dir)

    print(f"\n── Splitting dataset ───────────────────────────────────────────")
    train, val, test = split_dataset(
        samples, args.val_split, args.test_split, args.seed
    )
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    print(f"\n── Converting to conversation format ───────────────────────────")
    train_convs = [sample_to_conversation(s) for s in train]
    val_convs   = [sample_to_conversation(s) for s in val]
    test_convs  = [sample_to_conversation(s) for s in test]

    print(f"\n── Writing JSONL files to {output_dir} ─────────────────────────")
    write_jsonl(train_convs, output_dir / "train.jsonl")
    write_jsonl(val_convs,   output_dir / "val.jsonl")
    write_jsonl(test_convs,  output_dir / "test.jsonl")

    # Also save a plain test-metadata file for evaluate.py
    test_meta = [s["meta"] for s in test]
    test_meta_path = output_dir / "test_metadata.jsonl"
    with open(test_meta_path, "w", encoding="utf-8") as f:
        for m in test_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(test_meta):>4} records → {test_meta_path}  (for evaluate.py)")

    print(f"\n✓  Dataset preparation complete.")
    print(f"   Next step: python scripts/finetune.py --config configs/finetune_config.yaml")


if __name__ == "__main__":
    main()
