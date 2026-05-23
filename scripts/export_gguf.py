#!/usr/bin/env python3
"""
export_gguf.py
==============
Merge the LoRA adapter into the base model and export to GGUF format
for LM Studio.

Why we bypass Unsloth's save methods
--------------------------------------
Unsloth's save_pretrained_merged / save_pretrained_gguf crash with:
  RuntimeError: Saving LoRA finetune failed since # of LoRAs = N
                does not match # of saved modules = 0
This is an open Unsloth bug on Windows (issues #3287, #3288, #4294).
The LoRA weights ARE present (our sanity check shows 1424 tensors) but
Unsloth's internal merge_and_overwrite_lora counts them as 0.

Workaround (two-stage):
  Stage 1  — Manual PEFT merge (no Unsloth):
    Load base model in bfloat16 via Transformers AutoModel
    Attach adapter via PeftModel.from_pretrained()
    Call peft_model.merge_and_unload()       ← standard PEFT, always works
    Save merged safetensors with save_pretrained()

  Stage 2  — GGUF conversion via llama.cpp Python wheel:
    pip install llama-cpp-python  (ships convert_hf_to_gguf internally)
    OR use the llama-cpp-python gguf package directly
    OR call llama.cpp CLI if installed

Usage:
  python scripts/export_gguf.py --config configs/finetune_config.yaml
  python scripts/export_gguf.py --config configs/finetune_config.yaml --quant q5_k_m
  python scripts/export_gguf.py --config configs/finetune_config.yaml --merged_only
  python scripts/export_gguf.py --config configs/finetune_config.yaml --adapter_dir ./my_adapter
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


UNSLOTH_NATIVE_QUANTS = {
    "q4_k_m", "q5_k_m", "q8_0", "f16", "f32",
    "q2_k", "q3_k_m", "q6_k",
}


def find_llama_cpp_convert() -> Path | None:
    """Find llama.cpp's convert_hf_to_gguf.py."""
    llama_dir = os.environ.get("LLAMA_CPP_DIR")
    candidates = []
    if llama_dir:
        candidates.append(Path(llama_dir) / "convert_hf_to_gguf.py")
    candidates += [
        Path("./llama.cpp/convert_hf_to_gguf.py"),
        Path("~/llama.cpp/convert_hf_to_gguf.py").expanduser(),
    ]
    return next((c for c in candidates if c.exists()), None)


def find_llama_quantize() -> Path | None:
    """Find llama-quantize binary."""
    llama_dir = os.environ.get("LLAMA_CPP_DIR")
    candidates = []
    if llama_dir:
        candidates += [
            Path(llama_dir) / "build" / "bin" / "llama-quantize",
            Path(llama_dir) / "llama-quantize",
        ]
    candidates += [
        Path("./llama.cpp/build/bin/llama-quantize"),
        Path("~/llama.cpp/build/bin/llama-quantize").expanduser(),
        Path("llama-quantize"),
    ]
    for c in candidates:
        try:
            if Path(c).exists():
                return c
            if sys.platform != "win32":
                if subprocess.run(["which", str(c)], capture_output=True).returncode == 0:
                    return c
        except Exception:
            pass
    return None


# ── Stage 1: Manual PEFT merge ────────────────────────────────────────────────

def merge_adapter_manually(base_model_id: str, adapter_dir: Path,
                            merged_dir: Path, load_in_4bit: bool) -> bool:
    """
    Bypass Unsloth entirely:
      1. Load base model in bfloat16 via AutoModelForCausalLM / AutoModel
      2. Attach LoRA via PeftModel.from_pretrained
      3. Merge with peft_model.merge_and_unload()
      4. Save to merged_dir

    Returns True on success.
    """
    print(f"\n── Stage 1: Manual PEFT merge (bypassing Unsloth save) ─────────")
    print(f"  Base model  : {base_model_id}")
    print(f"  LoRA adapter: {adapter_dir}")
    print(f"  Output      : {merged_dir}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        from peft import PeftModel, PeftConfig
    except ImportError as e:
        print(f"ERROR: {e}  —  pip install transformers peft")
        return False

    # Determine dtype — use bfloat16 for merge (avoids NaN; no 4-bit needed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"  Merge dtype : {dtype}")

    # Read base model name from adapter_config if not provided
    try:
        import json
        adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        resolved_base = adapter_cfg.get("base_model_name_or_path", base_model_id)
        if resolved_base and resolved_base != base_model_id:
            print(f"  (adapter_config base model: {resolved_base})")
            base_model_id = resolved_base
    except Exception:
        pass

    print(f"  Loading base model in {dtype} (no quantisation for merge) …")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype   = dtype,
            device_map    = "auto",
            trust_remote_code = True,
        )
    except Exception as e:
        print(f"  AutoModelForCausalLM failed ({e}), trying AutoModel …")
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(
            base_model_id,
            torch_dtype   = dtype,
            device_map    = "auto",
            trust_remote_code = True,
        )

    print(f"  Attaching LoRA adapter …")
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        torch_dtype = dtype,
    )

    print(f"  Merging weights (merge_and_unload) …")
    merged = peft_model.merge_and_unload()

    print(f"  Saving merged model …")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)

    # Save tokenizer / processor
    try:
        proc = AutoProcessor.from_pretrained(str(adapter_dir), trust_remote_code=True)
        proc.save_pretrained(str(merged_dir))
        print(f"  Saved processor/tokenizer")
    except Exception:
        try:
            tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
            tok.save_pretrained(str(merged_dir))
            print(f"  Saved tokenizer")
        except Exception as e2:
            print(f"  WARNING: could not save tokenizer/processor: {e2}")

    # Free VRAM
    import gc
    del merged, peft_model, base_model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"  ✓ Merged model saved → {merged_dir.resolve()}")
    return True


# ── Stage 2: GGUF conversion ──────────────────────────────────────────────────

def convert_to_gguf(merged_dir: Path, gguf_dir: Path, quant: str) -> Path | None:
    """
    Convert merged HF model → GGUF.
    Tries (in order):
      A) llama.cpp convert_hf_to_gguf.py  (if LLAMA_CPP_DIR set or ./llama.cpp exists)
      B) gguf Python package               (pip install gguf)
      C) Print manual instructions and exit
    """
    gguf_dir.mkdir(parents=True, exist_ok=True)

    convert_script = find_llama_cpp_convert()
    if convert_script:
        return _convert_via_llama_cpp(merged_dir, gguf_dir, quant, convert_script)

    # Try the standalone gguf-py package convert script
    try:
        import gguf  # noqa
        convert_py = Path(gguf.__file__).parent.parent / "scripts" / "convert_hf_to_gguf.py"
        if convert_py.exists():
            return _convert_via_llama_cpp(merged_dir, gguf_dir, quant, convert_py)
    except ImportError:
        pass

    # Nothing found — print instructions
    print(f"\nERROR: Cannot find llama.cpp convert_hf_to_gguf.py.")
    print(f"\nOption A — Install llama.cpp (recommended for Windows):")
    print(f"  git clone https://github.com/ggerganov/llama.cpp")
    print(f"  cd llama.cpp && pip install -r requirements.txt")
    print(f"  set LLAMA_CPP_DIR=C:\\path\\to\\llama.cpp   (then re-run this script)")
    print(f"\nOption B — Install the gguf Python package:")
    print(f"  pip install gguf")
    print(f"  (then re-run this script)")
    print(f"\nManual conversion once llama.cpp is available:")
    print(f"  python convert_hf_to_gguf.py {merged_dir} \\")
    print(f"      --outfile {gguf_dir}\\model-f16.gguf --outtype f16")
    if quant != "f16":
        print(f"  llama-quantize {gguf_dir}\\model-f16.gguf \\")
        print(f"      {gguf_dir}\\model-{quant}.gguf {quant.upper()}")
    return None


def _convert_via_llama_cpp(merged_dir: Path, gguf_dir: Path,
                            quant: str, convert_script: Path) -> Path | None:
    f16_path = gguf_dir / "model-f16.gguf"
    cmd = [sys.executable, str(convert_script),
           str(merged_dir), "--outfile", str(f16_path), "--outtype", "f16"]
    print(f"\n── Stage 2: Converting to GGUF ─────────────────────────────────")
    print(f"  {' '.join(str(x) for x in cmd)}")
    if subprocess.run(cmd).returncode != 0:
        print("ERROR: GGUF conversion failed.")
        return None

    if quant == "f16":
        print(f"  ✓ F16 GGUF → {f16_path}")
        return f16_path

    # Quantise
    final_path = gguf_dir / f"model-{quant}.gguf"
    quantize_bin = find_llama_quantize()
    if quantize_bin is None:
        print(f"  WARNING: llama-quantize not found; leaving as F16.")
        print(f"  Quantise manually: llama-quantize {f16_path} {final_path} {quant.upper()}")
        return f16_path

    cmd_q = [str(quantize_bin), str(f16_path), str(final_path), quant.upper()]
    print(f"  {' '.join(str(x) for x in cmd_q)}")
    if subprocess.run(cmd_q).returncode != 0:
        print("ERROR: Quantisation failed.")
        return f16_path
    f16_path.unlink(missing_ok=True)
    print(f"  ✓ {quant.upper()} GGUF → {final_path}")
    return final_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and export to GGUF for LM Studio."
    )
    parser.add_argument("--config",      required=True)
    parser.add_argument("--adapter_dir", default=None,
                        help="Override adapter directory (default: <output_dir>/lora_adapter)")
    parser.add_argument("--quant",       default=None,
                        help="Quantisation method (q4_k_m | q5_k_m | q8_0 | f16 …)")
    parser.add_argument("--merged_only", action="store_true",
                        help="Only merge to HF safetensors; skip GGUF conversion")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    export_cfg  = cfg["export"]
    quant       = (args.quant or export_cfg.get("quantisation", "q4_k_m")).lower()
    output_dir  = Path(cfg["training"]["output_dir"])
    merged_dir  = Path(export_cfg["merged_dir"])
    gguf_dir    = Path(export_cfg["gguf_dir"])
    base_model  = cfg["base_model"]

    # Resolve adapter directory
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
    else:
        adapter_dir = output_dir / "lora_adapter"
        if not (adapter_dir / "adapter_config.json").exists():
            # Fallback: output_dir itself
            if (output_dir / "adapter_config.json").exists():
                adapter_dir = output_dir
            else:
                print(f"\nERROR: adapter_config.json not found in:")
                print(f"  {adapter_dir}")
                print(f"  {output_dir}")
                print(f"\nRe-run finetune.py with the updated script, which saves the")
                print(f"LoRA adapter to <output_dir>/lora_adapter/ automatically.")
                print(f"Or pass --adapter_dir <path> to point to your adapter folder.")
                sys.exit(1)

    print(f"\n{'='*60}")
    print(f" Schrödinger LoRA → GGUF Export")
    print(f"{'='*60}")
    print(f" Adapter : {adapter_dir}")
    print(f" Merged  : {merged_dir}")
    print(f" GGUF    : {gguf_dir}")
    print(f" Quant   : {quant}")
    print(f"{'='*60}")

    # ── Stage 1: Merge ────────────────────────────────────────────────────────
    ok = merge_adapter_manually(base_model, adapter_dir, merged_dir,
                                 cfg.get("load_in_4bit", True))
    if not ok:
        sys.exit(1)

    if args.merged_only:
        print(f"\n✓  Merged HF model at: {merged_dir.resolve()}")
        print(f"   Load this folder directly in LM Studio as a HF model.")
        return

    # ── Stage 2: GGUF conversion ──────────────────────────────────────────────
    final_path = convert_to_gguf(merged_dir, gguf_dir, quant)
    if final_path is None:
        print(f"\n⚠  GGUF conversion could not run automatically.")
        print(f"   The merged HF model is ready at: {merged_dir.resolve()}")
        print(f"   Follow the manual instructions above to produce the GGUF.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" ✓  Export complete!")
    print(f" GGUF : {final_path.resolve()}")
    print(f"{'='*60}")
    print(f"\nLoad in LM Studio:")
    print(f"  1. My Models tab → folder icon → {gguf_dir.resolve()}")
    print(f"  2. Select {final_path.name} → Load")
    print(f"  3. Developer tab → Start Server (http://localhost:1234)")
    print(f"  4. python scripts/inference.py --config configs/finetune_config.yaml \\")
    print(f"             --image ./dataset/pdf_0000.png")


if __name__ == "__main__":
    main()
