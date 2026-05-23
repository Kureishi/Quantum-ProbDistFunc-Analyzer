# Schrödinger PDF → Parameter Fine-Tuning Pipeline

End-to-end pipeline to fine-tune a Vision-Language Model (VLM) that takes
**probability density function images** as input and returns the **Schrödinger
equation parameters** (potential type, quantum number, energy, well width, etc.)
as structured JSON output, served via **LM Studio** and accessible through a
**Streamlit web UI**.

---

## Pipeline Overview

```
Dataset (PNGs + JSON sidecars)
        │
        ▼
[1] prepare_dataset.py   ← converts to Unsloth conversation format
        │
        ▼
[2] finetune.py          ← QLoRA fine-tune via Unsloth (GPU required)
        │
        ▼
[3] export_gguf.py       ← merge adapter → GGUF for LM Studio
        │
        ▼
    LM Studio            ← load the GGUF + mmproj, start local server
        │
        ├──▶ [4] inference.py       ← CLI: PNG → JSON parameters
        │
        ├──▶ [5] test_inference.py  ← quick sanity check with ✓/✗ output
        │
        ├──▶ [6] evaluate.py        ← full benchmark on held-out test set
        │
        └──▶ [7] app.py             ← Streamlit web UI
```

---

## File Reference

| File | Purpose |
|---|---|
| `generate_probability_distributions.py` | Generate PNG dataset with JSON/TXT/JSONL annotations |
| `scripts/prepare_dataset.py` | Split dataset → train/val/test JSONL for Unsloth |
| `scripts/finetune.py` | QLoRA fine-tune via Unsloth `FastVisionModel` |
| `scripts/export_gguf.py` | Merge LoRA adapter → GGUF via PEFT + llama.cpp |
| `scripts/inference.py` | CLI batch inference against LM Studio server |
| `scripts/test_inference.py` | Interactive per-image test with ground truth comparison |
| `scripts/evaluate.py` | Aggregate accuracy metrics (MAE, RMSE, R², accuracy) |
| `app.py` | Streamlit web UI — upload image, view extracted parameters |
| `configs/finetune_config.yaml` | All training, LoRA, export, and server settings |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For the Streamlit UI:
```bash
pip install streamlit
```

### 2. Generate the dataset

```bash
python generate_probability_distributions.py --preset medium --outdir ./dataset
```

Preset sizes: `small` (~15 samples), `medium` (~50), `large` (~200).  
Each image gets a `.txt` caption sidecar, `.json` metadata sidecar, and entries
in `manifest.csv` and `metadata.jsonl` for fine-tuning frameworks.

### 3. Prepare training data

```bash
python scripts/prepare_dataset.py \
    --dataset_dir ./dataset \
    --output_dir  ./data \
    --val_split   0.1 \
    --test_split  0.1
```

Outputs `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` in Unsloth's
multi-turn vision conversation format.

### 4. Fine-tune

```bash
python scripts/finetune.py --config configs/finetune_config.yaml
```

> **Windows note:** `dataloader_num_workers` is automatically forced to `0` to
> avoid a pickling error with the Qwen2.5-VL processor on Windows
> (`multiprocessing` spawn method). This has no meaningful impact on throughput
> since training is GPU-bound.

The LoRA adapter is saved to `output/adapter/lora_adapter/` separately from
the SFTTrainer checkpoints to avoid a known Unsloth merge bug.

### 5. Export to GGUF

#### Stage 1 — Merge adapter (PEFT, no Unsloth)

Unsloth's built-in save methods (`save_pretrained_merged`, `save_pretrained_gguf`)
have an open bug on Windows where they crash even when LoRA weights are present.
The script bypasses them entirely using standard PEFT:

```bash
python scripts/export_gguf.py --config configs/finetune_config.yaml
```

This merges the adapter via `PeftModel.merge_and_unload()` and saves merged
safetensors to `output/merged/`.

#### Stage 2 — Convert to GGUF (llama.cpp)

If llama.cpp is installed via **winget**:

```bat
:: Get the conversion script (no build needed)
git clone --depth 1 https://github.com/ggerganov/llama.cpp llama-cpp-src
pip install -r llama-cpp-src\requirements.txt

:: Create output directory first (required — script won't create it)
mkdir output\gguf

:: Convert merged model to F16 GGUF
python llama-cpp-src\convert_hf_to_gguf.py ^
    output\merged ^
    --outfile output\gguf\model-f16.gguf ^
    --outtype f16

:: Quantise (llama-quantize is on PATH via winget)
llama-quantize ^
    output\gguf\model-f16.gguf ^
    output\gguf\quanum-image-model.gguf ^
    Q4_K_M
```

Available quantisation levels:

| Argument | File size (7B) | Quality |
|---|---|---|
| `Q4_K_M` | ~4 GB | Good — recommended default |
| `Q5_K_M` | ~5 GB | Better |
| `Q8_0` | ~8 GB | Near-lossless |
| `F16` | ~14 GB | Skip quantise step, use f16 directly |

> **Tip:** If your winget binary version differs from the cloned repo, pin the
> clone to the same release tag to avoid GGUF metadata mismatches:
> `git clone --depth 1 --branch b5083 https://github.com/ggerganov/llama.cpp llama-cpp-src`

### 6. Load in LM Studio

Qwen2.5-VL is a Vision-Language Model and requires **two files** — the main
GGUF and a multimodal projector (`mmproj`) for image support:

```bat
:: Generate the mmproj file
python llama-cpp-src\convert_hf_to_gguf.py ^
    output\merged ^
    --outfile output\gguf\mmproj-f16.gguf ^
    --mmproj
```

Place both files in the same folder:
```
output\gguf\
  quanum-image-model.gguf   ← main model
  mmproj-f16.gguf            ← vision projector (enables image input)
```

Then in LM Studio:
1. **My Models** tab → folder icon → navigate to `output\gguf\`
2. Select `quanum-image-model.gguf` → Load
3. LM Studio auto-detects `mmproj-f16.gguf` and enables vision support
   (you should see a camera icon on the model card)
4. **Developer** tab → **Start Server** (default: `http://localhost:1234`)

> **Without the mmproj file**, the model loads but returns
> `"Model does not support images"` on every request.

### 7. Test inference (quick check)

Verify the server is working and the model produces sensible output:

```bash
# Auto-detects dataset folder, tests first 5 images
python scripts/test_inference.py

# Test a specific image
python scripts/test_inference.py --image dataset/pdf_0000.png

# Test more images
python scripts/test_inference.py --image_dir dataset --n 10
```

Example output:
```
──────────────────────────────────────────────────────────────
  Image : pdf_0000.png
  True  : Infinite Square Well  n=1  E=4.9348 a.u.
──────────────────────────────────────────────────────────────
  Latency : 1.29s

  Predicted parameters:
  {
      "potential_type": "infinite_square_well",
      "quantum_number_n": 1,
      "mass_au": 1.0,
      "energy_au": 4.9348,
      "potential_params": { "L": 1 }
  }

  Accuracy vs ground truth:
  ✓   potential_type        'infinite_square_well' vs 'infinite_square_well'
  ✓   quantum_number_n      1 vs 1
  ✓   energy_au             4.9348 vs 4.9348  (Δ=0.0000)
  ✓   mass_au               1.000 vs 1.000
  ✓   params.L              1 vs 1
```

### 8. Run the web UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The UI provides:

- **Upload panel** — drag-and-drop or click to upload any `|ψ(x)|²` PNG
- **Results panel** — potential type (colour-coded by type), quantum number `n`,
  energy eigenvalue, particle mass, and all potential-specific parameters
  displayed as `st.metric` cards
- **Sidebar** — server URL, auto-detected model selector, temperature and
  max-tokens sliders, last 8 inference results in history
- **Raw JSON expander** — full model response for inspection

The sidebar model selector reads from `/v1/models` at startup — make sure the
LM Studio server is running before launching the app.

### 9. Full benchmark (optional)

Run the full evaluation suite on the held-out test set:

```bash
python scripts/evaluate.py \
    --dataset_dir ./dataset \
    --output_dir  ./eval_results
```

Produces `eval_results/metrics.json` with:
- `potential_type` classification accuracy
- `quantum_number_n` exact-match accuracy  
- `energy_au` MAE / RMSE / R²
- Per-parameter regression metrics (L, V0, omega, a, b)
- Per-potential-type breakdown

---

## `test_inference.py` vs `evaluate.py`

| | `test_inference.py` | `evaluate.py` |
|---|---|---|
| **Purpose** | Quick interactive sanity check | Systematic benchmark |
| **Typical use** | After loading model in LM Studio | After a full training run |
| **Output** | Human-readable ✓/✗ per field | Aggregate MAE/RMSE/R²/accuracy |
| **Scale** | 1–10 images | Full test split |
| **Saves results** | No | `metrics.json` + `detailed_results.jsonl` |

---

## Model Choice

The pipeline defaults to **Qwen2.5-VL-7B-Instruct** (best accuracy/VRAM balance).
Change `base_model` in `configs/finetune_config.yaml`:

| Model | VRAM (QLoRA 4-bit) | Speed | Notes |
|---|---|---|---|
| `unsloth/Qwen2.5-VL-7B-Instruct` | ~10 GB | ★★★ | **Default. Best balance.** |
| `unsloth/Qwen2.5-VL-3B-Instruct` | ~6 GB | ★★★★ | Faster, slightly less accurate |
| `unsloth/Llama-3.2-11B-Vision-Instruct` | ~14 GB | ★★ | Strong reasoning |
| `unsloth/gemma-3-4b-it` | ~8 GB | ★★★ | Good multilingual |

---

## Output Format

The model returns a structured JSON object:

```json
{
  "potential_type": "infinite_square_well",
  "quantum_number_n": 2,
  "mass_au": 1.0,
  "energy_au": 19.739,
  "potential_params": { "L": 1.0 }
}
```

`potential_params` keys by potential type:

| `potential_type` | Keys |
|---|---|
| `infinite_square_well` | `L` (well width) |
| `harmonic_oscillator` | `omega` (angular frequency) |
| `finite_square_well` | `L` (width), `V0` (barrier height) |
| `hydrogen_radial` | `l` (angular momentum) |
| `double_well` | `a` (separation), `b` (shape factor), `V0` (barrier height) |

---

## Known Issues & Fixes Applied

| Issue | Fix |
|---|---|
| `PicklingError` on Windows during training | `dataloader_num_workers` auto-forced to `0` on Windows |
| `NotImplementedError` in `revert_weight_conversion` | Replaced `model.save_pretrained()` with Unsloth's `save_pretrained_merged()` |
| Unsloth `# of LoRAs = N does not match # of saved modules = 0` | LoRA adapter saved to dedicated `lora_adapter/` subdir; export uses raw PEFT merge |
| `Model does not support images` in LM Studio | Generate and co-locate `mmproj-f16.gguf` alongside the main GGUF |
| `FileNotFoundError` during GGUF conversion | Create `output\gguf\` directory manually before running `convert_hf_to_gguf.py` |
| Streamlit result card renders as raw HTML/CSS text | Replaced `st.markdown(unsafe_allow_html=True)` with native `st.metric` / `st.columns` |

---

## Hardware Requirements

- **Fine-tuning**: NVIDIA GPU with ≥10 GB VRAM (RTX 3080/4070 or better),
  or Google Colab A100/T4. Apple Silicon (M1/M2/M3) via MLX also works.
- **Inference / UI**: Any machine that can run LM Studio. CPU inference is
  supported; GPU recommended for speed (<2s latency vs 10–30s on CPU).
