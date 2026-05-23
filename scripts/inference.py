#!/usr/bin/env python3
"""
inference.py
============
Send one or more PNG probability-density-function images to a fine-tuned VLM
running in LM Studio and parse the returned Schrödinger equation parameters.

LM Studio must be running with the local server enabled (default port 1234).
The model is called via the OpenAI-compatible /v1/chat/completions endpoint
with the image embedded as a base64 data URI.

Usage:
  # Single image
  python scripts/inference.py --image ./dataset/pdf_0000.png

  # Multiple images
  python scripts/inference.py --images ./dataset/pdf_0000.png ./dataset/pdf_0005.png

  # Batch from a directory
  python scripts/inference.py --image_dir ./dataset --output results.jsonl

  # With explicit model ID (otherwise auto-detected from /v1/models)
  python scripts/inference.py --image ./dataset/pdf_0000.png --model my-model-id

  # With custom LM Studio URL
  python scripts/inference.py --image ./dataset/pdf_0000.png --base_url http://192.168.1.10:1234/v1
"""

import argparse
import base64
import json
import sys
import time
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

EXPECTED_KEYS = {
    "potential_type", "quantum_number_n", "mass_au", "energy_au", "potential_params"
}


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def image_to_data_uri(image_path: str) -> str:
    """Read a PNG/JPEG and encode it as a base64 data URI."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(suffix, "image/png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def get_model_id(client) -> str:
    """Auto-detect the first loaded model from LM Studio's /v1/models."""
    models = client.models.list()
    if not models.data:
        print("ERROR: No models loaded in LM Studio. Load a model first.", file=sys.stderr)
        sys.exit(1)
    model_id = models.data[0].id
    print(f"  Auto-detected model: {model_id}")
    return model_id


def parse_response(raw: str) -> dict:
    """
    Parse the model's text output into a dict.
    Handles: plain JSON, ```json fenced JSON, trailing commas.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        inner = [l for l in lines[1:] if not l.startswith("```")]
        text = "\n".join(inner).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try removing trailing commas (common LLM mistake)
        import re
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {"_parse_error": str(e), "_raw": raw}


def validate_output(parsed: dict) -> tuple[bool, list[str]]:
    """Check the parsed output has all expected keys with valid types."""
    issues = []
    if "_parse_error" in parsed:
        return False, [f"JSON parse error: {parsed['_parse_error']}"]

    for key in EXPECTED_KEYS:
        if key not in parsed:
            issues.append(f"Missing key: {key}")

    if "potential_type" in parsed:
        valid_types = {"infinite_square_well", "harmonic_oscillator",
                       "finite_square_well", "hydrogen_radial", "double_well"}
        if parsed["potential_type"] not in valid_types:
            issues.append(f"Unknown potential_type: {parsed['potential_type']}")

    if "quantum_number_n" in parsed:
        if not isinstance(parsed["quantum_number_n"], int):
            try:
                parsed["quantum_number_n"] = int(parsed["quantum_number_n"])
            except (ValueError, TypeError):
                issues.append("quantum_number_n is not an integer")

    for float_key in ("mass_au", "energy_au"):
        if float_key in parsed:
            try:
                parsed[float_key] = float(parsed[float_key])
            except (ValueError, TypeError):
                issues.append(f"{float_key} is not a number")

    return len(issues) == 0, issues


def predict(client, model_id: str, image_path: str,
            temperature: float = 0.1, max_tokens: int = 512) -> dict:
    """Run inference on a single image. Returns parsed result dict."""
    data_uri = image_to_data_uri(image_path)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT,
                    },
                ],
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - t0

    raw_text = response.choices[0].message.content
    parsed   = parse_response(raw_text)
    valid, issues = validate_output(parsed)

    return {
        "image_path":   str(image_path),
        "prediction":   parsed,
        "valid":        valid,
        "issues":       issues,
        "raw_response": raw_text,
        "latency_s":    round(elapsed, 3),
        "model":        model_id,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference: PNG image → Schrödinger parameters via LM Studio"
    )
    parser.add_argument("--image",      help="Path to a single PNG image")
    parser.add_argument("--images",     nargs="+", help="Paths to multiple PNG images")
    parser.add_argument("--image_dir",  help="Directory of PNG images to process")
    parser.add_argument("--output",     default=None,
                        help="Write results to this JSONL file (default: stdout)")
    parser.add_argument("--model",      default=None,
                        help="LM Studio model ID (auto-detected if blank)")
    parser.add_argument("--base_url",   default="http://localhost:1234/v1")
    parser.add_argument("--api_key",    default="lm-studio")
    parser.add_argument("--temperature",type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int,   default=512)
    parser.add_argument("--config",     default=None,
                        help="Optional finetune_config.yaml to read LM Studio settings from")
    args = parser.parse_args()

    # Config overrides
    if args.config:
        cfg = load_config(args.config)
        lms = cfg.get("lmstudio", {})
        if not args.model:
            args.model = lms.get("model") or None
        args.base_url    = args.base_url or lms.get("base_url", "http://localhost:1234/v1")
        args.temperature = args.temperature or lms.get("temperature", 0.1)
        args.max_tokens  = args.max_tokens  or lms.get("max_tokens",  512)

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.images:
        image_paths.extend(args.images)
    if args.image_dir:
        image_paths.extend(sorted(Path(args.image_dir).glob("pdf_*.png")))

    if not image_paths:
        parser.error("Provide --image, --images, or --image_dir")

    # Connect to LM Studio
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    print(f"\n── Connecting to LM Studio at {args.base_url} ──────────────────")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    model_id = args.model or get_model_id(client)

    # Run inference
    print(f"\n── Running inference on {len(image_paths)} image(s) ─────────────")
    results = []
    for i, img_path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] {img_path} ... ", end="", flush=True)
        result = predict(
            client, model_id, img_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        status = "✓" if result["valid"] else f"✗ {result['issues']}"
        print(f"{result['latency_s']:.2f}s  {status}")
        results.append(result)

    # Output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n✓  Results written to {out_path.resolve()}")
    else:
        print(f"\n── Results ──────────────────────────────────────────────────")
        for r in results:
            print(f"\nImage : {r['image_path']}")
            print(f"Valid : {r['valid']}  ({r['latency_s']}s)")
            if r["issues"]:
                print(f"Issues: {r['issues']}")
            print(f"Output: {json.dumps(r['prediction'], indent=2)}")


if __name__ == "__main__":
    main()
