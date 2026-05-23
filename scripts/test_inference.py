#!/usr/bin/env python3
"""
test_inference.py
=================
Quick end-to-end test of the fine-tuned model served via LM Studio.

Sends a sample |ψ(x)|² image to the LM Studio local server and prints
the predicted Schrödinger parameters alongside the ground truth.

Requirements:
  - LM Studio running with the fine-tuned model loaded
  - Local server started (Developer tab → Start Server, default port 1234)
  - pip install openai

Usage:
  # Test with a single built-in example
  python test_inference.py

  # Test with your own image
  python test_inference.py --image path/to/your/image.png

  # Test all images in a directory
  python test_inference.py --image_dir ./dataset

  # Use a different server or model
  python test_inference.py --base_url http://localhost:1234/v1 --model my-model-id
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path


# ── Built-in example (base64-encoded 1×1 placeholder replaced at runtime) ────
# We ship a small hard-coded ground truth so the script works standalone
# even without the dataset directory present.
BUILTIN_EXAMPLE = {
    "potential_type":   "infinite_square_well",
    "quantum_number_n": 1,
    "mass_au":          1.0,
    "energy_au":        4.934802,
    "potential_params": {"L": 1.0},
}

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


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    suffix = Path(path).suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg"}.get(suffix, "image/png")
    return f"data:{mime};base64,{b64}"


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines[1:] if not l.startswith("```")).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"_error": "Could not parse response", "_raw": text}


def load_ground_truth(image_path: str) -> dict | None:
    """Load .json sidecar if it exists alongside the image."""
    json_path = Path(image_path).with_suffix(".json")
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return None


def compare(pred: dict, truth: dict) -> list[str]:
    """Return a list of comparison lines."""
    lines = []
    checks = [
        ("potential_type",   lambda p, t: p == t,
         lambda p, t: f"'{p}' vs '{t}'"),
        ("quantum_number_n", lambda p, t: int(p) == int(t),
         lambda p, t: f"{p} vs {t}"),
        ("energy_au",        lambda p, t: abs(float(p) - float(t)) < 0.5,
         lambda p, t: f"{float(p):.4f} vs {float(t):.4f}  (Δ={abs(float(p)-float(t)):.4f})"),
        ("mass_au",          lambda p, t: abs(float(p) - float(t)) < 0.01,
         lambda p, t: f"{float(p):.3f} vs {float(t):.3f}"),
    ]
    for key, eq_fn, fmt_fn in checks:
        p_val = pred.get(key)
        t_val = truth.get(key)
        if p_val is None:
            lines.append(f"  {'✗':2}  {key:<20}  MISSING in prediction")
            continue
        try:
            ok = eq_fn(p_val, t_val)
            icon = "✓" if ok else "✗"
            lines.append(f"  {icon:2}  {key:<20}  {fmt_fn(p_val, t_val)}")
        except (TypeError, ValueError):
            lines.append(f"  {'?':2}  {key:<20}  {p_val} vs {t_val}")

    # potential_params sub-keys
    pred_pp  = pred.get("potential_params",  {}) or {}
    truth_pp = truth.get("potential_params", {}) or {}
    for k in truth_pp:
        pv = pred_pp.get(k)
        tv = truth_pp[k]
        if pv is None:
            lines.append(f"  {'✗':2}  params.{k:<15}  MISSING")
        else:
            try:
                ok = abs(float(pv) - float(tv)) < 0.1
                icon = "✓" if ok else "✗"
                lines.append(f"  {icon:2}  params.{k:<15}  {float(pv):.4g} vs {float(tv):.4g}")
            except (TypeError, ValueError):
                lines.append(f"  {'?':2}  params.{k:<15}  {pv} vs {tv}")
    return lines


def run_test(client, model_id: str, image_path: str,
             ground_truth: dict | None, temperature: float, max_tokens: int):
    """Run one inference call and print a formatted result block."""
    width = 62
    print(f"\n{'─'*width}")
    print(f"  Image : {Path(image_path).name}")
    if ground_truth:
        gt_label = ground_truth.get("potential_label") or ground_truth.get("potential_type", "?")
        print(f"  True  : {gt_label}  n={ground_truth.get('quantum_number_n')}  "
              f"E={ground_truth.get('energy_au', 0):.4f} a.u.")
    print(f"{'─'*width}")

    data_uri = encode_image(image_path)
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text",      "text": USER_PROMPT},
                    ],
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - t0
        raw = response.choices[0].message.content
        pred = parse_json_response(raw)
        error = pred.get("_error")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        pred = {}
        error = str(e)
        raw = ""

    if error:
        print(f"  ✗  ERROR: {error}")
        if raw:
            print(f"  Raw response: {raw[:200]}")
        return

    print(f"  Latency : {elapsed:.2f}s")
    print(f"\n  Predicted parameters:")
    print(f"  {json.dumps(pred, indent=4).replace(chr(10), chr(10)+'  ')}")

    if ground_truth:
        print(f"\n  Accuracy vs ground truth:")
        for line in compare(pred, ground_truth):
            print(line)

    print(f"{'─'*width}")


def main():
    parser = argparse.ArgumentParser(
        description="Test the fine-tuned Schrödinger model via LM Studio."
    )
    parser.add_argument("--image",      default=None,
                        help="Path to a single PNG image to test")
    parser.add_argument("--image_dir",  default=None,
                        help="Directory of PNG images to test (uses first 5)")
    parser.add_argument("--base_url",   default="http://localhost:1234/v1",
                        help="LM Studio server URL (default: http://localhost:1234/v1)")
    parser.add_argument("--model",      default=None,
                        help="Model ID (auto-detected from /v1/models if blank)")
    parser.add_argument("--temperature",type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int,   default=512)
    parser.add_argument("--n",          type=int,   default=5,
                        help="Max images to test from --image_dir (default: 5)")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Schrödinger Model — Inference Test")
    print(f"{'='*62}")
    print(f"  Server : {args.base_url}")

    client = OpenAI(base_url=args.base_url, api_key="lm-studio")

    try:
        models = client.models.list()
    except Exception as e:
        print(f"\n✗  Cannot connect to LM Studio at {args.base_url}")
        print(f"   Error: {e}")
        print(f"\n   Make sure LM Studio is running and the server is started:")
        print(f"   Developer tab → Start Server")
        sys.exit(1)

    if not models.data:
        print("✗  No model loaded in LM Studio. Load your fine-tuned model first.")
        sys.exit(1)

    model_id = args.model or models.data[0].id
    print(f"  Model  : {model_id}")

    # ── Collect images ────────────────────────────────────────────────────────
    test_cases = []   # list of (image_path, ground_truth_or_None)

    if args.image:
        gt = load_ground_truth(args.image)
        test_cases.append((args.image, gt))

    if args.image_dir:
        pngs = sorted(Path(args.image_dir).glob("pdf_*.png"))[:args.n]
        for p in pngs:
            gt = load_ground_truth(str(p))
            test_cases.append((str(p), gt))

    # If nothing specified, look for the dataset next to this script
    if not test_cases:
        default_dirs = [
            Path("./dataset"),
            Path("./probability_distribution_dataset"),
            Path(__file__).parent.parent / "dataset",
            Path(__file__).parent.parent / "probability_distribution_dataset",
        ]
        for d in default_dirs:
            pngs = sorted(d.glob("pdf_*.png"))[:args.n]
            if pngs:
                for p in pngs:
                    gt = load_ground_truth(str(p))
                    test_cases.append((str(p), gt))
                print(f"  Dataset: {d.resolve()}")
                break

    if not test_cases:
        print(f"\n✗  No images found.")
        print(f"   Pass --image <path>  or  --image_dir <dir>")
        print(f"   Or place your dataset in ./dataset/")
        sys.exit(1)

    print(f"  Testing: {len(test_cases)} image(s)")

    # ── Run tests ─────────────────────────────────────────────────────────────
    for img_path, gt in test_cases:
        run_test(client, model_id, img_path, gt,
                 args.temperature, args.max_tokens)

    print(f"\n{'='*62}")
    print(f"  Done.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
