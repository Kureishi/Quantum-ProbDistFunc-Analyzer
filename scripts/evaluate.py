#!/usr/bin/env python3
"""
evaluate.py
===========
Benchmark the fine-tuned model's parameter extraction accuracy on the
held-out test set against ground-truth JSON sidecars.

Metrics reported:
  - potential_type accuracy          (exact match %)
  - quantum_number_n accuracy        (exact match %)
  - energy_au MAE / RMSE / R²        (regression)
  - potential_params MAE per key     (e.g. L, V0, omega, a, b)
  - Overall valid JSON rate
  - Per-potential-type breakdown

Usage:
  python scripts/evaluate.py --dataset_dir ./dataset --output_dir ./eval_results
  python scripts/evaluate.py --predictions ./results.jsonl --output_dir ./eval_results
  python scripts/evaluate.py --dataset_dir ./dataset --model my-model-id
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def mae(pred_vals: list, true_vals: list) -> float:
    if not pred_vals:
        return float("nan")
    return sum(abs(p - t) for p, t in zip(pred_vals, true_vals)) / len(pred_vals)


def rmse(pred_vals: list, true_vals: list) -> float:
    if not pred_vals:
        return float("nan")
    mse = sum((p - t) ** 2 for p, t in zip(pred_vals, true_vals)) / len(pred_vals)
    return mse ** 0.5


def r2(pred_vals: list, true_vals: list) -> float:
    if not pred_vals or len(pred_vals) < 2:
        return float("nan")
    mean_t = sum(true_vals) / len(true_vals)
    ss_res = sum((p - t) ** 2 for p, t in zip(pred_vals, true_vals))
    ss_tot = sum((t - mean_t) ** 2 for t in true_vals)
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Result collection ──────────────────────────────────────────────────────────

def collect_results(dataset_dir: Path, predictions_file: Path | None,
                    model_id: str, base_url: str, api_key: str,
                    temperature: float, max_tokens: int) -> list[dict]:
    """
    Returns list of dicts, each with keys:
      sample_id, image_path, true_*, pred_*, valid, issues, latency_s
    """
    # If a pre-computed predictions file is provided, load it
    if predictions_file and predictions_file.exists():
        print(f"  Loading predictions from {predictions_file}")
        records = []
        with open(predictions_file, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line.strip()))
        # Merge with ground truth
        results = []
        for rec in records:
            img_path = Path(rec["image_path"])
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue
            true_meta = json.loads(json_path.read_text(encoding="utf-8"))
            pred      = rec.get("prediction", {})
            results.append(_merge(true_meta, pred, img_path,
                                  rec.get("valid", False),
                                  rec.get("issues", []),
                                  rec.get("latency_s", 0.0)))
        return results

    # Otherwise run inference now
    from inference import predict
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)

    if not model_id:
        models = client.models.list()
        if not models.data:
            print("ERROR: No models loaded in LM Studio.")
            sys.exit(1)
        model_id = models.data[0].id
        print(f"  Auto-detected model: {model_id}")

    json_files = sorted(dataset_dir.glob("pdf_*.json"))
    print(f"  Running inference on {len(json_files)} test samples…")
    results = []
    for i, jf in enumerate(json_files):
        true_meta = json.loads(jf.read_text(encoding="utf-8"))
        img_path  = dataset_dir / true_meta["image_file"]
        if not img_path.exists():
            continue
        print(f"  [{i+1}/{len(json_files)}] {img_path.name} … ", end="", flush=True)
        rec = predict(client, model_id, img_path, temperature, max_tokens)
        print(f"{'✓' if rec['valid'] else '✗'}  {rec['latency_s']:.2f}s")
        results.append(_merge(true_meta, rec["prediction"], img_path,
                              rec["valid"], rec["issues"], rec["latency_s"]))
    return results


def _merge(true_meta: dict, pred: dict, img_path: Path,
           valid: bool, issues: list, latency: float) -> dict:
    true_pp = true_meta.get("potential_params", {})
    pred_pp = pred.get("potential_params", {}) if isinstance(pred.get("potential_params"), dict) else {}
    return {
        "sample_id":           true_meta.get("sample_id", img_path.stem),
        "image_path":          str(img_path),
        "true_potential_type": true_meta.get("potential_type", ""),
        "true_n":              true_meta.get("quantum_number_n", -1),
        "true_mass":           true_meta.get("mass_au", 1.0),
        "true_energy":         true_meta.get("energy_au", 0.0),
        "true_params":         true_pp,
        "pred_potential_type": pred.get("potential_type", ""),
        "pred_n":              pred.get("quantum_number_n"),
        "pred_mass":           safe_float(pred.get("mass_au")),
        "pred_energy":         safe_float(pred.get("energy_au")),
        "pred_params":         pred_pp,
        "valid":               valid,
        "issues":              issues,
        "latency_s":           latency,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    valid_results = [r for r in results if r["valid"]]
    n_valid = len(valid_results)

    # ── Classification metrics ────────────────────────────────────────────────
    type_correct = sum(1 for r in valid_results
                       if r["pred_potential_type"] == r["true_potential_type"])
    n_correct    = sum(1 for r in valid_results
                       if r["pred_n"] == r["true_n"])

    # ── Energy regression metrics ─────────────────────────────────────────────
    e_pairs = [(r["pred_energy"], r["true_energy"])
               for r in valid_results if r["pred_energy"] is not None]
    e_preds = [p for p, _ in e_pairs]
    e_trues = [t for _, t in e_pairs]

    # ── potential_params regression per key ───────────────────────────────────
    param_data = defaultdict(lambda: {"preds": [], "trues": []})
    for r in valid_results:
        for k, tv in r["true_params"].items():
            pv = safe_float(r["pred_params"].get(k))
            tv_f = safe_float(tv)
            if pv is not None and tv_f is not None:
                param_data[k]["preds"].append(pv)
                param_data[k]["trues"].append(tv_f)

    param_metrics = {}
    for k, data in param_data.items():
        param_metrics[k] = {
            "n":    len(data["preds"]),
            "mae":  round(mae(data["preds"], data["trues"]), 5),
            "rmse": round(rmse(data["preds"], data["trues"]), 5),
            "r2":   round(r2(data["preds"], data["trues"]), 4),
        }

    # ── Per-potential breakdown ───────────────────────────────────────────────
    per_potential = defaultdict(lambda: {"total": 0, "type_ok": 0, "n_ok": 0})
    for r in valid_results:
        pt = r["true_potential_type"]
        per_potential[pt]["total"] += 1
        if r["pred_potential_type"] == pt:
            per_potential[pt]["type_ok"] += 1
        if r["pred_n"] == r["true_n"]:
            per_potential[pt]["n_ok"] += 1

    per_potential_summary = {
        pt: {
            "total":          d["total"],
            "type_acc_%":     round(100 * d["type_ok"] / d["total"], 1) if d["total"] else 0,
            "n_acc_%":        round(100 * d["n_ok"]    / d["total"], 1) if d["total"] else 0,
        }
        for pt, d in per_potential.items()
    }

    return {
        "total_samples":    n,
        "valid_json_%":     round(100 * n_valid / n, 1),
        "potential_type_acc_%": round(100 * type_correct / n_valid, 1) if n_valid else 0,
        "quantum_n_acc_%":  round(100 * n_correct  / n_valid, 1) if n_valid else 0,
        "energy_mae_au":    round(mae(e_preds, e_trues), 6) if e_preds else None,
        "energy_rmse_au":   round(rmse(e_preds, e_trues), 6) if e_preds else None,
        "energy_r2":        round(r2(e_preds, e_trues), 4) if e_preds else None,
        "avg_latency_s":    round(sum(r["latency_s"] for r in results) / n, 3),
        "potential_params": param_metrics,
        "per_potential":    per_potential_summary,
    }


def print_report(metrics: dict):
    w = 52
    print("\n" + "═" * w)
    print(" EVALUATION REPORT")
    print("═" * w)
    print(f" Total samples       : {metrics['total_samples']}")
    print(f" Valid JSON output   : {metrics['valid_json_%']}%")
    print(f" Potential type acc. : {metrics['potential_type_acc_%']}%")
    print(f" Quantum number acc. : {metrics['quantum_n_acc_%']}%")
    if metrics.get("energy_mae_au") is not None:
        print(f" Energy MAE          : {metrics['energy_mae_au']} a.u.")
        print(f" Energy RMSE         : {metrics['energy_rmse_au']} a.u.")
        print(f" Energy R²           : {metrics['energy_r2']}")
    print(f" Avg latency         : {metrics['avg_latency_s']}s / image")

    if metrics.get("potential_params"):
        print("\n── Potential-parameter regression ──────────────────────────")
        for k, m in metrics["potential_params"].items():
            print(f"  {k:10s}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.3f}  (n={m['n']})")

    if metrics.get("per_potential"):
        print("\n── Per-potential breakdown ──────────────────────────────────")
        print(f"  {'Potential':<26}  {'N':>4}  {'Type%':>6}  {'n%':>6}")
        print(f"  {'─'*26}  {'─'*4}  {'─'*6}  {'─'*6}")
        for pt, m in metrics["per_potential"].items():
            print(f"  {pt:<26}  {m['total']:>4}  {m['type_acc_%']:>5.1f}%  {m['n_acc_%']:>5.1f}%")

    print("═" * w + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Schrödinger test set.")
    parser.add_argument("--dataset_dir",   default=None,
                        help="Dataset directory (runs inference if --predictions not given)")
    parser.add_argument("--predictions",   default=None,
                        help="Pre-computed predictions JSONL (from inference.py --output)")
    parser.add_argument("--output_dir",    default="./eval_results")
    parser.add_argument("--model",         default=None)
    parser.add_argument("--base_url",      default="http://localhost:1234/v1")
    parser.add_argument("--api_key",       default="lm-studio")
    parser.add_argument("--temperature",   type=float, default=0.1)
    parser.add_argument("--max_tokens",    type=int,   default=512)
    parser.add_argument("--config",        default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        lms = cfg.get("lmstudio", {})
        args.model      = args.model      or lms.get("model") or None
        args.base_url   = args.base_url   or lms.get("base_url", "http://localhost:1234/v1")
        args.temperature= args.temperature or lms.get("temperature", 0.1)
        args.max_tokens = args.max_tokens  or lms.get("max_tokens",  512)

    if not args.dataset_dir and not args.predictions:
        parser.error("Provide --dataset_dir or --predictions")

    print("\n── Collecting results ───────────────────────────────────────────")
    results = collect_results(
        dataset_dir      = Path(args.dataset_dir) if args.dataset_dir else Path("."),
        predictions_file = Path(args.predictions) if args.predictions else None,
        model_id         = args.model,
        base_url         = args.base_url,
        api_key          = args.api_key,
        temperature      = args.temperature,
        max_tokens       = args.max_tokens,
    )

    print(f"\n── Computing metrics on {len(results)} samples ─────────────────")
    metrics = evaluate(results)
    print_report(metrics)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved  → {metrics_path.resolve()}")

    results_path = output_dir / "detailed_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Results saved  → {results_path.resolve()}")


if __name__ == "__main__":
    main()
