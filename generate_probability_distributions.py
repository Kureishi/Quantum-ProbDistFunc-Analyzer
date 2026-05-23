"""
Schrödinger Equation — Probability Density Function (PDF) Dataset Generator
============================================================================
Generates a dataset of PNG images showing |ψ(x)|² (the probability
density function / position probability distribution) for solutions to the
time-independent Schrödinger equation.

Each image is a single focused plot of the PDF |ψ(x)|² with:
  - Annotated peak value
  - Energy eigenvalue and parameters in the corner
  - Colour-coded by potential type
  - Normalisation check: ∫|ψ|²dx ≈ 1

Supported potentials:
  1. Infinite Square Well (Particle in a Box)
  2. Quantum Harmonic Oscillator
  3. Finite Square Well
  4. Hydrogen Atom (radial)
  5. Symmetric Double Well

Usage:
  python generate_probability_distributions.py
  python generate_probability_distributions.py --preset small    # ~15 samples
  python generate_probability_distributions.py --preset medium   # ~50 samples
  python generate_probability_distributions.py --preset large    # ~200 samples
  python generate_probability_distributions.py --custom          # interactive
  python generate_probability_distributions.py --outdir my_dir
"""

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image, PngImagePlugin
from scipy.linalg import eigh_tridiagonal
from scipy.special import hermite, genlaguerre, factorial

# ── Constants (atomic units) ──────────────────────────────────────────────────
HBAR = 1.0
OUTPUT_DIR = Path("probability_distribution_dataset")

# ── Per-potential colour palettes (base, fill-lo, fill-hi) ───────────────────
PALETTE = {
    "infinite_square_well": ("#0EA5E9", "#BAE6FD", "#0369A1"),   # sky blue
    "harmonic_oscillator":  ("#8B5CF6", "#EDE9FE", "#5B21B6"),   # violet
    "finite_square_well":   ("#10B981", "#D1FAE5", "#065F46"),   # emerald
    "hydrogen_radial":      ("#F59E0B", "#FEF3C7", "#92400E"),   # amber
    "double_well":          ("#EF4444", "#FEE2E2", "#991B1B"),   # red
}

POTENTIAL_LABELS = {
    "infinite_square_well": "Infinite Square Well",
    "harmonic_oscillator":  "Quantum Harmonic Oscillator",
    "finite_square_well":   "Finite Square Well",
    "hydrogen_radial":      "Hydrogen Atom (Radial)",
    "double_well":          "Symmetric Double Well",
}


# ─────────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────────

def _trapz(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)


def _fdm_solve(x, V, mass, n_states=10):
    N = len(x)
    dx = x[1] - x[0]
    diag = HBAR**2 / (mass * dx**2) + V
    off  = -HBAR**2 / (2 * mass * dx**2) * np.ones(N - 1)
    k = min(n_states, N - 2)
    E, psi = eigh_tridiagonal(diag, off, eigvals_only=False,
                               select='i', select_range=(0, k - 1))
    for i in range(psi.shape[1]):
        norm = _trapz(psi[:, i]**2, x)
        psi[:, i] /= np.sqrt(norm)
    return E, psi


def solve_infinite_square_well(n, L, mass, N=600):
    x = np.linspace(0, L, N)
    psi = np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
    E = (n * np.pi * HBAR)**2 / (2 * mass * L**2)
    return x, psi, E


def solve_harmonic_oscillator(n, omega, mass, N=600):
    x_max = 5 * np.sqrt(HBAR / (mass * omega))
    x = np.linspace(-x_max, x_max, N)
    xi = np.sqrt(mass * omega / HBAR) * x
    Hn = hermite(n)
    norm = (1 / np.sqrt(2**n * float(factorial(n)))) * \
           (mass * omega / (np.pi * HBAR))**0.25
    psi = norm * np.exp(-xi**2 / 2) * Hn(xi)
    E = HBAR * omega * (n + 0.5)
    return x, psi, E


def solve_finite_square_well(n, L, V0, mass, N=1000):
    x = np.linspace(-2 * L, 2 * L, N)
    V = np.where(np.abs(x) <= L / 2, 0.0, V0)
    E_arr, psi_arr = _fdm_solve(x, V, mass, n_states=max(n, 5))
    idx = min(n - 1, len(E_arr) - 1)
    return x, psi_arr[:, idx], E_arr[idx]


def solve_hydrogen_radial(n, l, N=800):
    a0 = 1.0
    r_max = 5 * n**2 * a0 * 3
    r = np.linspace(0.01, r_max, N)
    rho = 2 * r / (n * a0)
    Laguerre = genlaguerre(n - l - 1, 2 * l + 1)
    norm = np.sqrt(
        (2 / (n * a0))**3 *
        float(factorial(n - l - 1)) /
        (2 * n * float(factorial(n + l))**3)
    )
    R = norm * np.exp(-rho / 2) * rho**l * Laguerre(rho)
    E_au = -0.5 / n**2
    return r, R, E_au


def solve_double_well(n, a, b, V0, mass, N=1000):
    x_max = a * 3
    x = np.linspace(-x_max, x_max, N)
    V = np.clip(V0 * (x**2 - a**2)**2 / b, 0, V0 * 5)
    E_arr, psi_arr = _fdm_solve(x, V, mass, n_states=max(n, 5))
    idx = min(n - 1, len(E_arr) - 1)
    return x, psi_arr[:, idx], E_arr[idx]


def solve(params):
    pt = params["potential_type"]
    n  = params["quantum_number_n"]
    m  = params["mass"]
    pp = json.loads(params["potential_params"])
    if pt == "infinite_square_well":
        return solve_infinite_square_well(n, pp["L"], m)
    elif pt == "harmonic_oscillator":
        return solve_harmonic_oscillator(n, pp["omega"], m)
    elif pt == "finite_square_well":
        return solve_finite_square_well(n, pp["L"], pp["V0"], m)
    elif pt == "hydrogen_radial":
        return solve_hydrogen_radial(n, pp.get("l", 0))
    elif pt == "double_well":
        return solve_double_well(n, pp["a"], pp["b"], pp["V0"], m)
    else:
        raise ValueError(f"Unknown potential: {pt}")


# ─────────────────────────────────────────────────────────────────────────────
# PDF plot — probability distribution only
# ─────────────────────────────────────────────────────────────────────────────

def _make_gradient_fill(ax, x, y, color_lo, color_hi):
    """Vertical gradient fill under the curve."""
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    n_bands = 200
    y_max = y.max()
    for k in range(n_bands):
        t0 = k / n_bands
        t1 = (k + 1) / n_bands
        level0 = t0 * y_max
        level1 = t1 * y_max
        mask = y >= level0
        if not mask.any():
            continue
        alpha = 0.55 * (1 - t0)**1.4
        c = tuple(
            (1 - t0) * np.array(matplotlib.colors.to_rgb(color_lo)) +
            t0 * np.array(matplotlib.colors.to_rgb(color_hi))
        )
        ax.fill_between(x, level0, np.minimum(y, level1),
                        where=mask, color=c, alpha=alpha,
                        linewidth=0, rasterized=True)


# ─────────────────────────────────────────────────────────────────────────────
# Caption builder  (natural-language + structured tags for fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable phrases for each potential
_POTENTIAL_PHRASES = {
    "infinite_square_well": (
        "a particle confined in an infinite square potential well of width {L} atomic units"
    ),
    "harmonic_oscillator": (
        "a quantum harmonic oscillator with angular frequency omega equals {omega} atomic units"
    ),
    "finite_square_well": (
        "a finite square potential well of width {L} atomic units "
        "and barrier height V0 equals {V0} atomic units"
    ),
    "hydrogen_radial": (
        "the hydrogen atom radial wavefunction with angular momentum quantum number l equals {l}"
    ),
    "double_well": (
        "a symmetric double-well potential with well separation a equals {a} atomic units, "
        "shape factor b equals {b}, and barrier height V0 equals {V0} atomic units"
    ),
}

_NODE_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven"]

def _node_count(n: int, pt: str) -> int:
    """Number of nodes in the PDF (|ψ|² has n-1 nodes for most potentials)."""
    if pt == "harmonic_oscillator":
        return n          # quantum number is 0-indexed; n nodes in ψ², = n peaks
    return n - 1


def build_caption(params: dict, E: float, prob: np.ndarray, x: np.ndarray) -> str:
    """
    Produce a rich natural-language caption suitable as a diffusion model
    training label.  Format:
        <natural sentence>, quantum number n=<n>, energy eigenvalue <E> a.u.,
        <node description>, <normalisation>, <style tags>

    Also returned as a second value: a compact tag string for the .txt sidecar.
    """
    pt  = params["potential_type"]
    n   = params["quantum_number_n"]
    m   = params["mass"]
    pp  = json.loads(params["potential_params"])
    sid = params["sample_id"]
    label = POTENTIAL_LABELS[pt]

    # Fill potential-specific phrase
    phrase = _POTENTIAL_PHRASES[pt].format(**pp)

    # Node / peak description
    nodes = _node_count(n, pt)
    peak_count = nodes + 1
    node_str = _NODE_WORDS[nodes] if nodes < len(_NODE_WORDS) else str(nodes)
    peak_str  = _NODE_WORDS[peak_count] if peak_count < len(_NODE_WORDS) else str(peak_count)

    # Normalisation
    norm_val = _trapz(prob, x)

    # Energy sign description
    e_desc = "bound state" if E < 0 else "positive energy state"

    # ── Long natural-language caption ─────────────────────────────────────────
    caption = (
        f"Probability density function |ψ(x)|² of {phrase}, "
        f"principal quantum number n equals {n}, "
        f"particle mass {m:.2g} atomic units, "
        f"energy eigenvalue {E:.5f} atomic units ({e_desc}), "
        f"the curve has {node_str} node{'s' if nodes != 1 else ''} "
        f"and {peak_str} peak{'s' if peak_count != 1 else ''}, "
        f"normalisation integral {norm_val:.4f}, "
        f"plotted in atomic units with a dark background, "
        f"scientific data visualisation, quantum mechanics"
    )

    return caption


def build_tags(params: dict, E: float) -> str:
    """
    Compact comma-separated tag string — used as the sidecar .txt file and
    embedded as a separate PNG tEXt chunk (key='tags').
    Compatible with kohya-ss / EveryDream2 / SimpleTuner tag formats.
    """
    pt  = params["potential_type"]
    n   = params["quantum_number_n"]
    pp  = json.loads(params["potential_params"])
    label = POTENTIAL_LABELS[pt].lower().replace(" ", "_")

    tags = [
        "probability density function",
        "|psi(x)|^2",
        "quantum mechanics",
        "schrodinger equation",
        label,
        f"n={n}",
        f"energy={E:.4f}_au",
        "dark background",
        "scientific plot",
        "atomic units",
        "wavefunction",
    ]
    for k, v in pp.items():
        tags.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")

    return ", ".join(tags)


def embed_metadata_and_write_sidecar(
    png_path: Path,
    caption: str,
    tags: str,
    params: dict,
    E: float,
):
    """
    1. Embed caption + tags + full JSON params into the PNG tEXt chunks
       (readable by Pillow, ExifTool, and kohya-ss metadata loaders).
    2. Write a <stem>.txt sidecar with the caption (EveryDream2 / SimpleTuner
       / most kohya-ss dataset configs read this automatically).
    3. Write a <stem>.json sidecar with structured metadata (for custom
       loaders and dataset inspection).
    """
    # ── PNG tEXt metadata ────────────────────────────────────────────────────
    img = Image.open(png_path)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", caption)          # A1111 / kohya convention
    meta.add_text("caption",    caption)
    meta.add_text("tags",       tags)
    meta.add_text("sample_id",  params["sample_id"])
    meta.add_text("potential",  params["potential_type"])
    meta.add_text("n",          str(params["quantum_number_n"]))
    meta.add_text("energy_au",  f"{E:.6f}")
    meta.add_text("mass_au",    f"{params['mass']:.4f}")
    meta.add_text("potential_params", params["potential_params"])
    img.save(png_path, pnginfo=meta)

    stem = png_path.with_suffix("")

    # ── .txt sidecar (caption only) ──────────────────────────────────────────
    stem.with_suffix(".txt").write_text(caption, encoding="utf-8")

    # ── .json sidecar (full structured metadata) ─────────────────────────────
    sidecar = {
        "sample_id":        params["sample_id"],
        "caption":          caption,
        "tags":             tags,
        "potential_type":   params["potential_type"],
        "potential_label":  POTENTIAL_LABELS[params["potential_type"]],
        "quantum_number_n": params["quantum_number_n"],
        "mass_au":          params["mass"],
        "energy_au":        E,
        "potential_params": json.loads(params["potential_params"]),
        "units":            "atomic units (hbar=1, m_e=1, a_0=1)",
        "image_file":       png_path.name,
    }
    stem.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )


def generate_image(params: dict, out_path: Path) -> float:
    x, psi, E = solve(params)
    prob = psi**2

    pt     = params["potential_type"]
    n      = params["quantum_number_n"]
    pp     = json.loads(params["potential_params"])
    sid    = params["sample_id"]
    label  = POTENTIAL_LABELS.get(pt, pt)
    col_line, col_lo, col_hi = PALETTE.get(pt, ("#334155", "#CBD5E1", "#0F172A"))

    x_label = "r (a.u.)" if pt == "hydrogen_radial" else "x (a.u.)"

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    fig.patch.set_facecolor("#0F172A")          # dark navy background
    ax.set_facecolor("#0F172A")

    # Subtle grid
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.grid(which="major", color="#1E293B", linewidth=0.8, linestyle="-")
    ax.grid(which="minor", color="#1E293B", linewidth=0.3, linestyle="-")

    # Gradient fill
    _make_gradient_fill(ax, x, prob, col_lo, col_hi)

    # Main curve
    ax.plot(x, prob, color=col_line, linewidth=2.0, zorder=5,
            solid_capstyle="round")

    # Thin baseline
    ax.axhline(0, color="#334155", linewidth=0.8, zorder=4)

    # Peak annotation
    peak_i = np.argmax(prob)
    ax.annotate(
        f"peak = {prob[peak_i]:.4f}",
        xy=(x[peak_i], prob[peak_i]),
        xytext=(x[peak_i] + (x[-1] - x[0]) * 0.04, prob[peak_i] * 1.06),
        fontsize=7.5,
        color="#CBD5E1",
        arrowprops=dict(arrowstyle="-", color="#475569", lw=0.8),
        va="bottom",
    )

    # Axes styling
    ax.tick_params(colors="#64748B", labelsize=8, length=4, width=0.6)
    for spine in ax.spines.values():
        spine.set_color("#1E293B")
        spine.set_linewidth(0.8)

    ax.set_xlabel(x_label, color="#94A3B8", fontsize=10, labelpad=8)
    ax.set_ylabel("|ψ(x)|²  (probability density)", color="#94A3B8",
                  fontsize=10, labelpad=8)
    ax.tick_params(axis="both", colors="#64748B")

    # y lower bound exactly 0
    ax.set_ylim(bottom=0)

    # ── Title block ───────────────────────────────────────────────────────────
    fig.text(
        0.06, 0.965,
        "PROBABILITY DISTRIBUTION FUNCTION",
        color="#64748B", fontsize=7.5, fontweight="bold",
        fontfamily="monospace", va="top",
        transform=fig.transFigure,
    )
    fig.text(
        0.06, 0.935,
        label,
        color="#F1F5F9", fontsize=15, fontweight="bold",
        va="top", transform=fig.transFigure,
    )
    # Quantum number badge
    badge_x = 0.06 + len(label) * 0.012 + 0.01
    fig.text(
        badge_x, 0.942,
        f"  n = {n}  ",
        color=col_line, fontsize=9, fontweight="bold",
        va="top", transform=fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E293B",
                  edgecolor=col_line, linewidth=1.2),
    )

    # ── Right-side parameter block ────────────────────────────────────────────
    param_lines = [f"E = {E:.5f} a.u.", f"mass = {params['mass']:.2f} a.u."]
    for k, v in pp.items():
        param_lines.append(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}")
    param_lines.append(f"∫|ψ|²dx = {_trapz(prob, x):.4f}")
    param_lines.append(f"ID: {sid}")

    fig.text(
        0.94, 0.965,
        "\n".join(param_lines),
        color="#94A3B8", fontsize=7.8, va="top", ha="right",
        fontfamily="monospace", linespacing=1.6,
        transform=fig.transFigure,
    )

    # ── Colour bar (density bar along x-axis) ────────────────────────────────
    cmap_custom = LinearSegmentedColormap.from_list(
        "prob_cmap", ["#0F172A", col_lo, col_line], N=256
    )
    # Normalise prob to [0, 1] and draw thin heatmap strip at y=-0.01 * ymax
    norm_prob = prob / (prob.max() + 1e-12)
    y_floor = ax.get_ylim()[0]
    strip_h = (ax.get_ylim()[1] - y_floor) * 0.025
    for j in range(len(x) - 1):
        ax.axvspan(x[j], x[j + 1],
                   ymin=0, ymax=strip_h / (ax.get_ylim()[1] - y_floor),
                   color=cmap_custom(norm_prob[j]),
                   alpha=0.9, linewidth=0)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(str(out_path), format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=180)
    plt.close(fig)

    # ── Annotation: embed metadata + write sidecars ───────────────────────────
    caption = build_caption(params, E, prob, x)
    tags    = build_tags(params, E)
    embed_metadata_and_write_sidecar(out_path, caption, tags, params, E)

    return float(E)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid
# ─────────────────────────────────────────────────────────────────────────────

def build_parameter_list(preset="small"):
    if preset == "small":
        cfg = {
            "infinite_square_well": {"n": [1, 2, 3],    "L": [1.0, 2.0]},
            "harmonic_oscillator":  {"n": [0, 1, 2],    "omega": [1.0]},
            "finite_square_well":   {"n": [1, 2],        "L": [2.0], "V0": [5.0]},
            "hydrogen_radial":      {"n": [1, 2],        "l": [0]},
            "double_well":          {"n": [1, 2],        "a": [1.5], "b": [2.0], "V0": [1.0]},
        }
    elif preset == "medium":
        cfg = {
            "infinite_square_well": {"n": [1,2,3,4,5],  "L": [1.0,2.0,3.0]},
            "harmonic_oscillator":  {"n": [0,1,2,3,4],  "omega": [0.5,1.0,2.0]},
            "finite_square_well":   {"n": [1,2,3],       "L": [1.0,2.0], "V0": [3.0,8.0]},
            "hydrogen_radial":      {"n": [1,2,3,4],     "l": [0,1]},
            "double_well":          {"n": [1,2,3],       "a": [1.0,2.0], "b": [1.5], "V0": [1.0]},
        }
    else:  # large
        cfg = {
            "infinite_square_well": {"n": list(range(1,9)), "L": [0.5,1.0,2.0,4.0]},
            "harmonic_oscillator":  {"n": list(range(0,8)), "omega": [0.25,0.5,1.0,2.0,4.0]},
            "finite_square_well":   {"n": [1,2,3,4],        "L": [1.0,2.0,3.0], "V0": [2.0,5.0,10.0,20.0]},
            "hydrogen_radial":      {"n": [1,2,3,4,5],      "l": [0,1,2]},
            "double_well":          {"n": [1,2,3,4],        "a": [1.0,1.5,2.0], "b": [1.0,2.0], "V0": [0.5,1.0,2.0]},
        }

    records = []
    for pt, grid in cfg.items():
        if pt == "infinite_square_well":
            combos = itertools.product(grid["n"], grid["L"])
            for n, L in combos:
                records.append(dict(potential_type=pt, quantum_number_n=n, mass=1.0,
                                    potential_params=json.dumps({"L": L})))
        elif pt == "harmonic_oscillator":
            for n, omega in itertools.product(grid["n"], grid["omega"]):
                records.append(dict(potential_type=pt, quantum_number_n=n, mass=1.0,
                                    potential_params=json.dumps({"omega": omega})))
        elif pt == "finite_square_well":
            for n, L, V0 in itertools.product(grid["n"], grid["L"], grid["V0"]):
                records.append(dict(potential_type=pt, quantum_number_n=n, mass=1.0,
                                    potential_params=json.dumps({"L": L, "V0": V0})))
        elif pt == "hydrogen_radial":
            for n, l in itertools.product(grid["n"], grid["l"]):
                if l >= n:
                    continue
                records.append(dict(potential_type=pt, quantum_number_n=n, mass=1.0,
                                    potential_params=json.dumps({"l": l})))
        elif pt == "double_well":
            for n, a, b, V0 in itertools.product(grid["n"], grid["a"], grid["b"], grid["V0"]):
                records.append(dict(potential_type=pt, quantum_number_n=n, mass=1.0,
                                    potential_params=json.dumps({"a": a, "b": b, "V0": V0})))

    for i, r in enumerate(records):
        r["sample_id"] = f"{i:04d}"
        r["energy"] = 0.0
    return records


def build_custom_list():
    print("\n── Custom parameter builder ──────────────────────────────────")
    print("1=infinite_square_well  2=harmonic_oscillator  3=finite_square_well")
    print("4=hydrogen_radial       5=double_well")
    records = []
    pot_map = {"1": "infinite_square_well", "2": "harmonic_oscillator",
               "3": "finite_square_well",   "4": "hydrogen_radial",
               "5": "double_well"}
    while True:
        c = input("\nSelect potential (1-5) or q to quit: ").strip()
        if c.lower() == "q":
            break
        if c not in pot_map:
            print("Invalid."); continue
        pt = pot_map[c]
        try:
            n = int(input("  n: "))
            m = float(input("  mass [1.0]: ") or "1.0")
            if pt == "infinite_square_well":
                pp = {"L": float(input("  L [1.0]: ") or "1.0")}
            elif pt == "harmonic_oscillator":
                pp = {"omega": float(input("  omega [1.0]: ") or "1.0")}
            elif pt == "finite_square_well":
                pp = {"L": float(input("  L [2.0]: ") or "2.0"),
                      "V0": float(input("  V0 [5.0]: ") or "5.0")}
            elif pt == "hydrogen_radial":
                pp = {"l": int(input(f"  l (0-{n-1}): ") or "0")}
            elif pt == "double_well":
                pp = {"a":  float(input("  a [1.5]: ") or "1.5"),
                      "b":  float(input("  b [2.0]: ") or "2.0"),
                      "V0": float(input("  V0 [1.0]: ") or "1.0")}
            r = dict(potential_type=pt, quantum_number_n=n, mass=m,
                     potential_params=json.dumps(pp),
                     sample_id=f"{len(records):04d}", energy=0.0)
            records.append(r)
            print(f"  ✓ Added {r['sample_id']}")
        except (ValueError, KeyboardInterrupt):
            print("  ✗ Skipped.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(params_list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(params_list)
    manifest = []
    print(f"\nGenerating {total} probability density function images in '{out_dir}/' …\n")

    for i, p in enumerate(params_list):
        fname = f"pdf_{p['sample_id']}.png"
        fpath = out_dir / fname
        caption, tags = "", ""
        try:
            E = generate_image(p, fpath)
            p["energy"] = E
            caption = build_caption(p, E,
                                    np.zeros(2), np.array([0.0, 1.0]))  # placeholder — real written in generate_image
            # Re-read the actual caption from the sidecar that was just written
            txt_path = fpath.with_suffix(".txt")
            if txt_path.exists():
                caption = txt_path.read_text(encoding="utf-8")
            tags = build_tags(p, E)
            status = "ok"
        except Exception as exc:
            status = f"ERROR: {exc}"

        row = dict(p, image_file=fname, caption=caption, tags=tags, status=status)
        manifest.append(row)

        bar = "█" * int(30 * (i + 1) / total) + "░" * (30 - int(30 * (i + 1) / total))
        print(f"  [{bar}] {i+1}/{total}  {fname}  E={p['energy']:.5f}  {status}",
              flush=True)

    # Manifest CSV  (image_file + caption + tags included — load directly into
    # most fine-tuning frameworks as a metadata CSV dataset)
    csv_path = out_dir / "manifest.csv"
    if manifest:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=manifest[0].keys())
            w.writeheader(); w.writerows(manifest)

    # HuggingFace-style metadata.jsonl  (one JSON object per line, each with
    # "file_name" and "text" — drop into a HF ImageFolder dataset directly)
    jsonl_path = out_dir / "metadata.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in manifest:
            if row["status"] == "ok":
                f.write(json.dumps({
                    "file_name": row["image_file"],
                    "text": row["caption"],
                    "tags": row["tags"],
                }) + "\n")

    # Dataset-level metadata JSON
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump({
            "total_samples": total,
            "ok_samples": sum(1 for r in manifest if r["status"] == "ok"),
            "output_format": "PNG image @ 180 dpi",
            "annotation_formats": [
                "PNG tEXt chunks: 'parameters', 'caption', 'tags' (kohya-ss / A1111)",
                ".txt sidecar: caption (EveryDream2 / SimpleTuner / kohya-ss)",
                ".json sidecar: full structured metadata",
                "manifest.csv: tabular, all fields",
                "metadata.jsonl: HuggingFace ImageFolder format",
            ],
            "caption_fields": ["parameters", "caption", "tags", "sample_id",
                               "potential", "n", "energy_au", "mass_au",
                               "potential_params"],
            "output": "probability_density_|psi|^2",
            "potentials": list({r["potential_type"] for r in manifest}),
            "units": "atomic units (hbar=1, m_e=1, a_0=1)",
        }, f, indent=2)

    ok = sum(1 for r in manifest if r["status"] == "ok")
    print(f"\n✓  {ok}/{total} images generated.")
    print(f"   Output dir   : {out_dir.resolve()}")
    print(f"   Manifest CSV : {csv_path.resolve()}")
    print(f"   HF JSONL     : {jsonl_path.resolve()}")
    print(f"   Each PNG also has a .txt and .json sidecar.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate PNG images of |ψ(x)|² probability density functions for Schrödinger equation solutions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--preset", choices=["small", "medium", "large"],
                        default="small")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--outdir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    if args.custom:
        params_list = build_custom_list()
        if not params_list:
            print("No parameters entered. Exiting."); sys.exit(0)
    else:
        params_list = build_parameter_list(args.preset)

    generate_dataset(params_list, Path(args.outdir))


if __name__ == "__main__":
    main()
