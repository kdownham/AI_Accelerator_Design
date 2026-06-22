#!/usr/bin/env python3
"""Compare start/end RMS-normalized phase space for no/mean/quantum SR."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "facet_sr_phase_space_rms_compare.png"

FILES = {
    "No SR": ROOT / "facet_realistic_independent_snapshots.npz",
    "Mean SR": ROOT / "facet_realistic_radiation_mean_start_end.npz",
    "Quantum SR": ROOT / "facet_realistic_radiation_quantum_start_end.npz",
}


def rms(values: np.ndarray, alive: np.ndarray) -> float:
    return float(np.std(values[alive]))


def load_start_end(label: str, path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    if label == "No SR":
        return {
            "start": {k: data[k][0] for k in ["x", "px", "y", "py", "zeta", "delta", "state"]},
            "end": {k: data[k][-1] for k in ["x", "px", "y", "py", "zeta", "delta", "state"]},
        }
    return {
        "start": {k: data[f"start_{k}"] for k in ["x", "px", "y", "py", "zeta", "delta", "state"]},
        "end": {k: data[f"end_{k}"] for k in ["x", "px", "y", "py", "zeta", "delta", "state"]},
    }


def normalized_pair(
    snap: dict[str, np.ndarray],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    alive = snap["state"] > 0
    sx = rms(snap[x_key], alive)
    sy = rms(snap[y_key], alive)
    return snap[x_key][alive] / sx, snap[y_key][alive] / sy, sx, sy


def fmt_rms(key: str, value: float) -> str:
    if key in {"x", "y", "zeta"}:
        return f"{value * 1e6:.3g} um"
    if key in {"px", "py"}:
        return f"{value * 1e6:.3g} urad"
    return f"{value:.3g}"


def panel_text(
    start_rms: tuple[float, float],
    end_rms: tuple[float, float],
    keys: tuple[str, str],
) -> str:
    return (
        f"start rms: {keys[0]}={fmt_rms(keys[0], start_rms[0])}, "
        f"{keys[1]}={fmt_rms(keys[1], start_rms[1])}\n"
        f"end rms:   {keys[0]}={fmt_rms(keys[0], end_rms[0])}, "
        f"{keys[1]}={fmt_rms(keys[1], end_rms[1])}"
    )


def main() -> None:
    planes = [
        ("x", "px", "x / rms(x)", "px / rms(px)", "x-px", "#0f766e"),
        ("y", "py", "y / rms(y)", "py / rms(py)", "y-py", "#c2410c"),
        ("zeta", "delta", "zeta / rms(zeta)", "delta / rms(delta)", "zeta-delta", "#6d28d9"),
    ]

    fig, axes = plt.subplots(
        len(FILES),
        len(planes),
        figsize=(17.0, 12.5),
        constrained_layout=True,
    )

    for row, (label, path) in enumerate(FILES.items()):
        snapshots = load_start_end(label, path)
        for col, (x_key, y_key, xlabel, ylabel, title, color) in enumerate(planes):
            ax = axes[row, col]
            start_x, start_y, start_sx, start_sy = normalized_pair(snapshots["start"], x_key, y_key)
            end_x, end_y, end_sx, end_sy = normalized_pair(snapshots["end"], x_key, y_key)

            ax.scatter(start_x, start_y, s=2.2, alpha=0.13, color="#525252", linewidths=0, label="start")
            ax.scatter(end_x, end_y, s=2.2, alpha=0.20, color=color, linewidths=0, label="end")
            ax.axhline(0, color="#404040", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="#404040", linewidth=0.8, alpha=0.5)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label}: {title}")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", markerscale=3)
            ax.text(
                0.02,
                0.02,
                panel_text((start_sx, start_sy), (end_sx, end_sy), (x_key, y_key)),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "#d4d4d4", "alpha": 0.86},
            )

    fig.suptitle(
        "FACET realistic independent beam: RMS-normalized start/end phase space "
        "for synchrotron radiation modes",
        fontsize=14,
    )
    fig.savefig(OUT_PNG, dpi=170)
    plt.close(fig)
    print(f"wrote={OUT_PNG}")


if __name__ == "__main__":
    main()
