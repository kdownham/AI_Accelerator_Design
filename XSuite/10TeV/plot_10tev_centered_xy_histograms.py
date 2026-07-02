#!/usr/bin/env python3
"""Plot endpoint x/y histograms for x-y centered 10 TeV FFS tracking outputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CASES = [
    (
        "No SR",
        ROOT / "ffs_10tev_tracking_no_sr_start_end_xy_centered.npz",
        ROOT / "ffs_10tev_tracking_no_sr_xy_centered_histograms.png",
    ),
    (
        "Mean SR",
        ROOT / "ffs_10tev_tracking_mean_sr_start_end_xy_centered.npz",
        ROOT / "ffs_10tev_tracking_mean_sr_xy_centered_histograms.png",
    ),
    (
        "Quantum SR",
        ROOT / "ffs_10tev_tracking_quantum_sr_start_end_xy_centered.npz",
        ROOT / "ffs_10tev_tracking_quantum_sr_xy_centered_histograms.png",
    ),
]


def rms(values: np.ndarray) -> float:
    return float(np.std(values))


def plot_case(label: str, input_npz: Path, out_png: Path) -> None:
    with np.load(input_npz) as data:
        alive = data["output_state"] > 0
        x_nm = np.asarray(data["output_x"], dtype=float)[alive] * 1e9
        y_nm = np.asarray(data["output_y"], dtype=float)[alive] * 1e9
        mode = str(data["mode"].item())
        offset_x_nm = float(data["xy_center_offset_x_m"]) * 1e9
        offset_y_nm = float(data["xy_center_offset_y_m"]) * 1e9

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    histograms = [
        (axes[0], x_nm, "x [nm]", "#0f766e", (-0.25e-6 * 1e9, 0.25e-6 * 1e9), offset_x_nm),
        (axes[1], y_nm, "y [nm]", "#c2410c", (-0.25e-7 * 1e9, 0.25e-7 * 1e9), offset_y_nm),
    ]

    for ax, values, xlabel, color, xlim, offset_nm in histograms:
        ax.hist(values, bins=400, color=color, alpha=0.82, edgecolor="white", linewidth=0.2)
        ax.axvline(np.mean(values), color="#202020", linewidth=1.0, label="mean")
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of particles per bin")
        ax.grid(True, alpha=0.25)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"mode: {mode}",
                    f"live particles: {values.size}",
                    f"removed mean: {offset_nm:.6g} nm",
                    f"mean: {np.mean(values):.6g} nm",
                    f"rms: {rms(values):.6g} nm",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            bbox={"facecolor": "white", "edgecolor": "#d4d4d4", "alpha": 0.9},
        )

    axes[0].set_title("Centered endpoint x distribution")
    axes[1].set_title("Centered endpoint y distribution")
    fig.suptitle(f"10 TeV FFS endpoint coordinate histograms, centered {label}", fontsize=13)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"wrote={out_png}")


def main() -> None:
    for label, input_npz, out_png in CASES:
        plot_case(label, input_npz, out_png)


if __name__ == "__main__":
    main()
