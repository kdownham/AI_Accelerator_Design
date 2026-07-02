#!/usr/bin/env python3
"""Render RMS-normalized 10 TeV FFS SR phase-space comparisons."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "ffs_10tev_tracking_sr_phase_space_rms.png"
INPUT_OUT_PNG = ROOT / "ffs_10tev_tracking_input_phase_space_rms.png"
FILES = {
    "No SR": ROOT / "ffs_10tev_tracking_no_sr_start_end.npz",
    "Mean SR": ROOT / "ffs_10tev_tracking_mean_sr_start_end.npz",
    "Quantum SR": ROOT / "ffs_10tev_tracking_quantum_sr_start_end.npz",
}


def rms(values: np.ndarray) -> float:
    return float(np.std(values))


def load_distribution(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    start = {key: data[f"input_{key}"] for key in ["x", "px", "y", "py", "zeta", "delta"]}
    alive = data["output_state"] > 0
    end = {key: data[f"output_{key}"][alive] for key in ["x", "px", "y", "py", "zeta", "delta"]}
    return start, end


def physical_rms_text(key: str, value: float) -> str:
    if key in {"x", "y", "zeta"}:
        return f"{value * 1e6:.6g} um"
    if key in {"px", "py"}:
        return f"{value * 1e6:.6g} urad"
    return f"{value:.6g}"


PLANES = [
    ("x", "px", "x / rms(x)", "px / rms(px)", "x-px", "#0f766e"),
    ("y", "py", "y / rms(y)", "py / rms(py)", "y-py", "#c2410c"),
    ("zeta", "delta", "zeta / rms(zeta)", "delta / rms(delta)", "zeta-delta", "#6d28d9"),
]


def annotate_rms(ax, coord: str, momentum: str, coord_rms: float, momentum_rms: float) -> None:
    ax.text(
        0.02,
        0.02,
        f"rms: {coord}={physical_rms_text(coord, coord_rms)}, "
        f"{momentum}={physical_rms_text(momentum, momentum_rms)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.7,
        bbox={"facecolor": "white", "edgecolor": "#d4d4d4", "alpha": 0.88},
    )


def plot_outputs(distributions: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(17, 12.2), constrained_layout=True)

    for row, (label, (start, end)) in enumerate(distributions.items()):
        for col, (coord, momentum, xlabel, ylabel, plane, color) in enumerate(PLANES):
            ax = axes[row, col]
            end_coord_rms = rms(end[coord])
            end_mom_rms = rms(end[momentum])

            ax.scatter(
                end[coord] / end_coord_rms,
                end[momentum] / end_mom_rms,
                s=2.0,
                alpha=0.20,
                color=color,
                linewidths=0,
            )
            ax.axhline(0, color="#404040", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="#404040", linewidth=0.8, alpha=0.5)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label}: {plane}")
            ax.grid(True, alpha=0.25)
            annotate_rms(ax, coord, momentum, end_coord_rms, end_mom_rms)

    fig.suptitle(
        "10 TeV Flat-Beam FFS: RMS-normalized output phase space by SR mode",
        fontsize=14,
    )
    fig.savefig(OUT_PNG, dpi=170)
    plt.close(fig)


def plot_input(start: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), constrained_layout=True)
    for ax, (coord, momentum, xlabel, ylabel, plane, color) in zip(axes, PLANES):
        coord_rms = rms(start[coord])
        momentum_rms = rms(start[momentum])
        ax.scatter(
            start[coord] / coord_rms,
            start[momentum] / momentum_rms,
            s=2.0,
            alpha=0.20,
            color=color,
            linewidths=0,
        )
        ax.axhline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Input: {plane}")
        ax.grid(True, alpha=0.25)
        annotate_rms(ax, coord, momentum, coord_rms, momentum_rms)

    fig.suptitle("10 TeV Flat-Beam FFS: RMS-normalized input phase space", fontsize=14)
    fig.savefig(INPUT_OUT_PNG, dpi=170)
    plt.close(fig)


def main() -> None:
    distributions = {label: load_distribution(path) for label, path in FILES.items()}
    plot_outputs(distributions)
    plot_input(next(iter(distributions.values()))[0])
    print(f"wrote={OUT_PNG}")
    print(f"wrote={INPUT_OUT_PNG}")


if __name__ == "__main__":
    main()
