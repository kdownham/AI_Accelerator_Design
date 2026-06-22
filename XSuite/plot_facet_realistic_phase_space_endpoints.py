#!/usr/bin/env python3
"""Plot initial and final phase-space occupancy from saved FACET snapshots."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SNAPSHOTS = ROOT / "facet_realistic_independent_snapshots.npz"
OUT_PNG = ROOT / "facet_realistic_independent_phase_space_endpoints.png"


def limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    lo = float(vals.min())
    hi = float(vals.max())
    span = hi - lo
    if span == 0:
        span = 1.0
    pad = 0.06 * span
    return lo - pad, hi + pad


def main() -> None:
    data = np.load(SNAPSHOTS, allow_pickle=True)
    state0 = data["state"][0] > 0
    statef = data["state"][-1] > 0
    names = data["element_names"]
    s = data["s"]

    x0 = data["x"][0, state0] * 1e6
    px0 = data["px"][0, state0] * 1e6
    y0 = data["y"][0, state0] * 1e6
    py0 = data["py"][0, state0] * 1e6
    z0 = data["zeta"][0, state0] * 1e6
    d0 = data["delta"][0, state0]
    xf = data["x"][-1, statef] * 1e6
    pxf = data["px"][-1, statef] * 1e6
    yf = data["y"][-1, statef] * 1e6
    pyf = data["py"][-1, statef] * 1e6
    zf = data["zeta"][-1, statef] * 1e6
    df = data["delta"][-1, statef]

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.5), constrained_layout=True)
    panels = [
        (axes[0, 0], x0, px0, "Initial x-px", limits(x0, xf), limits(px0, pxf), "x [um]", "px [urad]", "#0f766e"),
        (axes[0, 1], xf, pxf, f"Final x-px ({names[-1]})", limits(x0, xf), limits(px0, pxf), "x [um]", "px [urad]", "#0f766e"),
        (axes[1, 0], y0, py0, "Initial y-py", limits(y0, yf), limits(py0, pyf), "y [um]", "py [urad]", "#c2410c"),
        (axes[1, 1], yf, pyf, f"Final y-py ({names[-1]})", limits(y0, yf), limits(py0, pyf), "y [um]", "py [urad]", "#c2410c"),
        (axes[2, 0], z0, d0, "Initial zeta-delta", limits(z0, zf), limits(d0, df), "zeta [um]", "delta", "#6d28d9"),
        (axes[2, 1], zf, df, f"Final zeta-delta ({names[-1]})", limits(z0, zf), limits(d0, df), "zeta [um]", "delta", "#6d28d9"),
    ]

    for ax, pos, mom, title, xlim, ylim, xlabel, ylabel, color in panels:
        ax.scatter(pos, mom, s=3.0, alpha=0.22, color=color, linewidths=0)
        ax.axhline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "FACET realistic phase-space occupancy "
        f"| start to {names[-1]} at s={float(s[-1]):.3f} m "
        f"| alive={int(statef.sum())}/{len(statef)}",
        fontsize=13,
    )
    fig.savefig(OUT_PNG, dpi=170)
    plt.close(fig)
    print(f"wrote={OUT_PNG}")


if __name__ == "__main__":
    main()
