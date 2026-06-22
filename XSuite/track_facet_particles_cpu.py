#!/usr/bin/env python3
"""Track a Gaussian particle distribution through the FACET Xsuite line on CPU."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp

from run_facet_madx_in_xsuite import build_line


ROOT = Path(__file__).resolve().parent
N_PARTICLES = 10_000
RNG_SEED = 20260617

SIGMAS = {
    "x": 1.0e-6,
    "y": 1.0e-6,
    "px": 1.0e-6,
    "py": 1.0e-6,
    "zeta": 10.0e-6,
    "delta": 0.01,
}


def make_particles(context: xo.ContextCpu) -> tuple[xp.Particles, dict[str, np.ndarray]]:
    rng = np.random.default_rng(RNG_SEED)
    initial = {
        coord: rng.normal(loc=0.0, scale=sigma, size=N_PARTICLES)
        for coord, sigma in SIGMAS.items()
    }
    particles = xp.Particles(
        _context=context,
        p0c=10e9,
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        x=initial["x"],
        px=initial["px"],
        y=initial["y"],
        py=initial["py"],
        zeta=initial["zeta"],
        delta=initial["delta"],
    )
    return particles, initial


def as_cpu_array(values) -> np.ndarray:
    return np.asarray(values)


def plot_phase_space(
    initial: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    alive: np.ndarray,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    panels = [
        (axes[0, 0], initial["x"], initial["px"], "Initial x-px"),
        (axes[0, 1], final["x"][alive], final["px"][alive], "Final x-px"),
        (axes[1, 0], initial["y"], initial["py"], "Initial y-py"),
        (axes[1, 1], final["y"][alive], final["py"][alive], "Final y-py"),
    ]

    for ax, pos, mom, title in panels:
        ax.scatter(pos * 1e6, mom * 1e6, s=3.0, alpha=0.22, color="#0f766e", linewidths=0)
        ax.axhline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[0, 0].set_ylabel("px [urad]")
    axes[0, 1].set_ylabel("px [urad]")
    axes[1, 0].set_ylabel("py [urad]")
    axes[1, 1].set_ylabel("py [urad]")
    axes[0, 0].set_xlabel("x [um]")
    axes[0, 1].set_xlabel("x [um]")
    axes[1, 0].set_xlabel("y [um]")
    axes[1, 1].set_xlabel("y [um]")

    fig.suptitle(
        "FACET Xsuite CPU Tracking: 10,000 Gaussian Particles "
        f"({int(alive.sum())}/{len(alive)} alive at end)",
        fontsize=13,
    )
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    context = xo.ContextCpu()
    line, kinds = build_line()
    line.build_tracker(_context=context)

    particles, initial = make_particles(context)
    line.track(particles)

    final = {
        coord: as_cpu_array(getattr(particles, coord)).copy()
        for coord in ["x", "px", "y", "py", "zeta", "delta"]
    }
    state = as_cpu_array(particles.state).copy()
    alive = state > 0

    out_npz = ROOT / "facet_particles_cpu_tracking.npz"
    out_png = ROOT / "facet_particles_cpu_phase_space.png"
    summary_path = ROOT / "facet_particles_cpu_summary.txt"

    np.savez(
        out_npz,
        state=state,
        **{f"initial_{k}": v for k, v in initial.items()},
        **{f"final_{k}": v for k, v in final.items()},
    )
    plot_phase_space(initial, final, alive, out_png)

    summary = [
        f"context={context.__class__.__name__}",
        f"particles={N_PARTICLES}",
        f"rng_seed={RNG_SEED}",
        "sigmas=" + ", ".join(f"{k}:{v:g}" for k, v in SIGMAS.items()),
        f"alive={int(alive.sum())}",
        f"lost={int((~alive).sum())}",
        f"line_elements={len(line.elements)}",
        "element_types=" + ", ".join(f"{k}:{v}" for k, v in sorted(kinds.items())),
        f"output_npz={out_npz}",
        f"output_png={out_png}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
