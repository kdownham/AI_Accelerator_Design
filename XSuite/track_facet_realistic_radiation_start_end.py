#!/usr/bin/env python3
"""Track realistic FACET particles with mean and quantum synchrotron radiation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo

from run_facet_madx_in_xsuite import build_line
from track_facet_realistic_independent_gif import (
    ALPHA_X,
    ALPHA_Y,
    BETA_X,
    BETA_Y,
    BUNCH_INTENSITY,
    ENERGY,
    GAMMA,
    N_MACROPARTICLES_B1,
    PHYSEMIT_X,
    PHYSEMIT_Y,
    RNG_SEED,
    SIGMA_DELTA,
    SIGMA_PX,
    SIGMA_PY,
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    make_realistic_particles,
)


ROOT = Path(__file__).resolve().parent
OUT_PREFIX = ROOT / "facet_realistic_radiation"
RADIATION_MODES = ("mean", "quantum")


def as_array(values) -> np.ndarray:
    return np.asarray(values)


def snapshot(particles) -> dict[str, np.ndarray]:
    return {
        "x": as_array(particles.x).copy(),
        "px": as_array(particles.px).copy(),
        "y": as_array(particles.y).copy(),
        "py": as_array(particles.py).copy(),
        "zeta": as_array(particles.zeta).copy(),
        "delta": as_array(particles.delta).copy(),
        "state": as_array(particles.state).copy(),
        "weight": as_array(particles.weight).copy(),
    }


def save_mode(mode: str) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    context = xo.ContextCpu()
    line, kinds = build_line()
    line.configure_radiation(model=mode)
    line.build_tracker(_context=context)

    particles, initial_distribution = make_realistic_particles(context)
    if mode == "quantum" and hasattr(particles, "_init_random_number_generator"):
        particles._init_random_number_generator(seeds=np.arange(N_MACROPARTICLES_B1) + RNG_SEED)

    start = snapshot(particles)
    line.track(particles)
    end = snapshot(particles)

    out_npz = OUT_PREFIX.with_name(OUT_PREFIX.name + f"_{mode}_start_end.npz")
    np.savez_compressed(
        out_npz,
        mode=np.array(mode),
        rng_seed=np.array(RNG_SEED),
        energy_ev=np.array(ENERGY),
        gamma=np.array(GAMMA),
        bunch_intensity=np.array(BUNCH_INTENSITY),
        beta_x=np.array(BETA_X),
        beta_y=np.array(BETA_Y),
        alpha_x=np.array(ALPHA_X),
        alpha_y=np.array(ALPHA_Y),
        physemit_x=np.array(PHYSEMIT_X),
        physemit_y=np.array(PHYSEMIT_Y),
        sigma_x=np.array(SIGMA_X),
        sigma_y=np.array(SIGMA_Y),
        sigma_px=np.array(SIGMA_PX),
        sigma_py=np.array(SIGMA_PY),
        sigma_z=np.array(SIGMA_Z),
        sigma_delta=np.array(SIGMA_DELTA),
        line_elements=np.array(len(line.elements)),
        element_types=np.array([f"{k}:{v}" for k, v in sorted(kinds.items())], dtype=object),
        initial_distribution_x=initial_distribution["x"],
        initial_distribution_px=initial_distribution["px"],
        initial_distribution_y=initial_distribution["y"],
        initial_distribution_py=initial_distribution["py"],
        initial_distribution_zeta=initial_distribution["zeta"],
        initial_distribution_delta=initial_distribution["delta"],
        **{f"start_{k}": v for k, v in start.items()},
        **{f"end_{k}": v for k, v in end.items()},
    )
    return out_npz, start, end


def rms(values: np.ndarray, alive: np.ndarray) -> float:
    return float(np.std(values[alive]))


def mean(values: np.ndarray, alive: np.ndarray) -> float:
    return float(np.mean(values[alive]))


def plot_endpoint_comparison(results: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]) -> Path:
    out_png = OUT_PREFIX.with_name(OUT_PREFIX.name + "_start_end_phase_space.png")
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(17.0, 12.0),
        constrained_layout=True,
    )

    panels = []
    for col, mode in enumerate(RADIATION_MODES):
        start, end = results[mode]
        for offset, label, snap in [(0, "start", start), (1, "end", end)]:
            panel_col = 2 * col + offset
            alive = snap["state"] > 0
            panels.extend(
                [
                    (axes[0, panel_col], snap["x"][alive] * 1e6, snap["px"][alive] * 1e6, f"{mode} {label} x-px", "x [um]", "px [urad]", "#0f766e"),
                    (axes[1, panel_col], snap["y"][alive] * 1e6, snap["py"][alive] * 1e6, f"{mode} {label} y-py", "y [um]", "py [urad]", "#c2410c"),
                    (axes[2, panel_col], snap["zeta"][alive] * 1e6, snap["delta"][alive], f"{mode} {label} zeta-delta", "zeta [um]", "delta", "#6d28d9"),
                ]
            )

    for ax, xpos, ypos, title, xlabel, ylabel, color in panels:
        ax.scatter(xpos, ypos, s=2.5, alpha=0.18, color=color, linewidths=0)
        ax.axhline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle("FACET realistic independent beam: start/end tracking with synchrotron radiation")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def main() -> None:
    created = []
    results = {}
    summary_lines = [
        f"particles={N_MACROPARTICLES_B1}",
        f"rng_seed={RNG_SEED}",
        f"energy_ev={ENERGY:.12g}",
        f"gamma={GAMMA:.12g}",
        f"bunch_intensity_e={BUNCH_INTENSITY:.12g}",
        f"sigma_x={SIGMA_X:.12g}",
        f"sigma_y={SIGMA_Y:.12g}",
        f"sigma_px={SIGMA_PX:.12g}",
        f"sigma_py={SIGMA_PY:.12g}",
        f"sigma_z={SIGMA_Z:.12g}",
        f"sigma_delta={SIGMA_DELTA:.12g}",
        "sampling=independent r1/r2 pairs for x-px and y-py",
    ]

    for mode in RADIATION_MODES:
        out_npz, start, end = save_mode(mode)
        created.append(out_npz)
        results[mode] = (start, end)
        alive_start = start["state"] > 0
        alive_end = end["state"] > 0
        summary_lines.extend(
            [
                "",
                f"mode={mode}",
                f"output_npz={out_npz}",
                f"start_alive={int(alive_start.sum())}",
                f"end_alive={int(alive_end.sum())}",
                f"end_lost={int((~alive_end).sum())}",
                f"start_mean_delta={mean(start['delta'], alive_start):.12g}",
                f"end_mean_delta={mean(end['delta'], alive_end):.12g}",
                f"start_rms_delta={rms(start['delta'], alive_start):.12g}",
                f"end_rms_delta={rms(end['delta'], alive_end):.12g}",
                f"start_rms_x={rms(start['x'], alive_start):.12g}",
                f"end_rms_x={rms(end['x'], alive_end):.12g}",
                f"start_rms_y={rms(start['y'], alive_start):.12g}",
                f"end_rms_y={rms(end['y'], alive_end):.12g}",
                f"start_rms_zeta={rms(start['zeta'], alive_start):.12g}",
                f"end_rms_zeta={rms(end['zeta'], alive_end):.12g}",
            ]
        )

    plot_png = plot_endpoint_comparison(results)
    created.append(plot_png)
    summary_lines.append("")
    summary_lines.append(f"plot_png={plot_png}")

    summary_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_start_end_summary.txt")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    created.append(summary_path)

    print("\n".join(summary_lines))
    print("")
    print("created:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
