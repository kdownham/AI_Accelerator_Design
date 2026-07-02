#!/usr/bin/env python3
"""Track the 10 TeV FFS bunch with no, mean, and quantum SR."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

from convert_10tev_ffs_to_xsuite import build_line


ROOT = Path(__file__).resolve().parent
OUT_PREFIX = ROOT / "ffs_10tev_tracking"
N_MACROPARTICLES = int(1e4)
RNG_SEED = 20260624
ENERGY = 5e12
GAMMA = ENERGY / xt.ELECTRON_MASS_EV
BUNCH_INTENSITY = 3.72e-9 / 1.6e-19
PHYSEMIT_X = 660e-9 / GAMMA
PHYSEMIT_Y = 20e-9 / GAMMA
SIGMA_Z = 44e-6
SIGMA_DELTA = 3e-3
MODES: tuple[str | None, ...] = (None, "mean", "quantum")


def as_array(values) -> np.ndarray:
    return np.asarray(values)


def initial_twiss(resolver) -> dict[str, float]:
    return {
        "betx": resolver.value("BETX"),
        "bety": resolver.value("BETY"),
        "alfx": resolver.value("ALFX"),
        "alfy": resolver.value("ALFY"),
        "dx": resolver.value("DX"),
        "dpx": 0.0,
        "dy": 0.0,
        "dpy": 0.0,
    }


def make_distribution(twiss: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    sigma_x = np.sqrt(twiss["betx"] * PHYSEMIT_X)
    sigma_y = np.sqrt(twiss["bety"] * PHYSEMIT_Y)
    sigma_px = sigma_x / twiss["betx"]
    sigma_py = sigma_y / twiss["bety"]

    rng = np.random.default_rng(RNG_SEED)
    phi = 2 * np.pi * rng.random(N_MACROPARTICLES)
    r1 = rng.standard_normal(N_MACROPARTICLES)
    r2 = rng.standard_normal(N_MACROPARTICLES)
    distribution = {
        "phi": phi,
        "x": sigma_x * r1,
        "y": sigma_y * r1,
        "px": sigma_px * (-twiss["alfx"] * r1 + r2),
        "py": sigma_py * (-twiss["alfy"] * r1 + r2),
        "zeta": SIGMA_Z * rng.standard_normal(N_MACROPARTICLES),
        "delta": SIGMA_DELTA * rng.standard_normal(N_MACROPARTICLES),
        "weight": np.full(N_MACROPARTICLES, BUNCH_INTENSITY / N_MACROPARTICLES),
    }
    sigmas = {
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "sigma_px": float(sigma_px),
        "sigma_py": float(sigma_py),
        "sigma_z": SIGMA_Z,
        "sigma_delta": SIGMA_DELTA,
    }
    return distribution, sigmas


def make_particles(context: xo.ContextCpu, distribution: dict[str, np.ndarray]) -> xp.Particles:
    return xp.Particles(
        _context=context,
        q0=-1,
        p0c=ENERGY,
        mass0=xt.ELECTRON_MASS_EV,
        x=distribution["x"],
        y=distribution["y"],
        px=distribution["px"],
        py=distribution["py"],
        zeta=distribution["zeta"],
        delta=distribution["delta"],
        weight=distribution["weight"],
    )


def snapshot(particles: xp.Particles) -> dict[str, np.ndarray]:
    return {
        key: as_array(getattr(particles, key)).copy()
        for key in [
            "x",
            "px",
            "y",
            "py",
            "zeta",
            "delta",
            "state",
            "weight",
            "particle_id",
            "at_element",
            "at_turn",
        ]
    }


def rms(values: np.ndarray, alive: np.ndarray) -> float:
    return float(np.std(values[alive]))


def mean(values: np.ndarray, alive: np.ndarray) -> float:
    return float(np.mean(values[alive]))


def mode_label(mode: str | None) -> str:
    return "no_sr" if mode is None else f"{mode}_sr"


def track_mode(
    mode: str | None,
    twiss: dict[str, float],
    distribution: dict[str, np.ndarray],
    sigmas: dict[str, float],
) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    context = xo.ContextCpu()
    line, _active, _resolver = build_line()
    if mode is not None:
        line.configure_radiation(model=mode)
    line.build_tracker(_context=context)

    particles = make_particles(context, distribution)
    if mode == "quantum":
        particles._init_random_number_generator(seeds=np.arange(N_MACROPARTICLES) + RNG_SEED)

    start = snapshot(particles)
    line.track(particles)
    end = snapshot(particles)

    label = mode_label(mode)
    out_npz = OUT_PREFIX.with_name(OUT_PREFIX.name + f"_{label}_start_end.npz")
    np.savez_compressed(
        out_npz,
        mode=np.array("none" if mode is None else mode),
        rng_seed=np.array(RNG_SEED),
        energy_eV=np.array(ENERGY),
        p0c_eV=np.array(ENERGY),
        mass0_eV=np.array(xt.ELECTRON_MASS_EV),
        q0=np.array(-1),
        gamma=np.array(GAMMA),
        bunch_intensity=np.array(BUNCH_INTENSITY),
        macroparticle_weight=np.array(BUNCH_INTENSITY / N_MACROPARTICLES),
        physemit_x=np.array(PHYSEMIT_X),
        physemit_y=np.array(PHYSEMIT_Y),
        **{f"twiss_{key}": np.array(value) for key, value in twiss.items()},
        **{key: np.array(value) for key, value in sigmas.items()},
        **{f"input_{key}": value for key, value in distribution.items()},
        **{f"output_{key}": value for key, value in end.items()},
        **{f"start_{key}": value for key, value in start.items()},
    )
    return out_npz, start, end


def limits(values: list[np.ndarray]) -> tuple[float, float]:
    merged = np.concatenate([value[np.isfinite(value)] for value in values])
    low = float(merged.min())
    high = float(merged.max())
    span = high - low
    if span == 0:
        span = 1.0
    pad = 0.06 * span
    return low - pad, high + pad


def plot_comparison(results: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]) -> Path:
    out_png = OUT_PREFIX.with_name(OUT_PREFIX.name + "_sr_phase_space.png")
    plane_specs = [
        ("x", "px", "x [um]", "px [urad]", 1e6, 1e6, "x-px", "#0f766e"),
        ("y", "py", "y [um]", "py [urad]", 1e6, 1e6, "y-py", "#c2410c"),
        ("zeta", "delta", "zeta [um]", "delta", 1e6, 1.0, "zeta-delta", "#6d28d9"),
    ]
    labels = [("no_sr", "No SR"), ("mean_sr", "Mean SR"), ("quantum_sr", "Quantum SR")]
    fig, axes = plt.subplots(3, 3, figsize=(17, 12.2), constrained_layout=True)

    for col, (coord, momentum, xlabel, ylabel, scale_x, scale_y, plane_name, color) in enumerate(plane_specs):
        all_values_x: list[np.ndarray] = []
        all_values_y: list[np.ndarray] = []
        for start, end in results.values():
            all_values_x.extend([start[coord] * scale_x, end[coord] * scale_x])
            all_values_y.extend([start[momentum] * scale_y, end[momentum] * scale_y])
        xlim = limits(all_values_x)
        ylim = limits(all_values_y)

        for row, (key, label) in enumerate(labels):
            ax = axes[row, col]
            start, end = results[key]
            alive_start = start["state"] > 0
            alive_end = end["state"] > 0
            ax.scatter(
                start[coord][alive_start] * scale_x,
                start[momentum][alive_start] * scale_y,
                s=2.1,
                alpha=0.12,
                color="#525252",
                linewidths=0,
                label="input",
            )
            ax.scatter(
                end[coord][alive_end] * scale_x,
                end[momentum][alive_end] * scale_y,
                s=2.1,
                alpha=0.20,
                color=color,
                linewidths=0,
                label="output",
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.axhline(0, color="#404040", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="#404040", linewidth=0.8, alpha=0.5)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"{label}: {plane_name}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", markerscale=3)

            text = (
                f"input rms: {coord}={rms(start[coord], alive_start) * scale_x:.6g}, "
                f"{momentum}={rms(start[momentum], alive_start) * scale_y:.6g}\\n"
                f"output rms: {coord}={rms(end[coord], alive_end) * scale_x:.6g}, "
                f"{momentum}={rms(end[momentum], alive_end) * scale_y:.6g}\\n"
                f"alive={int(alive_end.sum())}/{len(alive_end)}"
            )
            ax.text(
                0.02,
                0.02,
                text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.6,
                bbox={"facecolor": "white", "edgecolor": "#d4d4d4", "alpha": 0.88},
            )

    fig.suptitle(
        "10 TeV Flat-Beam FFS: input/output phase space for synchrotron-radiation modes",
        fontsize=14,
    )
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return out_png


def main() -> None:
    _line, _active, resolver = build_line()
    twiss = initial_twiss(resolver)
    distribution, sigmas = make_distribution(twiss)

    results: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    summary = [
        "active_line=FFS",
        f"particles={N_MACROPARTICLES}",
        f"rng_seed={RNG_SEED}",
        f"energy_eV={ENERGY:.12g}",
        f"gamma={GAMMA:.12g}",
        f"bunch_intensity_e={BUNCH_INTENSITY:.12g}",
        f"macroparticle_weight={BUNCH_INTENSITY / N_MACROPARTICLES:.12g}",
        f"physemit_x={PHYSEMIT_X:.12g}",
        f"physemit_y={PHYSEMIT_Y:.12g}",
        "sampling=shared r1/r2 in x and y as requested",
    ]
    summary.extend(f"initial_{key}={value:.12g}" for key, value in twiss.items())
    summary.extend(f"{key}={value:.12g}" for key, value in sigmas.items())

    for mode in MODES:
        out_npz, start, end = track_mode(mode, twiss, distribution, sigmas)
        label = mode_label(mode)
        results[label] = (start, end)
        alive_start = start["state"] > 0
        alive_end = end["state"] > 0
        summary.extend(
            [
                "",
                f"mode={'none' if mode is None else mode}",
                f"file={out_npz}",
                f"start_alive={int(alive_start.sum())}",
                f"end_alive={int(alive_end.sum())}",
                f"end_lost={int((~alive_end).sum())}",
                f"end_mean_delta={mean(end['delta'], alive_end):.12g}",
                f"end_rms_x={rms(end['x'], alive_end):.12g}",
                f"end_rms_y={rms(end['y'], alive_end):.12g}",
                f"end_rms_zeta={rms(end['zeta'], alive_end):.12g}",
                f"end_rms_delta={rms(end['delta'], alive_end):.12g}",
            ]
        )

    plot_png = plot_comparison(results)
    summary.append("")
    summary.append(f"plot_png={plot_png}")
    summary_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_sr_summary.txt")
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
