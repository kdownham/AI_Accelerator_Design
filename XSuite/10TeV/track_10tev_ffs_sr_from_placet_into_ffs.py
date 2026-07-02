#!/usr/bin/env python3
"""Track a PLACET-derived 10 TeV FFS bunch through the Xsuite FFS line."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import track_10tev_ffs_sr as base
from convert_10tev_ffs_to_xsuite import build_line


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "electron_placet_into_FFS_xsuite_particles.npz"
OUT_PREFIX = ROOT / "ffs_10tev_tracking_placet_into_ffs"


def load_distribution(source: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with np.load(source) as data:
        required = ["x", "px", "y", "py", "zeta", "delta", "weight"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{source} is missing required arrays: {', '.join(missing)}")

        distribution = {
            "phi": np.zeros(len(data["x"]), dtype=float),
            "x": np.asarray(data["x"], dtype=float),
            "px": np.asarray(data["px"], dtype=float),
            "y": np.asarray(data["y"], dtype=float),
            "py": np.asarray(data["py"], dtype=float),
            "zeta": np.asarray(data["zeta"], dtype=float),
            "delta": np.asarray(data["delta"], dtype=float),
            "weight": np.asarray(data["weight"], dtype=float),
        }

    alive = np.ones(len(distribution["x"]), dtype=bool)
    sigmas = {
        "sigma_x": base.rms(distribution["x"], alive),
        "sigma_y": base.rms(distribution["y"], alive),
        "sigma_px": base.rms(distribution["px"], alive),
        "sigma_py": base.rms(distribution["py"], alive),
        "sigma_z": base.rms(distribution["zeta"], alive),
        "sigma_delta": base.rms(distribution["delta"], alive),
    }
    return distribution, sigmas


def main() -> None:
    _line, _active, resolver = build_line()
    twiss = base.initial_twiss(resolver)
    distribution, sigmas = load_distribution(SOURCE)

    base.OUT_PREFIX = OUT_PREFIX
    base.N_MACROPARTICLES = len(distribution["x"])

    results: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    summary = [
        "active_line=FFS",
        f"source_file={SOURCE}",
        f"particles={base.N_MACROPARTICLES}",
        f"rng_seed={base.RNG_SEED}",
        f"energy_eV={base.ENERGY:.12g}",
        f"gamma={base.GAMMA:.12g}",
        f"bunch_intensity_e={base.BUNCH_INTENSITY:.12g}",
        f"macroparticle_weight_mean={float(np.mean(distribution['weight'])):.12g}",
        f"macroparticle_weight_min={float(np.min(distribution['weight'])):.12g}",
        f"macroparticle_weight_max={float(np.max(distribution['weight'])):.12g}",
        "sampling=external PLACET-derived distribution from electron_placet_into_FFS_xsuite_particles.npz",
    ]
    summary.extend(f"initial_{key}={value:.12g}" for key, value in twiss.items())
    summary.extend(f"{key}={value:.12g}" for key, value in sigmas.items())

    for mode in base.MODES:
        out_npz, start, end = base.track_mode(mode, twiss, distribution, sigmas)
        label = base.mode_label(mode)
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
                f"end_mean_x={base.mean(end['x'], alive_end):.12g}",
                f"end_mean_y={base.mean(end['y'], alive_end):.12g}",
                f"end_mean_zeta={base.mean(end['zeta'], alive_end):.12g}",
                f"end_mean_delta={base.mean(end['delta'], alive_end):.12g}",
                f"end_rms_x={base.rms(end['x'], alive_end):.12g}",
                f"end_rms_y={base.rms(end['y'], alive_end):.12g}",
                f"end_rms_zeta={base.rms(end['zeta'], alive_end):.12g}",
                f"end_rms_delta={base.rms(end['delta'], alive_end):.12g}",
            ]
        )

    plot_png = base.plot_comparison(results)
    summary.append("")
    summary.append(f"plot_png={plot_png}")
    summary_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_sr_summary.txt")
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
