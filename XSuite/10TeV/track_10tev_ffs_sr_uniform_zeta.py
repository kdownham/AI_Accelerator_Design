#!/usr/bin/env python3
"""Track the 10 TeV FFS bunch with zeta sampled from a uniform distribution."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import track_10tev_ffs_sr as base


ROOT = Path(__file__).resolve().parent
OUT_PREFIX = ROOT / "ffs_10tev_tracking_uniform_zeta"
ZETA_MIN = -22.0e-6
ZETA_MAX = 22.0e-6


def make_distribution(twiss: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    sigma_x = np.sqrt(twiss["betx"] * base.PHYSEMIT_X)
    sigma_y = np.sqrt(twiss["bety"] * base.PHYSEMIT_Y)
    sigma_px = sigma_x / twiss["betx"]
    sigma_py = sigma_y / twiss["bety"]

    rng = np.random.default_rng(base.RNG_SEED)
    phi = 2 * np.pi * rng.random(base.N_MACROPARTICLES)
    r1 = rng.standard_normal(base.N_MACROPARTICLES)
    r2 = rng.standard_normal(base.N_MACROPARTICLES)
    zeta = rng.uniform(ZETA_MIN, ZETA_MAX, base.N_MACROPARTICLES)
    distribution = {
        "phi": phi,
        "x": sigma_x * r1,
        "y": sigma_y * r1,
        "px": sigma_px * (-twiss["alfx"] * r1 + r2),
        "py": sigma_py * (-twiss["alfy"] * r1 + r2),
        "zeta": zeta,
        "delta": base.SIGMA_DELTA * rng.standard_normal(base.N_MACROPARTICLES),
        "weight": np.full(base.N_MACROPARTICLES, base.BUNCH_INTENSITY / base.N_MACROPARTICLES),
    }
    sigmas = {
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "sigma_px": float(sigma_px),
        "sigma_py": float(sigma_py),
        "sigma_z": float(np.std(zeta)),
        "sigma_delta": base.SIGMA_DELTA,
        "zeta_min": ZETA_MIN,
        "zeta_max": ZETA_MAX,
    }
    return distribution, sigmas


def main() -> None:
    base.OUT_PREFIX = OUT_PREFIX
    _line, _active, resolver = base.build_line()
    twiss = base.initial_twiss(resolver)
    distribution, sigmas = make_distribution(twiss)

    results: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    summary = [
        "active_line=FFS",
        f"particles={base.N_MACROPARTICLES}",
        f"rng_seed={base.RNG_SEED}",
        f"energy_eV={base.ENERGY:.12g}",
        f"gamma={base.GAMMA:.12g}",
        f"bunch_intensity_e={base.BUNCH_INTENSITY:.12g}",
        f"macroparticle_weight={base.BUNCH_INTENSITY / base.N_MACROPARTICLES:.12g}",
        f"physemit_x={base.PHYSEMIT_X:.12g}",
        f"physemit_y={base.PHYSEMIT_Y:.12g}",
        "sampling=shared r1/r2 in x and y as in the current baseline",
        f"zeta_distribution=uniform[{ZETA_MIN:.12g},{ZETA_MAX:.12g}] m",
        f"zeta_input_rms_m={float(np.std(distribution['zeta'])):.12g}",
        f"input_corr_x_y={float(np.corrcoef(distribution['x'], distribution['y'])[0, 1]):.12g}",
    ]
    for key, value in twiss.items():
        summary.append(f"twiss_{key}={value:.12g}")
    for key, value in sigmas.items():
        summary.append(f"{key}={value:.12g}")

    for mode in base.MODES:
        out_npz, start, end = base.track_mode(mode, twiss, distribution, sigmas)
        key = base.mode_label(mode)
        results[key] = (start, end)
        alive = end["state"] > 0
        summary.extend(
            [
                f"{key}_file={out_npz.name}",
                f"{key}_alive={int(alive.sum())}",
                f"{key}_lost={int((~alive).sum())}",
                f"{key}_end_mean_delta={base.mean(end['delta'], alive):.12g}",
                f"{key}_end_rms_x_m={base.rms(end['x'], alive):.12g}",
                f"{key}_end_rms_y_m={base.rms(end['y'], alive):.12g}",
                f"{key}_end_rms_zeta_m={base.rms(end['zeta'], alive):.12g}",
                f"{key}_end_rms_delta={base.rms(end['delta'], alive):.12g}",
            ]
        )
        print(f"{key}: wrote {out_npz} alive={int(alive.sum())}/{len(alive)}")

    out_png = base.plot_comparison(results)
    summary.append(f"plot={out_png.name}")
    summary_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_sr_summary.txt")
    summary_path.write_text("\n".join(summary) + "\n", encoding="ascii")
    print(f"wrote {out_png}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
