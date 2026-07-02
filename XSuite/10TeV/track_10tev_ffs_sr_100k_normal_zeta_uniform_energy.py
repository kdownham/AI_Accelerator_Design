#!/usr/bin/env python3
"""Track 100k 10 TeV FFS particles with normal zeta and uniform energy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import track_10tev_ffs_sr as base


ROOT = Path(__file__).resolve().parent
OUT_PREFIX = ROOT / "ffs_10tev_tracking_100k_normal_zeta_uniform_energy"
N_MACROPARTICLES = int(1e5)
ZETA_RMS = 44.0e-6
ENERGY_MIN_GEV = 4975.0
ENERGY_MAX_GEV = 5025.0
ENERGY_MEAN_GEV = 5000.0
DELTA_MAX = (ENERGY_MAX_GEV - ENERGY_MEAN_GEV) / ENERGY_MEAN_GEV


def standardized_normal(rng: np.random.Generator, n_particles: int, rms: float) -> np.ndarray:
    values = rng.standard_normal(n_particles)
    values = values - np.mean(values)
    values = values / np.std(values)
    return values * rms


def symmetric_uniform_delta(rng: np.random.Generator, n_particles: int) -> np.ndarray:
    if n_particles % 2 != 0:
        raise ValueError("Antithetic uniform energy sampling requires an even particle count")
    half = n_particles // 2
    positive = rng.uniform(0.0, DELTA_MAX, half)
    delta = np.concatenate([positive, -positive])
    rng.shuffle(delta)
    return delta


def make_distribution(twiss: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    sigma_x = np.sqrt(twiss["betx"] * base.PHYSEMIT_X)
    sigma_y = np.sqrt(twiss["bety"] * base.PHYSEMIT_Y)
    sigma_px = sigma_x / twiss["betx"]
    sigma_py = sigma_y / twiss["bety"]

    rng = np.random.default_rng(base.RNG_SEED)
    phi = 2 * np.pi * rng.random(N_MACROPARTICLES)
    r1 = rng.standard_normal(N_MACROPARTICLES)
    r2 = rng.standard_normal(N_MACROPARTICLES)
    zeta = standardized_normal(rng, N_MACROPARTICLES, ZETA_RMS)
    delta = symmetric_uniform_delta(rng, N_MACROPARTICLES)
    distribution = {
        "phi": phi,
        "x": sigma_x * r1,
        "y": sigma_y * r1,
        "px": sigma_px * (-twiss["alfx"] * r1 + r2),
        "py": sigma_py * (-twiss["alfy"] * r1 + r2),
        "zeta": zeta,
        "delta": delta,
        "weight": np.full(N_MACROPARTICLES, base.BUNCH_INTENSITY / N_MACROPARTICLES),
    }
    sigmas = {
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "sigma_px": float(sigma_px),
        "sigma_py": float(sigma_py),
        "sigma_z": float(np.std(zeta)),
        "sigma_delta": float(np.std(delta)),
        "zeta_rms_target_m": ZETA_RMS,
        "energy_min_GeV": ENERGY_MIN_GEV,
        "energy_max_GeV": ENERGY_MAX_GEV,
        "energy_mean_GeV": ENERGY_MEAN_GEV,
        "delta_min": -DELTA_MAX,
        "delta_max": DELTA_MAX,
    }
    return distribution, sigmas


def main() -> None:
    base.OUT_PREFIX = OUT_PREFIX
    base.N_MACROPARTICLES = N_MACROPARTICLES
    _line, _active, resolver = base.build_line()
    twiss = base.initial_twiss(resolver)
    distribution, sigmas = make_distribution(twiss)

    input_energy_GeV = ENERGY_MEAN_GEV * (1.0 + distribution["delta"])
    summary = [
        "active_line=FFS",
        f"particles={N_MACROPARTICLES}",
        f"rng_seed={base.RNG_SEED}",
        f"reference_energy_eV={base.ENERGY:.12g}",
        f"gamma={base.GAMMA:.12g}",
        f"bunch_intensity_e={base.BUNCH_INTENSITY:.12g}",
        f"macroparticle_weight={base.BUNCH_INTENSITY / N_MACROPARTICLES:.12g}",
        f"physemit_x={base.PHYSEMIT_X:.12g}",
        f"physemit_y={base.PHYSEMIT_Y:.12g}",
        "sampling=shared r1/r2 in x and y as in the current baseline",
        "zeta_distribution=standardized_normal",
        f"zeta_input_mean_m={float(np.mean(distribution['zeta'])):.12g}",
        f"zeta_input_rms_m={float(np.std(distribution['zeta'])):.12g}",
        f"input_energy_min_GeV={float(np.min(input_energy_GeV)):.12g}",
        f"input_energy_max_GeV={float(np.max(input_energy_GeV)):.12g}",
        f"input_energy_mean_GeV={float(np.mean(input_energy_GeV)):.12g}",
        f"input_energy_rms_GeV={float(np.std(input_energy_GeV)):.12g}",
        f"input_corr_x_y={float(np.corrcoef(distribution['x'], distribution['y'])[0, 1]):.12g}",
    ]
    for key, value in twiss.items():
        summary.append(f"twiss_{key}={value:.12g}")
    for key, value in sigmas.items():
        summary.append(f"{key}={value:.12g}")

    for mode in base.MODES:
        out_npz, _start, end = base.track_mode(mode, twiss, distribution, sigmas)
        key = base.mode_label(mode)
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

    summary_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_sr_summary.txt")
    summary_path.write_text("\n".join(summary) + "\n", encoding="ascii")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
