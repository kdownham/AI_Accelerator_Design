#!/usr/bin/env python3
"""Track a realistic FACET distribution with independent x/y samples."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

from run_facet_madx_in_xsuite import build_line
from track_facet_particles_element_gif import cumulative_s, render_gif, snapshot_particles


ROOT = Path(__file__).resolve().parent
OUT_PREFIX = ROOT / "facet_realistic_independent_snapshots"

N_MACROPARTICLES_B1 = int(1e4)
RNG_SEED = 20260617
ENERGY = 10e9
GAMMA = ENERGY / xt.ELECTRON_MASS_EV
BUNCH_INTENSITY = 1.6e-9 / 1.6e-19

BETA_X = 29.51561770054117
BETA_Y = 45.17494646985441
ALPHA_X = 2.66413224525837
ALPHA_Y = -3.58216043939153

PHYSEMIT_X = 5e-6 / GAMMA
PHYSEMIT_Y = 5e-6 / GAMMA

SIGMA_X = np.sqrt(BETA_X * PHYSEMIT_X)
SIGMA_Y = np.sqrt(BETA_Y * PHYSEMIT_Y)
SIGMA_PX = SIGMA_X / BETA_X
SIGMA_PY = SIGMA_Y / BETA_Y

SIGMA_Z = 20e-6
SIGMA_DELTA = 1e-4


def make_realistic_particles(context: xo.ContextCpu) -> tuple[xp.Particles, dict[str, np.ndarray]]:
    rng = np.random.default_rng(RNG_SEED)

    r1x = rng.standard_normal(N_MACROPARTICLES_B1)
    r2x = rng.standard_normal(N_MACROPARTICLES_B1)
    r1y = rng.standard_normal(N_MACROPARTICLES_B1)
    r2y = rng.standard_normal(N_MACROPARTICLES_B1)

    initial = {
        "x": SIGMA_X * r1x,
        "px": SIGMA_PX * (-ALPHA_X * r1x + r2x),
        "y": SIGMA_Y * r1y,
        "py": SIGMA_PY * (-ALPHA_Y * r1y + r2y),
        "zeta": SIGMA_Z * rng.standard_normal(N_MACROPARTICLES_B1),
        "delta": SIGMA_DELTA * rng.standard_normal(N_MACROPARTICLES_B1),
    }
    weights = np.full(N_MACROPARTICLES_B1, BUNCH_INTENSITY / N_MACROPARTICLES_B1)

    particles = xp.Particles(
        _context=context,
        q0=-1,
        p0c=ENERGY,
        mass0=xt.ELECTRON_MASS_EV,
        x=initial["x"],
        px=initial["px"],
        y=initial["y"],
        py=initial["py"],
        zeta=initial["zeta"],
        delta=initial["delta"],
        weight=weights,
    )
    return particles, initial


def track_and_save() -> Path:
    context = xo.ContextCpu()
    line, kinds = build_line()
    line.build_tracker(_context=context)
    particles, initial = make_realistic_particles(context)

    n_frames = len(line.elements) + 1
    names = ["START"] + list(line.element_names)
    s_positions = cumulative_s(line)

    data = {
        "x": np.empty((n_frames, N_MACROPARTICLES_B1), dtype=np.float32),
        "px": np.empty((n_frames, N_MACROPARTICLES_B1), dtype=np.float32),
        "y": np.empty((n_frames, N_MACROPARTICLES_B1), dtype=np.float32),
        "py": np.empty((n_frames, N_MACROPARTICLES_B1), dtype=np.float32),
        "zeta": np.empty((n_frames, N_MACROPARTICLES_B1), dtype=np.float32),
        "delta": np.empty((n_frames, N_MACROPARTICLES_B1), dtype=np.float32),
        "state": np.empty((n_frames, N_MACROPARTICLES_B1), dtype=np.int16),
    }

    snap = snapshot_particles(particles)
    for key, values in snap.items():
        data[key][0, :] = values

    for ii in range(len(line.elements)):
        line.track(particles, ele_start=ii, num_elements=1)
        snap = snapshot_particles(particles)
        for key, values in snap.items():
            data[key][ii + 1, :] = values

    weight = np.asarray(particles.weight).astype(np.float64).copy()
    out_npz = OUT_PREFIX.with_name(OUT_PREFIX.name + ".npz")
    np.savez_compressed(
        out_npz,
        element_names=np.array(names, dtype=object),
        s=s_positions,
        rng_seed=np.array(RNG_SEED),
        particle_weight=weight,
        initial_x=initial["x"],
        initial_px=initial["px"],
        initial_y=initial["y"],
        initial_py=initial["py"],
        initial_zeta=initial["zeta"],
        initial_delta=initial["delta"],
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
        **data,
    )

    final_alive = int((data["state"][-1] > 0).sum())
    corr_xy = float(np.corrcoef(initial["x"], initial["y"])[0, 1])
    corr_xpx = float(np.corrcoef(initial["x"], initial["px"])[0, 1])
    corr_ypy = float(np.corrcoef(initial["y"], initial["py"])[0, 1])

    summary_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.txt")
    summary = [
        f"context={context.__class__.__name__}",
        f"particles={N_MACROPARTICLES_B1}",
        f"frames={n_frames}",
        f"elements={len(line.elements)}",
        f"rng_seed={RNG_SEED}",
        f"energy_ev={ENERGY:.12g}",
        f"gamma={GAMMA:.12g}",
        f"bunch_intensity_e={BUNCH_INTENSITY:.12g}",
        f"macroparticle_weight={BUNCH_INTENSITY / N_MACROPARTICLES_B1:.12g}",
        f"beta_x={BETA_X:.12g}",
        f"beta_y={BETA_Y:.12g}",
        f"alpha_x={ALPHA_X:.12g}",
        f"alpha_y={ALPHA_Y:.12g}",
        f"physemit_x={PHYSEMIT_X:.12g}",
        f"physemit_y={PHYSEMIT_Y:.12g}",
        f"sigma_x={SIGMA_X:.12g}",
        f"sigma_y={SIGMA_Y:.12g}",
        f"sigma_px={SIGMA_PX:.12g}",
        f"sigma_py={SIGMA_PY:.12g}",
        f"sigma_z={SIGMA_Z:.12g}",
        f"sigma_delta={SIGMA_DELTA:.12g}",
        "sampling=independent r1/r2 pairs for x-px and y-py",
        f"initial_corr_x_y={corr_xy:.12g}",
        f"initial_corr_x_px={corr_xpx:.12g}",
        f"initial_corr_y_py={corr_ypy:.12g}",
        f"final_alive={final_alive}",
        f"final_lost={N_MACROPARTICLES_B1 - final_alive}",
        "element_types=" + ", ".join(f"{k}:{v}" for k, v in sorted(kinds.items())),
        f"snapshots_npz={out_npz}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return out_npz


def main() -> None:
    out_npz = track_and_save()
    out_gif = OUT_PREFIX.with_name(OUT_PREFIX.name + "_phase_space.gif")
    render_gif(
        out_npz,
        out_gif,
        fps=4,
        title_label="FACET realistic independent x/y CPU tracking",
    )


if __name__ == "__main__":
    main()
