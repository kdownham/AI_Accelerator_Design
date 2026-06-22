#!/usr/bin/env python3
"""Save particle snapshots at each FACET element and render phase-space GIF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo

from run_facet_madx_in_xsuite import build_line
from track_facet_particles_cpu import N_PARTICLES, RNG_SEED, SIGMAS, make_particles


ROOT = Path(__file__).resolve().parent
OUT_PREFIX = ROOT / "facet_particles_element_snapshots"


def as_array(values) -> np.ndarray:
    return np.asarray(values)


def cumulative_s(line) -> np.ndarray:
    positions = [0.0]
    s = 0.0
    for elem in line.elements:
        s += float(getattr(elem, "length", 0.0) or 0.0)
        positions.append(s)
    return np.array(positions, dtype=np.float64)


def snapshot_particles(particles) -> dict[str, np.ndarray]:
    return {
        "x": as_array(particles.x).astype(np.float32).copy(),
        "px": as_array(particles.px).astype(np.float32).copy(),
        "y": as_array(particles.y).astype(np.float32).copy(),
        "py": as_array(particles.py).astype(np.float32).copy(),
        "zeta": as_array(particles.zeta).astype(np.float32).copy(),
        "delta": as_array(particles.delta).astype(np.float32).copy(),
        "state": as_array(particles.state).astype(np.int16).copy(),
    }


def track_and_save() -> tuple[Path, dict[str, np.ndarray | list[str]]]:
    context = xo.ContextCpu()
    line, kinds = build_line()
    line.build_tracker(_context=context)

    particles, _initial = make_particles(context)
    n_frames = len(line.elements) + 1
    names = ["START"] + list(line.element_names)
    s_positions = cumulative_s(line)

    data = {
        "x": np.empty((n_frames, N_PARTICLES), dtype=np.float32),
        "px": np.empty((n_frames, N_PARTICLES), dtype=np.float32),
        "y": np.empty((n_frames, N_PARTICLES), dtype=np.float32),
        "py": np.empty((n_frames, N_PARTICLES), dtype=np.float32),
        "zeta": np.empty((n_frames, N_PARTICLES), dtype=np.float32),
        "delta": np.empty((n_frames, N_PARTICLES), dtype=np.float32),
        "state": np.empty((n_frames, N_PARTICLES), dtype=np.int16),
    }

    snap = snapshot_particles(particles)
    for key, values in snap.items():
        data[key][0, :] = values

    for ii in range(len(line.elements)):
        line.track(particles, ele_start=ii, num_elements=1)
        snap = snapshot_particles(particles)
        for key, values in snap.items():
            data[key][ii + 1, :] = values

    out_npz = OUT_PREFIX.with_name(OUT_PREFIX.name + ".npz")
    np.savez_compressed(
        out_npz,
        element_names=np.array(names, dtype=object),
        s=s_positions,
        rng_seed=np.array(RNG_SEED),
        sigmas=np.array([SIGMAS[k] for k in ["x", "y", "px", "py", "zeta", "delta"]]),
        sigma_names=np.array(["x", "y", "px", "py", "zeta", "delta"], dtype=object),
        **data,
    )

    summary_path = OUT_PREFIX.with_name(OUT_PREFIX.name + "_summary.txt")
    final_alive = int((data["state"][-1] > 0).sum())
    summary = [
        f"context={context.__class__.__name__}",
        f"particles={N_PARTICLES}",
        f"frames={n_frames}",
        f"elements={len(line.elements)}",
        f"rng_seed={RNG_SEED}",
        "sigmas=" + ", ".join(f"{k}:{v:g}" for k, v in SIGMAS.items()),
        f"final_alive={final_alive}",
        f"final_lost={N_PARTICLES - final_alive}",
        "element_types=" + ", ".join(f"{k}:{v}" for k, v in sorted(kinds.items())),
        f"snapshots_npz={out_npz}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return out_npz, {"names": names, "s": s_positions, **data}


def axis_limits(pos_um: np.ndarray, mom_urad: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    finite_pos = pos_um[np.isfinite(pos_um)]
    finite_mom = mom_urad[np.isfinite(mom_urad)]
    xlim = (float(finite_pos.min()), float(finite_pos.max()))
    ylim = (float(finite_mom.min()), float(finite_mom.max()))
    for lim_name, lim in [("x", xlim), ("y", ylim)]:
        span = lim[1] - lim[0]
        if span == 0:
            span = 1.0
        pad = 0.06 * span
        if lim_name == "x":
            xlim = (lim[0] - pad, lim[1] + pad)
        else:
            ylim = (lim[0] - pad, lim[1] + pad)
    return xlim, ylim


def render_gif(
    snapshot_npz: Path | None = None,
    out_gif: Path | None = None,
    fps: int = 8,
    title_label: str = "FACET CPU tracking phase space",
) -> Path:
    if snapshot_npz is None:
        snapshot_npz = OUT_PREFIX.with_name(OUT_PREFIX.name + ".npz")
    data = np.load(snapshot_npz, allow_pickle=True)

    names = data["element_names"]
    s_positions = data["s"]
    x_um = data["x"] * 1e6
    px_urad = data["px"] * 1e6
    y_um = data["y"] * 1e6
    py_urad = data["py"] * 1e6
    state = data["state"]

    xlim, pxlim = axis_limits(x_um, px_urad)
    ylim, pylim = axis_limits(y_um, py_urad)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), constrained_layout=True)
    ax_x, ax_y = axes
    sx = ax_x.scatter([], [], s=2.2, alpha=0.22, color="#0f766e", linewidths=0)
    sy = ax_y.scatter([], [], s=2.2, alpha=0.22, color="#c2410c", linewidths=0)

    for ax, lim_x, lim_y, xlabel, ylabel in [
        (ax_x, xlim, pxlim, "x [um]", "px [urad]"),
        (ax_y, ylim, pylim, "y [um]", "py [urad]"),
    ]:
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="#404040", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.25)

    ax_x.set_title("x-px")
    ax_y.set_title("y-py")
    title = fig.suptitle("")

    def update(frame: int):
        alive = state[frame] > 0
        sx.set_offsets(np.column_stack([x_um[frame, alive], px_urad[frame, alive]]))
        sy.set_offsets(np.column_stack([y_um[frame, alive], py_urad[frame, alive]]))
        title.set_text(
            f"{title_label} | frame {frame + 1}/{len(names)} | "
            f"element: {names[frame]} | s={s_positions[frame]:.3f} m | "
            f"alive={int(alive.sum())}/{alive.size}"
        )
        return sx, sy, title

    anim = animation.FuncAnimation(fig, update, frames=len(names), interval=120, blit=False)
    if out_gif is None:
        out_gif = OUT_PREFIX.with_name(OUT_PREFIX.name + "_phase_space.gif")
    anim.save(out_gif, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"gif={out_gif}")
    return out_gif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true", help="Use existing snapshot NPZ without tracking again.")
    parser.add_argument("--snapshot-npz", type=Path, default=OUT_PREFIX.with_name(OUT_PREFIX.name + ".npz"))
    parser.add_argument("--output-gif", type=Path, default=OUT_PREFIX.with_name(OUT_PREFIX.name + "_phase_space.gif"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--title-label", default="FACET CPU tracking phase space")
    args = parser.parse_args()

    if args.render_only:
        render_gif(args.snapshot_npz, args.output_gif, fps=args.fps, title_label=args.title_label)
    else:
        snapshot_npz, _data = track_and_save()
        render_gif(snapshot_npz, args.output_gif, fps=args.fps, title_label=args.title_label)


if __name__ == "__main__":
    main()
