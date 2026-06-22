#!/usr/bin/env python3
"""Find simple upright-quadrupole lattices for the flat optics target."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from plot_twiss import read_tfs


PC_GEV = 5000.0
BRHO_TM = PC_GEV / 0.299792458
TARGET_BETX = 80.0
TARGET_BETY = 30.0
INITIAL_BETA = 0.2351


def parse_float_list(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def warning_count(log_file: Path) -> int:
    text = log_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Number of warnings:\s+(\d+)", text)
    if match:
        return int(match.group(1))
    return text.count("++++++ warning:")


def default_positions(length: float, n_quads: int, margin: float, step: float) -> list[float]:
    raw = np.linspace(margin, length - margin - step, n_quads)
    return [round(float(value) / step) * step for value in raw]


def generate_lattice(
    path: Path,
    twiss_file: Path,
    length: float,
    step: float,
    quad_starts: list[float],
    gradients_tm: list[float],
) -> None:
    gradient_by_start = dict(zip(quad_starts, gradients_tm))
    positions = [i * step for i in range(int(length / step) + 1)]
    lines = [
        'title, "5 TeV electron upright-quadrupole-only transformer";',
        "",
        "beam, particle=electron, energy=5000;",
        "",
        f"! Brho = {BRHO_TM:.9f} T*m",
        f"upright: sequence, l={length:.6f};",
    ]
    for index, start in enumerate(positions):
        lines.append(f"  mk{index:03d}: marker, at={start:.6f};")
        if index + 1 >= len(positions):
            continue
        center = start + step / 2.0
        if start in gradient_by_start:
            k1 = gradient_by_start[start] / BRHO_TM
            lines.append(f"  q{len([s for s in quad_starts if s <= start]):03d}: quadrupole, l={step:.6f}, k1={k1:.12g}, at={center:.6f};")
        else:
            lines.append(f"  dr{index:03d}: drift, l={step:.6f}, at={center:.6f};")
    lines.extend(
        [
            "endsequence;",
            "",
            "use, sequence=upright;",
            "",
            "twiss,",
            f"  betx={INITIAL_BETA:.12g}, alfx=0,",
            f"  bety={INITIAL_BETA:.12g}, alfy=0,",
            "  dx=0, dpx=0,",
            "  dy=0, dpy=0,",
            f'  file="{twiss_file}";',
            "",
            "stop;",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    values: np.ndarray,
    args: argparse.Namespace,
    work_dir: Path,
    quad_starts: list[float],
    serial: int,
) -> dict[str, float | str | Path]:
    gradients = [float(value) for value in values]
    madx_file = work_dir / "candidate.madx"
    log_file = work_dir / "candidate.log"
    twiss_file = work_dir / "candidate.tfs"
    generate_lattice(madx_file, twiss_file, args.length, args.step, quad_starts, gradients)
    with log_file.open("w", encoding="utf-8") as handle:
        subprocess.run(["./madx", str(madx_file)], cwd=args.base_dir, stdout=handle, stderr=subprocess.STDOUT, check=True)

    twiss = read_tfs(twiss_file)
    last = twiss.iloc[-1]
    betx = float(last["BETX"])
    bety = float(last["BETY"])
    alfx = float(last["ALFX"])
    alfy = float(last["ALFY"])
    max_betx = float(twiss["BETX"].max())
    max_bety = float(twiss["BETY"].max())
    eps_m = args.emittance_nm * 1.0e-9
    max_sigma_x = math.sqrt(max_betx * eps_m)
    max_sigma_y = math.sqrt(max_bety * eps_m)
    max_sigma = max(max_sigma_x, max_sigma_y)
    warnings = warning_count(log_file)

    if not all(math.isfinite(v) and v > 0 for v in (betx, bety, max_betx, max_bety, max_sigma)):
        raise ValueError("Non-finite or non-positive optics")

    endpoint = (math.log(betx / args.target_betx) / args.endpoint_log_scale) ** 2
    endpoint += (math.log(bety / args.target_bety) / args.endpoint_log_scale) ** 2
    alpha = args.alpha_weight * (alfx**2 + alfy**2)
    beam_size = args.beam_size_weight * (math.log(max_sigma / args.beam_size_goal_m) ** 2)
    strength = args.strength_weight * sum((g / args.gradient_bound_tm) ** 2 for g in gradients)
    warning_penalty = args.warning_weight * warnings
    objective = endpoint + alpha + beam_size + strength + warning_penalty

    if args.keep_every > 0 and serial % args.keep_every == 0:
        shutil.copyfile(madx_file, work_dir / f"candidate_{serial:05d}.madx")
        shutil.copyfile(log_file, work_dir / f"candidate_{serial:05d}.log")
        shutil.copyfile(twiss_file, work_dir / f"candidate_{serial:05d}.tfs")

    return {
        "objective": objective,
        "gradients_tm": ",".join(f"{value:.6g}" for value in gradients),
        "betx": betx,
        "bety": bety,
        "alfx": alfx,
        "alfy": alfy,
        "max_betx": max_betx,
        "max_bety": max_bety,
        "max_sigma_x_m": max_sigma_x,
        "max_sigma_y_m": max_sigma_y,
        "max_sigma_m": max_sigma,
        "warnings": warnings,
        "madx_file": madx_file,
        "log_file": log_file,
        "twiss_file": twiss_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-quads", type=int, required=True)
    parser.add_argument("--length", type=float, default=360.0)
    parser.add_argument("--step", type=float, default=4.0)
    parser.add_argument("--margin", type=float, default=20.0)
    parser.add_argument("--positions", default="")
    parser.add_argument("--initial-gradients-tm", default="")
    parser.add_argument("--gradient-bound-tm", type=float, default=140.0)
    parser.add_argument("--target-betx", type=float, default=TARGET_BETX)
    parser.add_argument("--target-bety", type=float, default=TARGET_BETY)
    parser.add_argument("--emittance-nm", type=float, default=20.0)
    parser.add_argument("--beam-size-goal-m", type=float, default=0.02)
    parser.add_argument("--beam-size-weight", type=float, default=120.0)
    parser.add_argument("--endpoint-log-scale", type=float, default=0.06)
    parser.add_argument("--alpha-weight", type=float, default=0.05)
    parser.add_argument("--strength-weight", type=float, default=0.02)
    parser.add_argument("--warning-weight", type=float, default=300.0)
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--local-maxiter", type=int, default=120)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--keep-every", type=int, default=0)
    parser.add_argument("--work-dir", default="")
    args = parser.parse_args()
    args.base_dir = Path(__file__).resolve().parent
    work_dir = args.base_dir / (args.work_dir or f"upright_{args.n_quads}q_work")
    work_dir.mkdir(exist_ok=True)

    if args.positions:
        quad_starts = parse_float_list(args.positions)
    else:
        quad_starts = default_positions(args.length, args.n_quads, args.margin, args.step)
    if len(quad_starts) != args.n_quads:
        raise ValueError("--positions count must match --n-quads")

    bounds = [(-args.gradient_bound_tm, args.gradient_bound_tm)] * args.n_quads
    initial_values = parse_float_list(args.initial_gradients_tm) if args.initial_gradients_tm else None
    if initial_values is not None and len(initial_values) != args.n_quads:
        raise ValueError("--initial-gradients-tm count must match --n-quads")
    counter = {"n": 0}
    best: dict[str, float | str | Path] | None = None

    def objective(values: np.ndarray) -> float:
        counter["n"] += 1
        nonlocal best
        try:
            result = evaluate(values, args, work_dir, quad_starts, counter["n"])
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            return 1.0e12
        if best is None or float(result["objective"]) < float(best["objective"]):
            best = result
            print(
                "best "
                f"eval={counter['n']} obj={result['objective']:.6g} "
                f"G={result['gradients_tm']} T/m "
                f"end=({result['betx']:.4g},{result['bety']:.4g}) "
                f"alpha=({result['alfx']:.3g},{result['alfy']:.3g}) "
                f"max_sigma=({result['max_sigma_x_m']:.4g},{result['max_sigma_y_m']:.4g})m "
                f"warnings={result['warnings']}",
                flush=True,
            )
            shutil.copyfile(result["madx_file"], work_dir / "best.madx")
            shutil.copyfile(result["log_file"], work_dir / "best.log")
            shutil.copyfile(result["twiss_file"], work_dir / "best.tfs")
        return float(result["objective"])

    if initial_values is not None:
        minimize(
            objective,
            np.array(initial_values, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": args.local_maxiter, "xatol": 1.0e-3, "fatol": 1.0e-3},
        )
    if not args.local_only:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=args.seed,
            polish=False,
            updating="immediate",
            workers=1,
            tol=0.01,
        )
        minimize(
            objective,
            result.x,
            method="Nelder-Mead",
            options={"maxiter": args.local_maxiter, "xatol": 1.0e-3, "fatol": 1.0e-3},
        )
    if best is None:
        raise RuntimeError("No valid candidate found")

    print("\nFinal best")
    print(f"positions_m: {','.join(f'{p:g}' for p in quad_starts)}")
    for key in (
        "objective",
        "gradients_tm",
        "betx",
        "bety",
        "alfx",
        "alfy",
        "max_betx",
        "max_bety",
        "max_sigma_x_m",
        "max_sigma_y_m",
        "warnings",
    ):
        print(f"{key}: {best[key]}")


if __name__ == "__main__":
    main()
