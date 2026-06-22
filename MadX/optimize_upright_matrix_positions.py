#!/usr/bin/env python3
"""Fast transfer-matrix search for upright quadrupole positions and gradients."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from optimize_upright_positions import BRHO_TM, INITIAL_BETA, generate_lattice, parse_float_list, warning_count
from plot_twiss import read_tfs


def plane_matrix(k: float, length: float) -> np.ndarray:
    if abs(k) < 1.0e-14:
        return np.array([[1.0, length], [0.0, 1.0]])
    root = math.sqrt(abs(k))
    phase = root * length
    if k > 0:
        return np.array(
            [
                [math.cos(phase), math.sin(phase) / root],
                [-root * math.sin(phase), math.cos(phase)],
            ]
        )
    return np.array(
        [
            [math.cosh(phase), math.sinh(phase) / root],
            [root * math.sinh(phase), math.cosh(phase)],
        ]
    )


def propagate_twiss(beta: float, alpha: float, matrix: np.ndarray) -> tuple[float, float]:
    gamma = (1.0 + alpha * alpha) / beta
    m11, m12 = matrix[0]
    m21, m22 = matrix[1]
    next_beta = m11 * m11 * beta - 2.0 * m11 * m12 * alpha + m12 * m12 * gamma
    next_alpha = -m11 * m21 * beta + (m11 * m22 + m12 * m21) * alpha - m12 * m22 * gamma
    return next_beta, next_alpha


def unpack(values: np.ndarray, n_quads: int) -> tuple[list[float], list[float]]:
    centers = sorted(float(value) for value in values[:n_quads])
    gradients = [float(value) for value in values[n_quads:]]
    return centers, gradients


def unpack_candidate(values: np.ndarray, args: argparse.Namespace) -> tuple[float, list[float], list[float]]:
    if args.length_window > 0:
        length = float(values[0])
        centers, gradients = unpack(values[1:], args.n_quads)
    else:
        length = args.length
        centers, gradients = unpack(values, args.n_quads)
    return length, centers, gradients


def optics(
    centers: list[float],
    gradients_tm: list[float],
    length: float,
    quad_length: float,
    sample_step: float,
) -> dict[str, float]:
    beta_x = INITIAL_BETA
    beta_y = INITIAL_BETA
    alpha_x = 0.0
    alpha_y = 0.0
    max_betx = beta_x
    max_bety = beta_y
    s = 0.0

    for center, gradient in zip(centers, gradients_tm):
        start = center - quad_length / 2.0
        end = center + quad_length / 2.0
        if start < s or end > length:
            raise ValueError("overlapping or out-of-bounds quadrupole")
        beta_x, alpha_x, beta_y, alpha_y, max_betx, max_bety = propagate_segment(
            beta_x, alpha_x, beta_y, alpha_y, max_betx, max_bety, 0.0, start - s, sample_step
        )
        k1 = gradient / BRHO_TM
        beta_x, alpha_x, beta_y, alpha_y, max_betx, max_bety = propagate_segment(
            beta_x, alpha_x, beta_y, alpha_y, max_betx, max_bety, k1, quad_length, sample_step
        )
        s = end

    beta_x, alpha_x, beta_y, alpha_y, max_betx, max_bety = propagate_segment(
        beta_x, alpha_x, beta_y, alpha_y, max_betx, max_bety, 0.0, length - s, sample_step
    )
    return {
        "betx": beta_x,
        "bety": beta_y,
        "alfx": alpha_x,
        "alfy": alpha_y,
        "max_betx": max_betx,
        "max_bety": max_bety,
    }


def propagate_segment(
    beta_x: float,
    alpha_x: float,
    beta_y: float,
    alpha_y: float,
    max_betx: float,
    max_bety: float,
    k1: float,
    length: float,
    sample_step: float,
) -> tuple[float, float, float, float, float, float]:
    if length <= 0:
        return beta_x, alpha_x, beta_y, alpha_y, max_betx, max_bety
    pieces = max(1, int(math.ceil(length / sample_step)))
    piece_length = length / pieces
    mx = plane_matrix(k1, piece_length)
    my = plane_matrix(-k1, piece_length)
    for _ in range(pieces):
        beta_x, alpha_x = propagate_twiss(beta_x, alpha_x, mx)
        beta_y, alpha_y = propagate_twiss(beta_y, alpha_y, my)
        if beta_x <= 0 or beta_y <= 0 or not math.isfinite(beta_x + beta_y + alpha_x + alpha_y):
            raise ValueError("invalid optics")
        max_betx = max(max_betx, beta_x)
        max_bety = max(max_bety, beta_y)
    return beta_x, alpha_x, beta_y, alpha_y, max_betx, max_bety


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-quads", type=int, required=True)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--length-window", type=float, default=0.0)
    parser.add_argument("--quad-length", type=float, default=4.0)
    parser.add_argument("--marker-step", type=float, default=4.0)
    parser.add_argument("--sample-step", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=20.0)
    parser.add_argument("--min-spacing", type=float, default=12.0)
    parser.add_argument("--initial-centers", required=True)
    parser.add_argument("--initial-gradients-tm", required=True)
    parser.add_argument("--position-window", type=float, default=28.0)
    parser.add_argument("--gradient-window-tm", type=float, default=55.0)
    parser.add_argument("--gradient-bound-tm", type=float, default=170.0)
    parser.add_argument("--target-betx", type=float, default=80.0)
    parser.add_argument("--target-bety", type=float, default=30.0)
    parser.add_argument("--emittance-nm", type=float, default=20.0)
    parser.add_argument("--beam-size-goal-m", type=float, default=0.035)
    parser.add_argument("--beam-size-weight", type=float, default=80.0)
    parser.add_argument("--endpoint-log-scale", type=float, default=0.04)
    parser.add_argument("--alpha-weight", type=float, default=4.0)
    parser.add_argument("--strength-weight", type=float, default=0.02)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--local-maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    work_dir = base_dir / (args.work_dir or f"upright_matrix_{args.n_quads}q_work")
    work_dir.mkdir(exist_ok=True)
    centers = parse_float_list(args.initial_centers)
    gradients = parse_float_list(args.initial_gradients_tm)
    if len(centers) != args.n_quads or len(gradients) != args.n_quads:
        raise ValueError("initial centers and gradients must match --n-quads")

    bounds: list[tuple[float, float]] = []
    if args.length_window > 0:
        min_length = max(args.margin * 2.0, args.length - args.length_window)
        max_length = args.length + args.length_window
        bounds.append((min_length, max_length))
    for center in centers:
        bounds.append((max(args.margin, center - args.position_window), min(args.length - args.margin, center + args.position_window)))
    for gradient in gradients:
        bounds.append((max(-args.gradient_bound_tm, gradient - args.gradient_window_tm), min(args.gradient_bound_tm, gradient + args.gradient_window_tm)))

    best: dict[str, float | list[float]] | None = None
    counter = {"n": 0}

    def objective(values: np.ndarray) -> float:
        counter["n"] += 1
        nonlocal best
        try:
            cand_length, cand_centers, cand_gradients = unpack_candidate(values, args)
            if cand_centers[-1] + args.quad_length / 2.0 > cand_length:
                return 1.0e12
            if any(right - left < args.min_spacing for left, right in zip(cand_centers, cand_centers[1:])):
                return 1.0e12
            result = optics(cand_centers, cand_gradients, cand_length, args.quad_length, args.sample_step)
        except ValueError:
            return 1.0e12
        eps_m = args.emittance_nm * 1.0e-9
        sigma_x = math.sqrt(result["max_betx"] * eps_m)
        sigma_y = math.sqrt(result["max_bety"] * eps_m)
        max_sigma = max(sigma_x, sigma_y)
        endpoint = (math.log(result["betx"] / args.target_betx) / args.endpoint_log_scale) ** 2
        endpoint += (math.log(result["bety"] / args.target_bety) / args.endpoint_log_scale) ** 2
        alpha = args.alpha_weight * (result["alfx"] ** 2 + result["alfy"] ** 2)
        beam_size = args.beam_size_weight * (math.log(max_sigma / args.beam_size_goal_m) ** 2)
        strength = args.strength_weight * sum((g / args.gradient_bound_tm) ** 2 for g in cand_gradients)
        score = endpoint + alpha + beam_size + strength
        if best is None or score < float(best["objective"]):
            best = {
                "objective": score,
                "length": cand_length,
                "centers": cand_centers,
                "gradients": cand_gradients,
                **result,
                "max_sigma_x_m": sigma_x,
                "max_sigma_y_m": sigma_y,
            }
            print(
                "best "
                f"eval={counter['n']} obj={score:.6g} "
                f"L={cand_length:.4g} m "
                f"S={','.join(f'{v:.4g}' for v in cand_centers)} m "
                f"G={','.join(f'{v:.4g}' for v in cand_gradients)} T/m "
                f"end=({result['betx']:.4g},{result['bety']:.4g}) "
                f"alpha=({result['alfx']:.3g},{result['alfy']:.3g}) "
                f"max_sigma=({sigma_x:.4g},{sigma_y:.4g})m",
                flush=True,
            )
        return score

    initial_values = np.array(
        ([args.length] if args.length_window > 0 else []) + [*centers, *gradients],
        dtype=float,
    )
    objective(initial_values)
    de_result = differential_evolution(
        objective,
        bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        polish=False,
        updating="immediate",
        workers=1,
        tol=0.003,
    )
    minimize(
        objective,
        de_result.x,
        method="Powell",
        bounds=bounds,
        options={"maxiter": args.local_maxiter, "xtol": 1.0e-5, "ftol": 1.0e-5},
    )
    if best is None:
        raise RuntimeError("no valid candidate found")

    best_length = float(best["length"])
    best_centers = [float(v) for v in best["centers"]]
    best_gradients = [float(v) for v in best["gradients"]]
    madx_file = work_dir / "best.madx"
    twiss_file = work_dir / "best.tfs"
    log_file = work_dir / "best.log"
    generate_lattice(madx_file, twiss_file, best_length, args.marker_step, args.quad_length, best_centers, best_gradients)
    with log_file.open("w", encoding="utf-8") as handle:
        subprocess.run(["./madx", str(madx_file)], cwd=base_dir, stdout=handle, stderr=subprocess.STDOUT, check=True)

    twiss = read_tfs(twiss_file)
    last = twiss.iloc[-1]
    eps_m = args.emittance_nm * 1.0e-9
    sigma_x = math.sqrt(float(twiss["BETX"].max()) * eps_m)
    sigma_y = math.sqrt(float(twiss["BETY"].max()) * eps_m)
    warnings = warning_count(log_file)

    if args.output_prefix:
        prefix = base_dir / args.output_prefix
        shutil.copyfile(madx_file, prefix.with_suffix(".madx"))
        shutil.copyfile(log_file, prefix.with_suffix(".log"))
        shutil.copyfile(twiss_file, prefix.with_name(prefix.name + "_twiss.tfs"))

    print("\nMAD-X verified best")
    print(f"length_m: {best_length:.9g}")
    print(f"centers_m: {','.join(f'{v:.6g}' for v in best_centers)}")
    print(f"gradients_tm: {','.join(f'{v:.6g}' for v in best_gradients)}")
    print(f"betx: {float(last['BETX']):.9g}")
    print(f"bety: {float(last['BETY']):.9g}")
    print(f"alfx: {float(last['ALFX']):.9g}")
    print(f"alfy: {float(last['ALFY']):.9g}")
    print(f"max_betx: {float(twiss['BETX'].max()):.9g}")
    print(f"max_bety: {float(twiss['BETY'].max()):.9g}")
    print(f"max_sigma_x_m: {sigma_x:.9g}")
    print(f"max_sigma_y_m: {sigma_y:.9g}")
    print(f"warnings: {warnings}")


if __name__ == "__main__":
    main()
