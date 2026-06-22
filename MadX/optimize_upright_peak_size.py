#!/usr/bin/env python3
"""Constrained peak beam-size reduction for upright-quadrupole lattices."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from optimize_upright_matrix_positions import plane_matrix, propagate_twiss
from optimize_upright_positions import INITIAL_BETA, generate_lattice, parse_float_list, warning_count
from plot_twiss import beam_size_meters, read_tfs


BASELINE_CENTERS = "53.3209,92.0096,166.617,178.617,238,358.43,380.928,437.371"
BASELINE_GRADIENTS = "-130.96418,101.36698,-46.700356,16.907841,-18.115517,-132.67901,50.576922,-29.140564"


def unpack(values: np.ndarray, n_quads: int) -> tuple[float, list[float], list[float]]:
    length = float(values[0])
    centers = sorted(float(value) for value in values[1 : 1 + n_quads])
    gradients = [float(value) for value in values[1 + n_quads :]]
    return length, centers, gradients


def sampled_optics(
    length: float,
    centers: list[float],
    gradients_tm: list[float],
    brho_tm: float,
    quad_length: float,
    sample_step: float,
) -> dict[str, float | np.ndarray]:
    beta_x = INITIAL_BETA
    beta_y = INITIAL_BETA
    alpha_x = 0.0
    alpha_y = 0.0
    s = 0.0
    betx_samples = [beta_x]
    bety_samples = [beta_y]

    def advance(k1: float, segment_length: float) -> None:
        nonlocal beta_x, alpha_x, beta_y, alpha_y
        if segment_length <= 0:
            return
        pieces = max(1, int(math.ceil(segment_length / sample_step)))
        piece_length = segment_length / pieces
        mx = plane_matrix(k1, piece_length)
        my = plane_matrix(-k1, piece_length)
        for _ in range(pieces):
            beta_x, alpha_x = propagate_twiss(beta_x, alpha_x, mx)
            beta_y, alpha_y = propagate_twiss(beta_y, alpha_y, my)
            if beta_x <= 0 or beta_y <= 0 or not math.isfinite(beta_x + beta_y + alpha_x + alpha_y):
                raise ValueError("invalid optics")
            betx_samples.append(beta_x)
            bety_samples.append(beta_y)

    for center, gradient in zip(centers, gradients_tm):
        start = center - quad_length / 2.0
        end = center + quad_length / 2.0
        if start < s or end > length:
            raise ValueError("overlapping or out-of-bounds quadrupole")
        advance(0.0, start - s)
        advance(gradient / brho_tm, quad_length)
        s = end
    advance(0.0, length - s)

    return {
        "betx": beta_x,
        "bety": beta_y,
        "alfx": alpha_x,
        "alfy": alpha_y,
        "betx_samples": np.array(betx_samples),
        "bety_samples": np.array(bety_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=528.137925)
    parser.add_argument("--centers", default=BASELINE_CENTERS)
    parser.add_argument("--gradients-tm", default=BASELINE_GRADIENTS)
    parser.add_argument("--quad-length", type=float, default=4.0)
    parser.add_argument("--marker-step", type=float, default=4.0)
    parser.add_argument("--sample-step", type=float, default=1.0)
    parser.add_argument("--position-window", type=float, default=45.0)
    parser.add_argument("--length-window", type=float, default=80.0)
    parser.add_argument("--gradient-window-tm", type=float, default=70.0)
    parser.add_argument("--gradient-bound-tm", type=float, default=170.0)
    parser.add_argument("--min-spacing", type=float, default=12.0)
    parser.add_argument("--target-betx", type=float, default=80.0)
    parser.add_argument("--target-bety", type=float, default=30.0)
    parser.add_argument("--emittance-nm", type=float, default=20.0)
    parser.add_argument("--p-norm", type=float, default=20.0)
    parser.add_argument("--maxiter", type=int, default=700)
    parser.add_argument("--output-prefix", default="upright_peak_reduced_8q")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    centers = parse_float_list(args.centers)
    gradients = parse_float_list(args.gradients_tm)
    if len(centers) != len(gradients):
        raise ValueError("centers and gradients must have the same length")
    n_quads = len(centers)
    brho_tm = 5000.0 / 0.299792458
    eps_m = args.emittance_nm * 1.0e-9

    initial = np.array([args.length, *centers, *gradients], dtype=float)
    bounds: list[tuple[float, float]] = [
        (max(centers[-1] + args.quad_length / 2.0 + 1.0, args.length - args.length_window), args.length + args.length_window)
    ]
    bounds.extend((max(20.0, center - args.position_window), min(args.length - 20.0, center + args.position_window)) for center in centers)
    bounds.extend(
        (max(-args.gradient_bound_tm, gradient - args.gradient_window_tm), min(args.gradient_bound_tm, gradient + args.gradient_window_tm))
        for gradient in gradients
    )

    def evaluate(values: np.ndarray) -> dict[str, float | np.ndarray]:
        length, trial_centers, trial_gradients = unpack(values, n_quads)
        if trial_centers[-1] + args.quad_length / 2.0 > length:
            raise ValueError("last quadrupole outside lattice")
        if any(right - left < args.min_spacing for left, right in zip(trial_centers, trial_centers[1:])):
            raise ValueError("spacing violation")
        return sampled_optics(length, trial_centers, trial_gradients, brho_tm, args.quad_length, args.sample_step)

    def peak_size_objective(values: np.ndarray) -> float:
        try:
            result = evaluate(values)
        except ValueError:
            return 1.0e6
        sigma_x = np.sqrt(np.asarray(result["betx_samples"]) * eps_m)
        sigma_y = np.sqrt(np.asarray(result["bety_samples"]) * eps_m)
        sigma = np.r_[sigma_x, sigma_y]
        normalized = sigma / 0.04
        return float(np.mean(normalized**args.p_norm) ** (1.0 / args.p_norm))

    def endpoint_constraints(values: np.ndarray) -> np.ndarray:
        try:
            result = evaluate(values)
        except ValueError:
            return np.ones(4) * 1.0e3
        return np.array(
            [
                math.log(float(result["betx"]) / args.target_betx),
                math.log(float(result["bety"]) / args.target_bety),
                float(result["alfx"]),
                float(result["alfy"]),
            ]
        )

    spacing_constraints = []
    for index in range(n_quads - 1):
        spacing_constraints.append(
            {
                "type": "ineq",
                "fun": lambda values, i=index: sorted(values[1 : 1 + n_quads])[i + 1]
                - sorted(values[1 : 1 + n_quads])[i]
                - args.min_spacing,
            }
        )
    constraints = [{"type": "eq", "fun": endpoint_constraints}, *spacing_constraints]
    result = minimize(
        peak_size_objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": args.maxiter, "ftol": 1.0e-11, "disp": True},
    )

    length, matched_centers, matched_gradients = unpack(result.x, n_quads)
    prefix = base_dir / args.output_prefix
    madx_file = prefix.with_suffix(".madx")
    log_file = prefix.with_suffix(".log")
    twiss_file = prefix.with_name(prefix.name + "_twiss.tfs")
    generate_lattice(
        madx_file,
        twiss_file,
        length,
        args.marker_step,
        args.quad_length,
        matched_centers,
        matched_gradients,
    )
    with log_file.open("w", encoding="utf-8") as handle:
        subprocess.run(["./madx", str(madx_file)], cwd=base_dir, stdout=handle, stderr=subprocess.STDOUT, check=True)

    twiss = read_tfs(twiss_file)
    last = twiss.iloc[-1]
    sigma_x, sigma_y = beam_size_meters(twiss, args.emittance_nm)
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"length_m: {length:.9g}")
    print(f"centers_m: {','.join(f'{value:.6g}' for value in matched_centers)}")
    print(f"gradients_tm: {','.join(f'{value:.6g}' for value in matched_gradients)}")
    print(f"betx: {float(last['BETX']):.9g}")
    print(f"bety: {float(last['BETY']):.9g}")
    print(f"alfx: {float(last['ALFX']):.9g}")
    print(f"alfy: {float(last['ALFY']):.9g}")
    print(f"max_sigma_x_m: {float(sigma_x.max()):.9g}")
    print(f"max_sigma_y_m: {float(sigma_y.max()):.9g}")
    print(f"warnings: {warning_count(log_file)}")


if __name__ == "__main__":
    main()
