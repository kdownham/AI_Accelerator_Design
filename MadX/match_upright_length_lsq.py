#!/usr/bin/env python3
"""Least-squares rematch of an upright-quadrupole lattice length and gradients."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from optimize_upright_matrix_positions import optics
from optimize_upright_positions import generate_lattice, parse_float_list, warning_count
from plot_twiss import beam_size_meters, read_tfs


DEFAULT_CENTERS = "53.3209,92.0096,166.617,178.617,238,358.43,380.928,437.371"
DEFAULT_GRADIENTS = "-136.496,108,-70.6271,0.733649,-5.47106,-109.445,97.7129,-28.529"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", default=DEFAULT_CENTERS)
    parser.add_argument("--initial-gradients-tm", default=DEFAULT_GRADIENTS)
    parser.add_argument("--initial-length", type=float, default=480.0)
    parser.add_argument("--min-length", type=float, default=440.0)
    parser.add_argument("--max-length", type=float, default=560.0)
    parser.add_argument("--quad-length", type=float, default=4.0)
    parser.add_argument("--marker-step", type=float, default=4.0)
    parser.add_argument("--target-betx", type=float, default=80.0)
    parser.add_argument("--target-bety", type=float, default=30.0)
    parser.add_argument("--gradient-window-tm", type=float, default=80.0)
    parser.add_argument("--gradient-bound-tm", type=float, default=170.0)
    parser.add_argument("--emittance-nm", type=float, default=20.0)
    parser.add_argument("--output-prefix", default="upright_length_alpha_8q")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    centers = np.array(parse_float_list(args.centers), dtype=float)
    gradients = np.array(parse_float_list(args.initial_gradients_tm), dtype=float)
    if len(centers) != len(gradients):
        raise ValueError("centers and gradients must have the same length")

    def residual(values: np.ndarray) -> np.ndarray:
        length = float(values[0])
        trial_gradients = values[1:]
        if length < float(centers[-1]) + args.quad_length / 2.0:
            return np.ones(4) * 1.0e4
        try:
            result = optics(
                list(centers),
                list(trial_gradients),
                length,
                args.quad_length,
                sample_step=0.5,
            )
        except ValueError:
            return np.ones(4) * 1.0e4
        return np.array(
            [
                math.log(result["betx"] / args.target_betx) / 0.02,
                math.log(result["bety"] / args.target_bety) / 0.02,
                result["alfx"] / 0.35,
                result["alfy"] / 0.35,
            ]
        )

    lower_bounds = np.r_[
        args.min_length,
        np.maximum(-args.gradient_bound_tm, gradients - args.gradient_window_tm),
    ]
    upper_bounds = np.r_[
        args.max_length,
        np.minimum(args.gradient_bound_tm, gradients + args.gradient_window_tm),
    ]
    initial_values = np.r_[args.initial_length, gradients]
    result = least_squares(
        residual,
        initial_values,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=4000,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )

    length = float(result.x[0])
    matched_gradients = [float(value) for value in result.x[1:]]
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
        list(centers),
        matched_gradients,
    )
    with log_file.open("w", encoding="utf-8") as handle:
        subprocess.run(["./madx", str(madx_file)], cwd=base_dir, stdout=handle, stderr=subprocess.STDOUT, check=True)

    twiss = read_tfs(twiss_file)
    last = twiss.iloc[-1]
    sigma_x, sigma_y = beam_size_meters(twiss, args.emittance_nm)
    print(f"length_m: {length:.9g}")
    print(f"centers_m: {','.join(f'{value:.6g}' for value in centers)}")
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
