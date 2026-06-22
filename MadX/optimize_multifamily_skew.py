#!/usr/bin/env python3
"""Search fixed solenoid and multi-family skew settings around MAD-X."""

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


TARGET_BETX = 80.0
TARGET_BETY = 30.0
BASELINE_SOLENOID_T = -2.097259868973885
BASELINE_SKEW_GRADIENTS_TM = "-105.187,51.865,50.6233"
BASELINE_NORMAL_GRADIENT_TM = 79.12910940962888


def parse_positions(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def warning_count(log_file: Path) -> int:
    text = log_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Number of warnings:\s+(\d+)", text)
    if match:
        return int(match.group(1))
    return text.count("++++++ warning:")


def run_candidate(
    work_dir: Path,
    variables: np.ndarray,
    args: argparse.Namespace,
    serial: int,
) -> dict[str, float | str | Path]:
    solenoid_t = float(variables[0])
    skew_gradients = [float(value) for value in variables[1 : 1 + args.skew_families]]
    normal_start = 1 + args.skew_families
    normal_gradients = [float(value) for value in variables[normal_start : normal_start + args.normal_families]]

    madx_file = work_dir / "candidate.madx"
    log_file = work_dir / "candidate.log"
    twiss_file = work_dir / "candidate.tfs"

    generator_cmd = [
        "python3",
        "generate_flatbeam_lattice.py",
        "--length",
        str(args.length),
        "--solenoid-length",
        str(args.solenoid_length),
        "--solenoid-module-length",
        str(args.solenoid_module_length),
        "--solenoid-gap",
        str(args.solenoid_gap),
        "--distributed-skew-positions",
        args.skew_positions,
        "--distributed-skew-families",
        str(args.skew_families),
        f"--distributed-skew-gradients-tm={','.join(f'{value:.12g}' for value in skew_gradients)}",
        f"--initial-solenoid-t={solenoid_t:.12g}",
        "--normal-mode",
        "fixed",
        f"--normal-gradient-tm={normal_gradients[0]:.12g}",
        f"--normal-gradients-tm={','.join(f'{value:.12g}' for value in normal_gradients)}",
        "--normal-families",
        str(args.normal_families),
        "--normal-positions",
        args.normal_positions,
        "--target-betx",
        str(args.target_betx),
        "--target-bety",
        str(args.target_bety),
        "--match-strategy",
        "none",
        "--twiss-file",
        str(twiss_file),
        "--output",
        str(madx_file),
    ]
    subprocess.run(generator_cmd, cwd=args.base_dir, check=True, stdout=subprocess.DEVNULL)
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
    max_beta = max(max_betx, max_bety)
    eps_m = args.emittance_nm * 1.0e-9
    sigma_x = math.sqrt(max_betx * eps_m)
    sigma_y = math.sqrt(max_bety * eps_m)
    max_sigma = max(sigma_x, sigma_y)
    warnings = warning_count(log_file)

    if not all(math.isfinite(value) for value in (betx, bety, alfx, alfy, max_beta, max_sigma)):
        raise ValueError("Non-finite optics result")

    endpoint = (math.log(betx / args.target_betx) / args.endpoint_log_scale) ** 2
    endpoint += (math.log(bety / args.target_bety) / args.endpoint_log_scale) ** 2
    alpha = args.alpha_weight * (alfx**2 + alfy**2)
    beam_size = args.beam_size_weight * (math.log(max_sigma / args.beam_size_goal_m) ** 2)
    peak_beta = 0.0
    if max_beta > args.peak_beta_cap:
        peak_beta = args.peak_weight * (math.log(max_beta / args.peak_beta_cap) ** 2)
    warning_penalty = args.warning_weight * warnings
    objective = endpoint + alpha + beam_size + peak_beta + warning_penalty

    if args.keep_every > 0 and serial % args.keep_every == 0:
        shutil.copyfile(madx_file, work_dir / f"candidate_{serial:05d}.madx")
        shutil.copyfile(log_file, work_dir / f"candidate_{serial:05d}.log")
        shutil.copyfile(twiss_file, work_dir / f"candidate_{serial:05d}.tfs")

    return {
        "objective": objective,
        "solenoid_t": solenoid_t,
        "normal_gradients_tm": ",".join(f"{value:.6g}" for value in normal_gradients),
        "skew_gradients_tm": ",".join(f"{value:.6g}" for value in skew_gradients),
        "betx": betx,
        "bety": bety,
        "alfx": alfx,
        "alfy": alfy,
        "max_betx": max_betx,
        "max_bety": max_bety,
        "max_beta": max_beta,
        "max_sigma_x_m": sigma_x,
        "max_sigma_y_m": sigma_y,
        "max_sigma_m": max_sigma,
        "warnings": warnings,
        "madx_file": madx_file,
        "log_file": log_file,
        "twiss_file": twiss_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=900.0)
    parser.add_argument("--solenoid-length", type=float, default=60.0)
    parser.add_argument("--solenoid-module-length", type=float, default=8.0)
    parser.add_argument("--solenoid-gap", type=float, default=4.0)
    parser.add_argument("--skew-positions", default="460,500,540,580,620,660")
    parser.add_argument("--skew-families", type=int, default=3)
    parser.add_argument("--initial-solenoid-t", type=float, default=BASELINE_SOLENOID_T)
    parser.add_argument("--initial-skew-gradients-tm", default=BASELINE_SKEW_GRADIENTS_TM)
    parser.add_argument("--initial-normal-gradients-tm", default="")
    parser.add_argument(
        "--normal-positions",
        default="64,84,104,124,144,164,184,204,224,244,264,284,304,324,344,364,384,404,424,444,464,484,504,524,544,564,584,604,624,644,664,684",
    )
    parser.add_argument("--normal-families", type=int, default=1)
    parser.add_argument("--target-betx", type=float, default=TARGET_BETX)
    parser.add_argument("--target-bety", type=float, default=TARGET_BETY)
    parser.add_argument("--solenoid-bound-t", type=float, default=8.0)
    parser.add_argument("--skew-bound-tm", type=float, default=160.0)
    parser.add_argument("--normal-min-tm", type=float, default=40.0)
    parser.add_argument("--normal-max-tm", type=float, default=120.0)
    parser.add_argument("--emittance-nm", type=float, default=20.0)
    parser.add_argument("--beam-size-goal-m", type=float, default=0.02)
    parser.add_argument("--beam-size-weight", type=float, default=120.0)
    parser.add_argument("--peak-beta-cap", type=float, default=5000.0)
    parser.add_argument("--endpoint-log-scale", type=float, default=0.06)
    parser.add_argument("--alpha-weight", type=float, default=0.02)
    parser.add_argument("--peak-weight", type=float, default=12.0)
    parser.add_argument("--warning-weight", type=float, default=150.0)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--local-maxiter", type=int, default=120)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep-every", type=int, default=0)
    parser.add_argument("--work-dir", default="multifamily_optimizer_work")
    args = parser.parse_args()
    args.base_dir = Path(__file__).resolve().parent

    if len(parse_positions(args.skew_positions)) < args.skew_families:
        raise ValueError("Need at least as many skew positions as skew families")
    if args.normal_families < 1:
        raise ValueError("--normal-families must be at least 1")

    work_dir = (args.base_dir / args.work_dir).resolve()
    work_dir.mkdir(exist_ok=True)

    bounds = [(-args.solenoid_bound_t, args.solenoid_bound_t)]
    bounds.extend([(-args.skew_bound_tm, args.skew_bound_tm)] * args.skew_families)
    bounds.extend([(args.normal_min_tm, args.normal_max_tm)] * args.normal_families)
    initial_skew = parse_float_list(args.initial_skew_gradients_tm)
    if len(initial_skew) != args.skew_families:
        raise ValueError("--initial-skew-gradients-tm must match --skew-families")
    if args.initial_normal_gradients_tm:
        initial_normal = parse_float_list(args.initial_normal_gradients_tm)
    else:
        initial_normal = [BASELINE_NORMAL_GRADIENT_TM] * args.normal_families
    if len(initial_normal) != args.normal_families:
        raise ValueError("--initial-normal-gradients-tm must match --normal-families")
    initial_values = np.array([args.initial_solenoid_t, *initial_skew, *initial_normal], dtype=float)

    counter = {"n": 0}
    best: dict[str, float | str | Path] | None = None

    def objective(values: np.ndarray) -> float:
        counter["n"] += 1
        nonlocal best
        try:
            result = run_candidate(work_dir, values, args, counter["n"])
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            return 1.0e12
        if best is None or float(result["objective"]) < float(best["objective"]):
            best = result
            print(
                "best "
                f"eval={counter['n']} obj={result['objective']:.6g} "
                f"B={result['solenoid_t']:.4g}T "
                f"skew={result['skew_gradients_tm']}T/m "
                f"normal={result['normal_gradients_tm']}T/m "
                f"end=({result['betx']:.4g},{result['bety']:.4g}) "
                f"alpha=({result['alfx']:.3g},{result['alfy']:.3g}) "
                f"max_beta=({result['max_betx']:.4g},{result['max_bety']:.4g}) "
                f"max_sigma=({result['max_sigma_x_m']:.4g},{result['max_sigma_y_m']:.4g})m "
                f"warnings={result['warnings']}",
                flush=True,
            )
            shutil.copyfile(result["madx_file"], work_dir / "best.madx")
            shutil.copyfile(result["log_file"], work_dir / "best.log")
            shutil.copyfile(result["twiss_file"], work_dir / "best.tfs")
        return float(result["objective"])

    minimize(
        objective,
        initial_values,
        method="Nelder-Mead",
        options={"maxiter": args.local_maxiter, "xatol": 1.0e-3, "fatol": 1.0e-3},
    )
    if not args.local_only:
        de_result = differential_evolution(
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
            de_result.x,
            method="Nelder-Mead",
            options={"maxiter": args.local_maxiter, "xatol": 1.0e-3, "fatol": 1.0e-3},
        )
    if best is None:
        raise RuntimeError("No valid candidate found")

    print("\nFinal best")
    for key in (
        "objective",
        "solenoid_t",
        "skew_gradients_tm",
        "normal_gradients_tm",
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
