#!/usr/bin/env python3
"""Tune selected FACET quadrupoles to moderately raise local beta minima."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from optimize_facet_less_focusing import (
    FAMILIES,
    current_gradient_tm,
    warning_count,
    write_candidate,
)
from plot_twiss import read_tfs


def row_value(twiss, name: str, column: str) -> float:
    rows = twiss[twiss["NAME"].str.upper() == name.upper()]
    if rows.empty:
        raise ValueError(f"missing {name}")
    return float(rows.iloc[-1][column])


def metrics(tfs_file: Path, baseline_end: tuple[float, float]) -> dict[str, float]:
    twiss = read_tfs(tfs_file)
    last = twiss.iloc[-1]
    betx_dex20_11 = row_value(twiss, "DEX20_11", "BETX")
    bety_dex20_10 = row_value(twiss, "DEX20_10", "BETY")
    end_betx = float(last["BETX"])
    end_bety = float(last["BETY"])
    return {
        "betx_dex20_11": betx_dex20_11,
        "bety_dex20_10": bety_dex20_10,
        "end_betx": end_betx,
        "end_bety": end_bety,
        "end_alfx": float(last["ALFX"]),
        "end_alfy": float(last["ALFY"]),
        "endpoint_log_error": abs(math.log(end_betx / baseline_end[0]))
        + abs(math.log(end_bety / baseline_end[1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="local_beta_work")
    parser.add_argument("--target-beta", type=float, default=2.0)
    parser.add_argument("--soft-upper-beta", type=float, default=4.0)
    parser.add_argument("--endpoint-scale", type=float, default=0.18)
    parser.add_argument("--maxiter", type=int, default=32)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1312)
    parser.add_argument("--output-prefix", default="facet_local_beta_moderate")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    work_dir = base_dir / args.work_dir
    work_dir.mkdir(exist_ok=True)

    baseline = read_tfs(base_dir / "facet_twiss.tfs")
    baseline_end = (float(baseline.iloc[-1]["BETX"]), float(baseline.iloc[-1]["BETY"]))
    bounds = [(-cap, cap) for _, _, cap in FAMILIES]
    initial = np.array([current_gradient_tm(names[0]) for _, names, _ in FAMILIES], dtype=float)

    best: dict[str, float] | None = None
    best_gradients: np.ndarray | None = None
    counter = {"n": 0}

    def objective(values: np.ndarray) -> float:
        nonlocal best, best_gradients
        counter["n"] += 1
        try:
            twiss_file = work_dir / "facet_local_beta_candidate.tfs"
            madx_file = write_candidate(work_dir, values, twiss_file)
            log_file = work_dir / "facet_local_beta_candidate.log"
            with log_file.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    ["../madx", str(madx_file)],
                    cwd=base_dir,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = metrics(twiss_file, baseline_end)
            warnings = warning_count(log_file)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError, FloatingPointError):
            return 1.0e9

        # Aim for a moderate local beta increase, not a large blow-up.
        local = (math.log(result["betx_dex20_11"] / args.target_beta) / 0.22) ** 2
        local += (math.log(result["bety_dex20_10"] / args.target_beta) / 0.22) ** 2
        endpoint = (result["endpoint_log_error"] / args.endpoint_scale) ** 2
        upper = 0.0
        for value in (result["betx_dex20_11"], result["bety_dex20_10"]):
            excess = max(0.0, math.log(value / args.soft_upper_beta))
            upper += 80.0 * excess**2
        strength_move = 0.004 * float(np.mean(((values - initial) / np.array([cap for _, _, cap in FAMILIES])) ** 2))
        score = local + endpoint + upper + 100.0 * warnings + strength_move

        if best is None or score < best["objective"]:
            best = {"objective": score, **result}
            best_gradients = np.array(values, dtype=float)
            print(
                "best "
                f"eval={counter['n']} obj={score:.5g} "
                f"local=({result['betx_dex20_11']:.4g},{result['bety_dex20_10']:.4g}) "
                f"end=({result['end_betx']:.4g},{result['end_bety']:.4g}) "
                f"G={','.join(f'{value:.3g}' for value in values)}",
                flush=True,
            )
        return score

    objective(initial)
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
        method="Powell",
        bounds=bounds,
        options={"maxiter": 260, "xtol": 1.0e-3, "ftol": 1.0e-3},
    )

    if best is None or best_gradients is None:
        raise RuntimeError("no valid candidate found")

    output_prefix = base_dir / args.output_prefix
    output_twiss = output_prefix.with_name(output_prefix.name + "_twiss.tfs")
    output_madx = write_candidate(base_dir, best_gradients, output_twiss)
    output_madx.rename(output_prefix.with_suffix(".madx"))
    output_log = output_prefix.with_suffix(".log")
    with output_log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            ["../madx", str(output_prefix.with_suffix(".madx"))],
            cwd=base_dir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    final = metrics(output_twiss, baseline_end)
    print("\nFinal candidate")
    print(f"families: {','.join(name for name, _, _ in FAMILIES)}")
    print(f"gradients_tm: {','.join(f'{value:.6g}' for value in best_gradients)}")
    for key in ("betx_dex20_11", "bety_dex20_10", "end_betx", "end_bety", "end_alfx", "end_alfy", "endpoint_log_error"):
        print(f"{key}: {final[key]}")
    print(f"warnings: {warning_count(output_log)}")


if __name__ == "__main__":
    main()
