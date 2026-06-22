#!/usr/bin/env python3
"""Vary selected FACET quadrupole strengths to increase beta between Q0FF and Q0D."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from plot_twiss import read_tfs
from prepare_facet_madx_run import SOURCE, convert_lcavity


BRHO_TM = 10.0 / 0.299792458

FAMILIES: list[tuple[str, list[str], float]] = [
    ("Q1EL", ["Q1EL"], 72.627),
    ("Q1ER", ["Q1ER"], 72.459),
    ("Q2EL", ["Q2EL"], 39.000),
    ("Q2ER", ["Q2ER"], 39.000),
    ("Q3E", ["Q3EL_1", "Q3EL_2", "Q3ER_1", "Q3ER_2"], 45.225),
    ("Q4E", ["Q4EL_1", "Q4EL_2", "Q4EL_3", "Q4ER_1", "Q4ER_2", "Q4ER_3"], 45.225),
    ("Q5EL", ["Q5EL"], 29.412),
    ("Q5ER", ["Q5ER"], 29.412),
    ("Q6E", ["Q6E_1", "Q6E_2"], 94.867),
]


def warning_count(log_file: Path) -> int:
    text = log_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Number of warnings:\s+(\d+)", text)
    if match:
        return int(match.group(1))
    return text.count("++++++ warning:")


def base_run_text(twiss_file: Path) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    text = re.sub(
        r"(?ms)^(?P<name>[A-Za-z_][A-Za-z0-9_]*):\s*lcavity,\s*(?P<body>.*?);",
        convert_lcavity,
        text,
    )
    text = text.replace(
        "twiss, beta0 = initial;",
        'select, flag=twiss, column=name,keyword,s,l,k1l,tilt,betx,bety,alfx,alfy,dx,dy,mux,muy;\n'
        f'twiss, beta0 = initial, file="{twiss_file}";',
    )
    return text


def replace_k1(text: str, element_name: str, k1: float) -> str:
    pattern = re.compile(
        rf"(?ms)^({re.escape(element_name)}:\s*quadrupole,\s*)(?P<body>.*?);"
    )

    def repl(match: re.Match[str]) -> str:
        body = match.group("body")
        if re.search(r"\bk1\s*=", body, flags=re.IGNORECASE):
            body = re.sub(r"\bk1\s*=\s*[^,;]+", f"k1 = {k1:.12g}", body, count=1, flags=re.IGNORECASE)
        else:
            body = f"k1 = {k1:.12g}, " + body
        return f"{match.group(1)}{body};"

    new_text, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise ValueError(f"could not find quadrupole {element_name}")
    return new_text


def current_gradient_tm(element_name: str) -> float:
    text = SOURCE.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(?ms)^{re.escape(element_name)}:\s*quadrupole,\s*(?P<body>.*?);"
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"could not find quadrupole {element_name}")
    k1_match = re.search(r"\bk1\s*=\s*([^,;]+)", match.group("body"), flags=re.IGNORECASE)
    if not k1_match:
        return 0.0
    return float(k1_match.group(1)) * BRHO_TM


def write_candidate(work_dir: Path, gradients_tm: np.ndarray, twiss_file: Path) -> Path:
    text = base_run_text(twiss_file.resolve())
    for gradient, (_, element_names, _) in zip(gradients_tm, FAMILIES):
        k1 = float(gradient) / BRHO_TM
        for element_name in element_names:
            text = replace_k1(text, element_name, k1)
    path = work_dir / "facet_less_focusing_candidate.madx"
    path.write_text(text, encoding="utf-8")
    return path


def metrics(tfs_file: Path, baseline_end: tuple[float, float]) -> dict[str, float]:
    twiss = read_tfs(tfs_file)
    q0ff_rows = twiss[twiss["NAME"].str.upper() == "Q0FF"]
    q0d_rows = twiss[twiss["NAME"].str.upper() == "Q0D"]
    if q0ff_rows.empty or q0d_rows.empty:
        raise ValueError("missing Q0FF or Q0D in TWISS")
    q0ff_s = float(q0ff_rows.iloc[-1]["S"])
    q0d_s = float(q0d_rows.iloc[-1]["S"])
    region = twiss[(twiss["S"] >= q0ff_s) & (twiss["S"] <= q0d_s)]
    last = twiss.iloc[-1]
    end_betx = float(last["BETX"])
    end_bety = float(last["BETY"])
    mean_betx = float(region["BETX"].mean())
    mean_bety = float(region["BETY"].mean())
    max_betx = float(region["BETX"].max())
    max_bety = float(region["BETY"].max())
    return {
        "end_betx": end_betx,
        "end_bety": end_bety,
        "end_alfx": float(last["ALFX"]),
        "end_alfy": float(last["ALFY"]),
        "mean_betx": mean_betx,
        "mean_bety": mean_bety,
        "max_betx": max_betx,
        "max_bety": max_bety,
        "region_score": math.sqrt(max(mean_betx, 1.0e-9) * max(mean_bety, 1.0e-9)),
        "endpoint_log_error": abs(math.log(end_betx / baseline_end[0])) + abs(math.log(end_bety / baseline_end[1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="less_focusing_work")
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--popsize", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1205)
    parser.add_argument("--endpoint-weight", type=float, default=2.5)
    parser.add_argument("--endpoint-soft-limit", type=float, default=1.2)
    parser.add_argument("--output-prefix", default="facet_less_focusing")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    work_dir = base_dir / args.work_dir
    work_dir.mkdir(exist_ok=True)

    baseline_twiss = read_tfs(base_dir / "facet_twiss.tfs")
    baseline_end = (float(baseline_twiss.iloc[-1]["BETX"]), float(baseline_twiss.iloc[-1]["BETY"]))
    bounds = [(-cap, cap) for _, _, cap in FAMILIES]

    best: dict[str, float] | None = None
    best_gradients: np.ndarray | None = None
    counter = {"n": 0}

    def objective(values: np.ndarray) -> float:
        counter["n"] += 1
        nonlocal best, best_gradients
        try:
            twiss_file = work_dir / "facet_less_focusing_candidate.tfs"
            madx_file = write_candidate(work_dir, values, twiss_file)
            log_file = work_dir / "facet_less_focusing_candidate.log"
            with log_file.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    ["../madx", str(madx_file)],
                    cwd=base_dir,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            warning_penalty = warning_count(log_file) * 100.0
            result = metrics(twiss_file, baseline_end)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError, FloatingPointError):
            return 1.0e9
        # Maximize region beta while softly preserving endpoint betas.
        score = -math.log(result["region_score"])
        score += args.endpoint_weight * result["endpoint_log_error"] ** 2
        endpoint_excess = max(0.0, result["endpoint_log_error"] - args.endpoint_soft_limit)
        score += 120.0 * endpoint_excess**2
        score += warning_penalty
        if best is None or score < best["objective"]:
            best = {"objective": score, **result}
            best_gradients = np.array(values, dtype=float)
            print(
                "best "
                f"eval={counter['n']} obj={score:.5g} "
                f"region_mean=({result['mean_betx']:.4g},{result['mean_bety']:.4g}) "
                f"region_max=({result['max_betx']:.4g},{result['max_bety']:.4g}) "
                f"end=({result['end_betx']:.4g},{result['end_bety']:.4g}) "
                f"G={','.join(f'{v:.3g}' for v in values)}",
                flush=True,
            )
        return score

    initial_gradients = np.array([current_gradient_tm(names[0]) for _, names, _ in FAMILIES], dtype=float)
    objective(initial_gradients)

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
        options={"maxiter": 280, "xtol": 1.0e-3, "ftol": 1.0e-3},
    )
    if best is None or best_gradients is None:
        raise RuntimeError("no valid candidate found")

    output_prefix = base_dir / args.output_prefix
    final_twiss = output_prefix.with_name(output_prefix.name + "_twiss.tfs")
    final_madx = write_candidate(base_dir, best_gradients, final_twiss)
    final_madx.rename(output_prefix.with_suffix(".madx"))
    log_file = output_prefix.with_suffix(".log")
    with log_file.open("w", encoding="utf-8") as handle:
        subprocess.run(
            ["../madx", str(output_prefix.with_suffix(".madx"))],
            cwd=base_dir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    final_metrics = metrics(final_twiss, baseline_end)
    print("\nFinal candidate")
    print(f"families: {','.join(name for name, _, _ in FAMILIES)}")
    print(f"gradients_tm: {','.join(f'{value:.6g}' for value in best_gradients)}")
    for key in (
        "mean_betx",
        "mean_bety",
        "max_betx",
        "max_bety",
        "end_betx",
        "end_bety",
        "end_alfx",
        "end_alfy",
        "endpoint_log_error",
    ):
        print(f"{key}: {final_metrics[key]}")
    print(f"warnings: {warning_count(log_file)}")


if __name__ == "__main__":
    main()
