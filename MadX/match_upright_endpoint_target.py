#!/usr/bin/env python3
"""Rematch the upright keeper endpoint with MAD-X MATCH using bounded quadrupole knobs."""

from __future__ import annotations

import argparse
import itertools
import math
import re
import shutil
import subprocess
from pathlib import Path

from optimize_upright_positions import BRHO_TM, generate_lattice, warning_count
from plot_twiss import beam_size_meters, read_tfs


INITIAL_BETA = 0.2351


def parse_quads(source: Path) -> tuple[float, list[float], list[float]]:
    text = source.read_text(encoding="utf-8")
    length_match = re.search(r"upright:\s*sequence,\s*l=([^;]+);", text, flags=re.IGNORECASE)
    if not length_match:
        raise ValueError(f"could not find sequence length in {source}")
    length = float(length_match.group(1))
    centers: list[float] = []
    gradients: list[float] = []
    for match in re.finditer(
        r"q\d+:\s*quadrupole,\s*l=[^,]+,\s*k1=([^,]+),\s*at=([^;]+);",
        text,
        flags=re.IGNORECASE,
    ):
        gradients.append(float(match.group(1)) * BRHO_TM)
        centers.append(float(match.group(2)))
    if not centers:
        raise ValueError(f"could not find quadrupoles in {source}")
    return length, centers, gradients


def build_match_deck(
    path: Path,
    twiss_file: Path,
    length: float,
    centers: list[float],
    gradients_tm: list[float],
    vary_indices: tuple[int, ...],
    gradient_bound_tm: float,
    target_betx: float,
    target_bety: float,
) -> None:
    knob_by_quad = {index: f"kq{index + 1:03d}" for index in vary_indices}
    elements: list[tuple[float, str]] = []
    marker_step = 4.0
    quad_length = 4.0
    half_length = quad_length / 2.0
    for index in range(int(length / marker_step) + 1):
        s = index * marker_step
        if any((center - half_length) < s < (center + half_length) for center in centers):
            continue
        elements.append((s, f"  mk{index:03d}: marker, at={s:.6f};"))
    for index, (center, gradient_tm) in enumerate(zip(centers, gradients_tm)):
        if index in knob_by_quad:
            k1_text = f"k1:={knob_by_quad[index]}"
        else:
            k1_text = f"k1={gradient_tm / BRHO_TM:.12g}"
        elements.append((center, f"  q{index + 1:03d}: quadrupole, l={quad_length:.6f}, {k1_text}, at={center:.6f};"))
    elements.sort(key=lambda item: item[0])

    lines = [
        'title, "5 TeV electron upright endpoint retarget with MAD-X MATCH";',
        "",
        "beam, particle=electron, energy=5000;",
        "",
        f"! Brho = {BRHO_TM:.9f} T*m",
    ]
    for index in vary_indices:
        lines.append(f"{knob_by_quad[index]} = {gradients_tm[index] / BRHO_TM:.12g};")
    lines.extend(["", f"upright: sequence, l={length:.6f};"])
    lines.extend(line for _, line in elements)
    lines.extend(
        [
            "endsequence;",
            "",
            "use, sequence=upright;",
            "",
            "match, sequence=upright,",
            f"  betx={INITIAL_BETA:.12g}, alfx=0,",
            f"  bety={INITIAL_BETA:.12g}, alfy=0,",
            "  dx=0, dpx=0,",
            "  dy=0, dpy=0;",
        ]
    )
    max_k1 = gradient_bound_tm / BRHO_TM
    for index in vary_indices:
        lines.append(f"  vary, name={knob_by_quad[index]}, step=1.0e-5, lower={-max_k1:.12g}, upper={max_k1:.12g};")
    lines.extend(
        [
            f"  constraint, range=#e, betx={target_betx:.15g};",
            f"  constraint, range=#e, bety={target_bety:.15g};",
            "  constraint, range=#e, alfx=0;",
            "  constraint, range=#e, alfy=0;",
            "  lmdif, calls=8000, tolerance=1.0e-12;",
            "endmatch;",
            "",
        ]
    )
    for index in vary_indices:
        lines.append(f"value, {knob_by_quad[index]};")
    lines.extend(
        [
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


def score_twiss(twiss_file: Path, target_betx: float, target_bety: float) -> dict[str, float]:
    twiss = read_tfs(twiss_file)
    last = twiss.iloc[-1]
    sigma_x, sigma_y = beam_size_meters(twiss, 20.0)
    return {
        "betx": float(last["BETX"]),
        "bety": float(last["BETY"]),
        "alfx": float(last["ALFX"]),
        "alfy": float(last["ALFY"]),
        "max_betx": float(twiss["BETX"].max()),
        "max_bety": float(twiss["BETY"].max()),
        "max_sigma_x_m": float(sigma_x.max()),
        "max_sigma_y_m": float(sigma_y.max()),
        "endpoint_error": abs(math.log(float(last["BETX"]) / target_betx))
        + abs(math.log(float(last["BETY"]) / target_bety))
        + abs(float(last["ALFX"]))
        + abs(float(last["ALFY"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="upright_peak_reduced_10q_matched.madx")
    parser.add_argument("--target-betx", type=float, required=True)
    parser.add_argument("--target-bety", type=float, required=True)
    parser.add_argument("--gradient-bound-tm", type=float, default=160.0)
    parser.add_argument("--output-prefix", default="upright_endpoint_98p808_26p776")
    parser.add_argument("--work-dir", default="upright_endpoint_match_work")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    source = base_dir / args.source
    length, centers, gradients_tm = parse_quads(source)
    work_dir = base_dir / args.work_dir
    work_dir.mkdir(exist_ok=True)

    best: tuple[float, tuple[int, ...], dict[str, float], Path, Path, Path] | None = None
    for vary_indices in itertools.combinations(range(len(centers)), 4):
        label = "_".join(f"q{index + 1:03d}" for index in vary_indices)
        madx_file = work_dir / f"match_{label}.madx"
        twiss_file = work_dir / f"match_{label}.tfs"
        log_file = work_dir / f"match_{label}.log"
        build_match_deck(
            madx_file,
            twiss_file,
            length,
            centers,
            gradients_tm,
            vary_indices,
            args.gradient_bound_tm,
            args.target_betx,
            args.target_bety,
        )
        with log_file.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(["./madx", str(madx_file)], cwd=base_dir, stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not twiss_file.exists() or warning_count(log_file) > 0:
            continue
        metrics = score_twiss(twiss_file, args.target_betx, args.target_bety)
        score = metrics["endpoint_error"]
        if best is None or score < best[0]:
            best = (score, vary_indices, metrics, madx_file, log_file, twiss_file)
            print(
                "best "
                f"score={score:.6g} knobs={label} "
                f"end=({metrics['betx']:.9g},{metrics['bety']:.9g}) "
                f"alpha=({metrics['alfx']:.3g},{metrics['alfy']:.3g}) "
                f"max_sigma=({metrics['max_sigma_x_m']:.6g},{metrics['max_sigma_y_m']:.6g})",
                flush=True,
            )
    if best is None:
        raise RuntimeError("no valid MAD-X MATCH candidate found")

    _, vary_indices, metrics, madx_file, log_file, twiss_file = best
    prefix = base_dir / args.output_prefix
    shutil.copyfile(madx_file, prefix.with_name(prefix.name + "_madx_match.madx"))
    shutil.copyfile(log_file, prefix.with_name(prefix.name + "_madx_match.log"))
    shutil.copyfile(twiss_file, prefix.with_name(prefix.name + "_madx_match_twiss.tfs"))

    matched_twiss = read_tfs(twiss_file)
    matched_gradients = list(gradients_tm)
    for index in vary_indices:
        row = matched_twiss[matched_twiss["NAME"].str.upper() == f"Q{index + 1:03d}"].iloc[-1]
        matched_gradients[index] = float(row["K1L"]) / float(row["L"]) * BRHO_TM

    result_madx = prefix.with_name(prefix.name + "_madx_match_result.madx")
    result_twiss = prefix.with_name(prefix.name + "_madx_match_result_twiss.tfs")
    result_log = prefix.with_name(prefix.name + "_madx_match_result.log")
    generate_lattice(result_madx, result_twiss, length, 4.0, 4.0, centers, matched_gradients)
    with result_log.open("w", encoding="utf-8") as handle:
        subprocess.run(["./madx", str(result_madx)], cwd=base_dir, stdout=handle, stderr=subprocess.STDOUT, check=True)

    final = score_twiss(result_twiss, args.target_betx, args.target_bety)
    print("\nFinal static result")
    print(f"knobs: {','.join(f'q{index + 1:03d}' for index in vary_indices)}")
    print(f"length_m: {length:.9g}")
    print(f"centers_m: {','.join(f'{value:.6g}' for value in centers)}")
    print(f"gradients_tm: {','.join(f'{value:.6g}' for value in matched_gradients)}")
    for key, value in final.items():
        print(f"{key}: {value}")
    print(f"warnings: {warning_count(result_log)}")


if __name__ == "__main__":
    main()
