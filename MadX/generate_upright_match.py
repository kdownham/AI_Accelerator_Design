#!/usr/bin/env python3
"""Generate a MAD-X match deck for upright-quadrupole-only optics."""

from __future__ import annotations

import argparse
from pathlib import Path


PC_GEV = 5000.0
BRHO_TM = PC_GEV / 0.299792458
INITIAL_BETA = 0.2351


def parse_positions(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def default_positions(length: float, n_quads: int, margin: float, step: float) -> list[float]:
    if n_quads == 1:
        return [round((length / 2.0) / step) * step]
    span = length - 2.0 * margin - step
    return [round((margin + index * span / (n_quads - 1)) / step) * step for index in range(n_quads)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-quads", type=int, required=True)
    parser.add_argument("--length", type=float, default=480.0)
    parser.add_argument("--step", type=float, default=4.0)
    parser.add_argument("--margin", type=float, default=20.0)
    parser.add_argument("--positions", default="")
    parser.add_argument("--gradient-bound-tm", type=float, default=180.0)
    parser.add_argument("--target-betx", type=float, default=80.0)
    parser.add_argument("--target-bety", type=float, default=30.0)
    parser.add_argument("--initial-gradients-tm", default="")
    parser.add_argument("--twiss-file", default="upright_match_twiss.tfs")
    parser.add_argument("--output", default="upright_match.madx")
    args = parser.parse_args()

    positions = parse_positions(args.positions) if args.positions else default_positions(args.length, args.n_quads, args.margin, args.step)
    if len(positions) != args.n_quads:
        raise ValueError("positions count must match n-quads")
    initial = parse_positions(args.initial_gradients_tm) if args.initial_gradients_tm else []
    if initial and len(initial) != args.n_quads:
        raise ValueError("initial gradients count must match n-quads")
    if not initial:
        initial = [80.0 if index % 2 == 0 else -80.0 for index in range(args.n_quads)]

    gradient_by_position = dict(zip(positions, initial))
    grid = [i * args.step for i in range(int(args.length / args.step) + 1)]
    max_k1 = args.gradient_bound_tm / BRHO_TM
    lines = [
        'title, "5 TeV electron upright-only matched lattice";',
        "",
        "beam, particle=electron, energy=5000;",
        "",
        f"! Brho = {BRHO_TM:.9f} T*m",
    ]
    for index, gradient in enumerate(initial, start=1):
        lines.append(f"kq{index:03d} = {gradient / BRHO_TM:.12g};")
    lines.extend(["", f"upright: sequence, l={args.length:.6f};"])
    quad_index = 0
    for index, start in enumerate(grid):
        lines.append(f"  mk{index:03d}: marker, at={start:.6f};")
        if index + 1 >= len(grid):
            continue
        center = start + args.step / 2.0
        if start in gradient_by_position:
            quad_index += 1
            lines.append(f"  q{quad_index:03d}: quadrupole, l={args.step:.6f}, k1:=kq{quad_index:03d}, at={center:.6f};")
        else:
            lines.append(f"  dr{index:03d}: drift, l={args.step:.6f}, at={center:.6f};")
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
    for index in range(1, args.n_quads + 1):
        lines.append(f"  vary, name=kq{index:03d}, step=1.0e-5, lower={-max_k1:.12g}, upper={max_k1:.12g};")
    lines.extend(
        [
            f"  constraint, range=#e, betx={args.target_betx:.12g};",
            f"  constraint, range=#e, bety={args.target_bety:.12g};",
            "  constraint, range=#e, alfx=0;",
            "  constraint, range=#e, alfy=0;",
            "  lmdif, calls=8000, tolerance=1.0e-12;",
            "endmatch;",
            "",
        ]
    )
    for index in range(1, args.n_quads + 1):
        lines.append(f"value, kq{index:03d};")
    lines.extend(
        [
            "",
            "twiss,",
            f"  betx={INITIAL_BETA:.12g}, alfx=0,",
            f"  bety={INITIAL_BETA:.12g}, alfy=0,",
            "  dx=0, dpx=0,",
            "  dy=0, dpy=0,",
            f'  file="{args.twiss_file}";',
            "",
            "stop;",
        ]
    )
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
