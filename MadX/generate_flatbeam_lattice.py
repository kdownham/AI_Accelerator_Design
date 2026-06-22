#!/usr/bin/env python3
"""Generate a constrained MAD-X flat-beam transformer lattice."""

from __future__ import annotations

import argparse
from pathlib import Path


PC_GEV = 5000.0
BRHO_TM = PC_GEV / 0.299792458

# Practical-ish field limits for a first-pass conceptual layout.
MAX_SOLENOID_T = 10.0
MAX_QUAD_GRADIENT_TM = 200.0

MAX_KSOL = MAX_SOLENOID_T / BRHO_TM
MAX_K1 = MAX_QUAD_GRADIENT_TM / BRHO_TM

def element_between(
    start: float,
    end: float,
    index: int,
    solenoid_length: float,
    solenoid_module_length: float,
    solenoid_gap: float,
    skq_positions: dict[float, str],
    distributed_skew_positions: dict[float, tuple[str, str]],
    normal_quad_positions: dict[float, tuple[str, str]],
) -> str:
    center = (start + end) / 2.0
    length = end - start
    name = skq_positions.get(start)
    distributed_skew_name = distributed_skew_positions.get(start)
    normal_name = normal_quad_positions.get(start)
    if center <= solenoid_length:
        if solenoid_module_length > 0:
            period = solenoid_module_length + solenoid_gap
            offset = start % period
            if offset >= solenoid_module_length:
                return f"  ds{index:03d}: drift, l={length:.6f}, at={center:.6f};"
        return f"  sol{index:03d}: solenoid, l={length:.6f}, ks:=ksol, at={center:.6f};"
    if name == "skq1":
        return f"  skq1: quadrupole, l={length:.6f}, k1:=kq1, tilt=pi/4, at={center:.6f};"
    if name == "skq2":
        return f"  skq2: quadrupole, l={length:.6f}, k1:=kq2, tilt=pi/4, at={center:.6f};"
    if name == "skq3":
        return f"  skq3: quadrupole, l={length:.6f}, k1:=kq1, tilt=pi/4, at={center:.6f};"
    if distributed_skew_name:
        element_name, family = distributed_skew_name
        return f"  {element_name}: quadrupole, l={length:.6f}, k1:={family}, tilt=pi/4, at={center:.6f};"
    if normal_name:
        element_name, kexpr = normal_name
        return f"  {element_name}: quadrupole, l={length:.6f}, k1:={kexpr}, at={center:.6f};"
    return f"  dr{index:03d}: drift, l={length:.6f}, at={center:.6f};"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=360.0)
    parser.add_argument("--step", type=float, default=4.0)
    parser.add_argument("--solenoid-length", type=float, default=80.0)
    parser.add_argument("--solenoid-module-length", type=float, default=0.0)
    parser.add_argument("--solenoid-gap", type=float, default=0.0)
    parser.add_argument("--skq1", type=float, default=80.0)
    parser.add_argument("--skq2", type=float, default=140.0)
    parser.add_argument("--skq3", type=float, default=200.0)
    parser.add_argument("--distributed-skew-positions", default="")
    parser.add_argument("--distributed-skew-gradient-tm", type=float, default=50.0)
    parser.add_argument("--distributed-skew-gradients-tm", default="")
    parser.add_argument("--distributed-skew-families", type=int, default=1)
    parser.add_argument("--initial-solenoid-t", type=float, default=8.0)
    parser.add_argument("--skq1-gradient-tm", type=float, default=100.0)
    parser.add_argument("--skq2-gradient-tm", type=float, default=-160.0)
    parser.add_argument("--target-betx", type=float, default=80.0)
    parser.add_argument("--target-bety", type=float, default=30.0)
    parser.add_argument("--mid-s", type=float, default=180.0)
    parser.add_argument("--mid-betx", type=float, default=400.0)
    parser.add_argument("--mid-bety", type=float, default=400.0)
    parser.add_argument("--normal-mode", choices=("none", "fixed", "match", "match-endpoint"), default="none")
    parser.add_argument("--normal-gradient-tm", type=float, default=50.0)
    parser.add_argument("--normal-gradients-tm", default="")
    parser.add_argument("--normal-families", type=int, default=1)
    parser.add_argument("--normal-positions", default="100,140,220,260")
    parser.add_argument("--match-strategy", choices=("madx", "none"), default="madx")
    parser.add_argument("--twiss-file", default="flatbeam_twiss.tfs")
    parser.add_argument("--output", default="flatbeam_triplet.madx")
    args = parser.parse_args()

    skq_positions = {args.skq1: "skq1", args.skq2: "skq2", args.skq3: "skq3"}
    distributed_skew_positions = {}
    if args.distributed_skew_positions:
        skq_positions = {}
        skew_positions = [float(value) for value in args.distributed_skew_positions.split(",") if value]
        if args.distributed_skew_families < 1:
            raise ValueError("--distributed-skew-families must be at least 1")
        distributed_skew_positions = {
            position: (f"sk{index + 1:03d}", f"ksk{index % args.distributed_skew_families + 1}")
            for index, position in enumerate(skew_positions)
        }
    normal_quad_positions = {}
    if args.normal_mode != "none":
        if args.normal_families < 1:
            raise ValueError("--normal-families must be at least 1")
        normal_positions = [float(value) for value in args.normal_positions.split(",") if value]
        normal_quad_positions = {
            position: (
                f"nq{index + 1:03d}",
                f"{'' if index % 2 == 0 else '-'}{'knch' if args.normal_families == 1 else f'knch{index % args.normal_families + 1}'}",
            )
            for index, position in enumerate(normal_positions)
        }
    mid_marker = f"mk{int(round(args.mid_s / args.step)):03d}"
    late_marker = f"mk{int(round((args.mid_s + 60.0) / args.step)):03d}"
    positions = [i * args.step for i in range(int(args.length / args.step) + 1)]
    lines = [
        'title, "5 TeV electron constrained round-to-flat beam transformer";',
        "",
        "beam, particle=electron, energy=5000;",
        "",
        f"! Brho = {BRHO_TM:.9f} T*m",
        f"! Solenoid cap: {MAX_SOLENOID_T:.3f} T => |KS| <= {MAX_KSOL:.12g} 1/m",
        f"! Skew quadrupole cap: {MAX_QUAD_GRADIENT_TM:.3f} T/m => |K1| <= {MAX_K1:.12g} 1/m^2",
        "",
        f"ksol = {args.initial_solenoid_t / BRHO_TM:.12g};",
        f"kq1 = {args.skq1_gradient_tm / BRHO_TM:.12g};",
        f"kq2 = {args.skq2_gradient_tm / BRHO_TM:.12g};",
    ]
    lines.append(f"knch = {args.normal_gradient_tm / BRHO_TM:.12g};")
    normal_gradient_seeds = []
    if args.normal_gradients_tm:
        normal_gradient_seeds = [float(value) for value in args.normal_gradients_tm.split(",") if value]
    for family in range(1, args.normal_families + 1):
        if family <= len(normal_gradient_seeds):
            gradient_tm = normal_gradient_seeds[family - 1]
        else:
            gradient_tm = args.normal_gradient_tm
        lines.append(f"knch{family} = {gradient_tm / BRHO_TM:.12g};")
    skew_gradient_seeds = []
    if args.distributed_skew_gradients_tm:
        skew_gradient_seeds = [float(value) for value in args.distributed_skew_gradients_tm.split(",") if value]
    for family in range(1, args.distributed_skew_families + 1):
        if family <= len(skew_gradient_seeds):
            gradient_tm = skew_gradient_seeds[family - 1]
        else:
            sign = 1.0 if family % 2 else -1.0
            gradient_tm = sign * args.distributed_skew_gradient_tm
        lines.append(f"ksk{family} = {gradient_tm / BRHO_TM:.12g};")
    lines.extend(
        [
            "",
            f"flatbeam: sequence, l={args.length:.6f};",
        ]
    )

    for index, start in enumerate(positions):
        lines.append(f"  mk{index:03d}: marker, at={start:.6f};")
        if index + 1 < len(positions):
            lines.append(
                element_between(
                    start,
                    positions[index + 1],
                    index,
                    args.solenoid_length,
                    args.solenoid_module_length,
                    args.solenoid_gap,
                    skq_positions,
                    distributed_skew_positions,
                    normal_quad_positions,
                )
            )

    lines.extend(
        [
            "endsequence;",
            "",
            "use, sequence=flatbeam;",
            "",
        ]
    )
    if args.match_strategy == "madx":
        lines.extend(
            [
                "match, sequence=flatbeam,",
                "  betx=0.2351, alfx=0,",
                "  bety=0.2351, alfy=0,",
                "  dx=0, dpx=0,",
                "  dy=0, dpy=0;",
                f"  vary, name=ksol, step=1.0e-5, lower={-MAX_KSOL:.12g}, upper={MAX_KSOL:.12g};",
            ]
        )
        if distributed_skew_positions:
            for family in range(1, args.distributed_skew_families + 1):
                lines.append(f"  vary, name=ksk{family}, step=1.0e-5, lower={-MAX_K1:.12g}, upper={MAX_K1:.12g};")
        else:
            lines.extend(
                [
                    f"  vary, name=kq1, step=1.0e-5, lower={-MAX_K1:.12g}, upper={MAX_K1:.12g};",
                    f"  vary, name=kq2, step=1.0e-5, lower={-MAX_K1:.12g}, upper={MAX_K1:.12g};",
                ]
            )
        lines.extend(
            [
                f"  constraint, range=#e, betx={args.target_betx:.12g};",
                f"  constraint, range=#e, bety={args.target_bety:.12g};",
                "  constraint, range=#e, alfx=0.0;",
            ]
        )
        if args.normal_mode in ("match", "match-endpoint"):
            if args.normal_families == 1:
                lines.append(f"  vary, name=knch, step=1.0e-5, lower={-MAX_K1:.12g}, upper={MAX_K1:.12g};")
            else:
                for family in range(1, args.normal_families + 1):
                    lines.append(f"  vary, name=knch{family}, step=1.0e-5, lower={-MAX_K1:.12g}, upper={MAX_K1:.12g};")
            lines.append("  constraint, range=#e, alfy=0.0;")
        elif distributed_skew_positions:
            lines.append("  constraint, range=#e, alfy=0.0;")
        if args.normal_mode == "match":
            lines.extend(
                [
                    f"  constraint, range={mid_marker}, betx={args.mid_betx:.12g};",
                    f"  constraint, range={mid_marker}, bety={args.mid_bety:.12g};",
                    f"  constraint, range={late_marker}, betx={args.mid_betx:.12g};",
                    f"  constraint, range={late_marker}, bety={args.mid_bety:.12g};",
                ]
            )
        lines.extend(["  lmdif, calls=8000, tolerance=1.0e-12;", "endmatch;", ""])
    lines.extend(["value, ksol;", "value, kq1;", "value, kq2;"])
    for family in range(1, args.distributed_skew_families + 1):
        lines.append(f"value, ksk{family};")
    lines.append("value, knch;")
    for family in range(1, args.normal_families + 1):
        lines.append(f"value, knch{family};")
    lines.extend(
        [
            "",
            "twiss,",
            "  betx=0.2351, alfx=0,",
            "  bety=0.2351, alfy=0,",
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
