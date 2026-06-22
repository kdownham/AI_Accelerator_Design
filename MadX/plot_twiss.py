#!/usr/bin/env python3
"""Plot selected optics columns from a MAD-X TFS TWISS table."""

from __future__ import annotations

import os
import shlex
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


DEFAULT_EMITTANCE_NM = 20.0


def read_tfs(path: Path) -> pd.DataFrame:
    columns: list[str] | None = None
    rows: list[list[str]] = []

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("@") or line.startswith("$"):
                continue
            if line.startswith("*"):
                columns = line.split()[1:]
                continue
            if columns is None:
                continue
            rows.append(shlex.split(line))

    if columns is None:
        raise ValueError(f"No TFS column header found in {path}")

    frame = pd.DataFrame(rows, columns=columns)
    for column in ("S", "BETX", "BETY", "ALFX", "ALFY", "DX", "DY"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def read_tfs_metadata(path: Path) -> dict[str, float | str]:
    metadata: dict[str, float | str] = {}
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("*"):
                break
            if not line.startswith("@"):
                continue
            parts = shlex.split(line)
            if len(parts) < 4:
                continue
            key = parts[1]
            value = parts[3]
            try:
                metadata[key] = float(value)
            except ValueError:
                metadata[key] = value
    return metadata


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def draw_lattice_survey(ax: plt.Axes, twiss: pd.DataFrame) -> None:
    s = numeric(twiss, "S")
    length = float(s.max())

    ax.set_xlim(0, length)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.set_ylabel("Optics")
    ax.grid(True, axis="x", alpha=0.2)

    markers = twiss[twiss["KEYWORD"] == "MARKER"]
    for position in numeric(markers, "S"):
        ax.vlines(position, -0.18, 0.18, color="#9aa0a6", linewidth=0.45, alpha=0.55)

    solenoids = twiss[twiss["KEYWORD"] == "SOLENOID"].copy()
    if not solenoids.empty:
        for _, solenoid in solenoids.iterrows():
            sol_s = float(solenoid["S"])
            sol_l = float(solenoid["L"])
            ax.add_patch(Rectangle((sol_s - sol_l, -0.36), sol_l, 0.72, color="#2f80ed", alpha=0.65))
        starts = numeric(solenoids, "S") - numeric(solenoids, "L")
        ends = numeric(solenoids, "S")
        ax.text(
            (float(starts.min()) + float(ends.max())) / 2,
            0,
            "SOL",
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )

    quads = twiss[twiss["KEYWORD"] == "QUADRUPOLE"].copy()
    for _, quad in quads.iterrows():
        q_s = float(quad["S"])
        q_l = float(quad["L"])
        k1l = float(quad["K1L"])
        y0 = 0.12 if k1l >= 0 else -0.72
        color = "#008b8b" if k1l >= 0 else "#b22222"
        ax.add_patch(Rectangle((q_s - q_l, y0), q_l, 0.60, color=color, alpha=0.75))
        ax.text(q_s - q_l / 2, y0 + 0.30, quad["NAME"], ha="center", va="center", color="white", fontsize=8)


def strength_summary(twiss: pd.DataFrame, metadata: dict[str, float | str]) -> str:
    lines: list[str] = []
    pc_gev = abs(float(metadata.get("PC", 0.0)))
    brho_tm = pc_gev / 0.299792458 if pc_gev else 0.0
    if brho_tm:
        lines.append(f"Beam rigidity: Brho={brho_tm:.3g} T*m")

    solenoids = twiss[twiss["KEYWORD"] == "SOLENOID"].copy()
    if not solenoids.empty and "KSI" in solenoids:
        total_l = numeric(solenoids, "L").sum()
        ks = numeric(solenoids, "KSI").sum() / total_l
        field_t = ks * brho_tm
        lines.append(f"Solenoid: L={total_l:.1f} m, B={field_t:.3g} T")

    quads = twiss[twiss["KEYWORD"] == "QUADRUPOLE"].copy()
    grouped_quads: dict[tuple[float, float], list[str]] = defaultdict(list)
    for _, quad in quads.iterrows():
        length = float(quad["L"])
        k1 = float(quad["K1L"]) / length if length else 0.0
        gradient_tm = k1 * brho_tm
        tilt_deg = float(quad["TILT"]) * 180.0 / 3.141592653589793
        grouped_quads[(round(gradient_tm, 6), round(tilt_deg, 3))].append(str(quad["NAME"]))
    for (gradient_tm, tilt_deg), names in grouped_quads.items():
        if len(names) == 1:
            label = names[0]
        else:
            prefixes = {name.rstrip("0123456789") for name in names}
            if len(prefixes) == 1:
                label = f"{next(iter(prefixes))}* x{len(names)}"
            else:
                label = f"{names[0]}..{names[-1]} x{len(names)}"
        lines.append(f"{label}: G={gradient_tm:.3g} T/m, tilt={tilt_deg:.0f} deg")
    return "\n".join(lines)


def beam_size_meters(twiss: pd.DataFrame, emittance_nm: float) -> tuple[pd.Series, pd.Series]:
    emittance_m = emittance_nm * 1.0e-9
    sigma_x_m = (numeric(twiss, "BETX") * emittance_m) ** 0.5
    sigma_y_m = (numeric(twiss, "BETY") * emittance_m) ** 0.5
    return sigma_x_m, sigma_y_m


def endpoint_summary(twiss: pd.DataFrame) -> str:
    last = twiss.iloc[-1]
    return "\n".join(
        [
            f"End S={float(last['S']):.3g} m",
            f"BETX={float(last['BETX']):.4g} m",
            f"BETY={float(last['BETY']):.4g} m",
            f"ALFX={float(last['ALFX']):.4g}",
            f"ALFY={float(last['ALFY']):.4g}",
        ]
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else base_dir / "fodo_twiss.tfs"
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else base_dir / "fodo_twiss_plot.png"
    emittance_nm = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_EMITTANCE_NM
    input_path = input_path if input_path.is_absolute() else base_dir / input_path
    output_path = output_path if output_path.is_absolute() else base_dir / output_path

    twiss = read_tfs(input_path)
    metadata = read_tfs_metadata(input_path)
    sigma_x_m, sigma_y_m = beam_size_meters(twiss, emittance_nm)

    fig, (survey_ax, beta_ax, alpha_ax, beam_ax) = plt.subplots(
        4,
        1,
        figsize=(12, 10.4),
        sharex=True,
        height_ratios=(0.7, 3.0, 1.6, 2.0),
        constrained_layout=True,
    )

    draw_lattice_survey(survey_ax, twiss)

    beta_ax.plot(twiss["S"], twiss["BETX"], color="#008b8b", linewidth=2.2, label="BETX")
    beta_ax.plot(twiss["S"], twiss["BETY"], color="#b22222", linewidth=2.2, label="BETY")
    beta_ax.set_ylabel("Beta function [m]")
    beta_ax.grid(True, alpha=0.3)
    beta_ax.legend(title="Optics")
    summary = strength_summary(twiss, metadata)
    if summary:
        beta_ax.text(
            0.99,
            0.98,
            summary,
            transform=beta_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.88},
        )

    alpha_ax.plot(twiss["S"], twiss["ALFX"], color="#2563eb", linewidth=2.0, label="ALFX")
    alpha_ax.plot(twiss["S"], twiss["ALFY"], color="#9333ea", linewidth=2.0, label="ALFY")
    alpha_ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    alpha_ax.set_ylabel("Alpha")
    alpha_ax.grid(True, alpha=0.3)
    alpha_ax.legend(title="Twiss")
    alpha_ax.text(
        0.99,
        0.95,
        endpoint_summary(twiss),
        transform=alpha_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.88},
    )

    beam_ax.plot(twiss["S"], sigma_x_m, color="#0f766e", linewidth=2.0, label=r"$\sigma_x$")
    beam_ax.plot(twiss["S"], sigma_y_m, color="#dc2626", linewidth=2.0, label=r"$\sigma_y$")
    beam_ax.set_ylabel("RMS beam size [m]")
    beam_ax.set_xlabel("Longitudinal position S [m]")
    beam_ax.grid(True, alpha=0.3)
    beam_ax.legend(title=f"epsilon={emittance_nm:g} nm")
    beam_ax.text(
        0.99,
        0.95,
        f"max sigma_x={sigma_x_m.max():.3g} m\nmax sigma_y={sigma_y_m.max():.3g} m",
        transform=beam_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.88},
    )

    fig.suptitle(input_path.stem.replace("_", " ").upper())
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
