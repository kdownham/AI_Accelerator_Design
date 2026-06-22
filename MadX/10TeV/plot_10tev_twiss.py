#!/usr/bin/env python3
"""Plot Twiss parameters for the combined 10 TeV MAD-X lattice."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


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
            parts = shlex.split(line)
            if len(parts) == len(columns):
                rows.append(parts)
    if columns is None:
        raise ValueError(f"No TFS column header found in {path}")
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("S", "L", "BETX", "BETY", "ALFX", "ALFY", "DX", "DY", "K1L", "K2L", "ANGLE"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def draw_lattice(ax: plt.Axes, twiss: pd.DataFrame) -> None:
    length = float(twiss["S"].max())
    ax.set_xlim(0.0, length)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.set_ylabel("Lattice")
    ax.grid(True, axis="x", alpha=0.2)

    styles = {
        "SBEND": ("#6b7280", -0.25, 0.50),
        "QUADRUPOLE": ("#008b8b", 0.20, 0.55),
        "SEXTUPOLE": ("#b45309", -0.80, 0.45),
        "MULTIPOLE": ("#7c3aed", -0.18, 0.36),
    }
    for _, row in twiss.iterrows():
        keyword = str(row.get("KEYWORD", "")).upper()
        if keyword not in styles:
            continue
        element_length = float(row.get("L", 0.0) or 0.0)
        if element_length <= 0.0:
            element_length = 0.4
        end_s = float(row["S"])
        start_s = max(0.0, end_s - element_length)
        color, y0, height = styles[keyword]
        if keyword == "QUADRUPOLE":
            y0 = 0.20 if float(row.get("K1L", 0.0) or 0.0) >= 0 else -0.75
        ax.add_patch(Rectangle((start_s, y0), element_length, height, color=color, alpha=0.72))


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bds_10TeV.twiss")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bds_10TeV_twiss_plot.png")
    twiss = read_tfs(input_path)

    fig, (survey_ax, beta_ax, alpha_ax, disp_ax) = plt.subplots(
        4,
        1,
        figsize=(13.0, 10.5),
        sharex=True,
        height_ratios=(0.65, 2.7, 1.5, 1.5),
        constrained_layout=True,
    )

    draw_lattice(survey_ax, twiss)

    beta_ax.plot(twiss["S"], twiss["BETX"], color="#007c89", linewidth=2.0, label="BETX")
    beta_ax.plot(twiss["S"], twiss["BETY"], color="#c2410c", linewidth=2.0, label="BETY")
    beta_ax.set_ylabel("Beta [m]")
    beta_ax.grid(True, alpha=0.3)
    beta_ax.legend(loc="upper left")
    beta_ax.text(
        0.99,
        0.96,
        f"max BETX={twiss['BETX'].max():.4g} m\nmax BETY={twiss['BETY'].max():.4g} m",
        transform=beta_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.88},
    )

    alpha_ax.plot(twiss["S"], twiss["ALFX"], color="#2563eb", linewidth=1.8, label="ALFX")
    alpha_ax.plot(twiss["S"], twiss["ALFY"], color="#9333ea", linewidth=1.8, label="ALFY")
    alpha_ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    alpha_ax.set_ylabel("Alpha")
    alpha_ax.grid(True, alpha=0.3)
    alpha_ax.legend(loc="upper left")

    disp_ax.plot(twiss["S"], twiss["DX"], color="#166534", linewidth=1.8, label="DX")
    disp_ax.plot(twiss["S"], twiss["DY"], color="#b91c1c", linewidth=1.8, label="DY")
    disp_ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    disp_ax.set_ylabel("Dispersion [m]")
    disp_ax.set_xlabel("S [m]")
    disp_ax.grid(True, alpha=0.3)
    disp_ax.legend(loc="upper left")

    last = twiss.iloc[-1]
    fig.suptitle(
        "10 TeV BDS Twiss Parameters: "
        f"End BETX={float(last['BETX']):.4g} m, BETY={float(last['BETY']):.4g} m"
    )
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
