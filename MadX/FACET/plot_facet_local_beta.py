#!/usr/bin/env python3
"""Plot FACET local-beta tuning results with a separate strengths panel."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt

from optimize_facet_ffd_local_beta import FAMILIES as FFD_FAMILIES
from optimize_facet_less_focusing import BRHO_TM, FAMILIES as LINE4_FAMILIES
from plot_twiss import draw_lattice_survey, numeric, read_tfs


def local_beta_summary(twiss) -> str:
    d11 = twiss[twiss["NAME"].str.upper() == "DEX20_11"].iloc[-1]
    d10 = twiss[twiss["NAME"].str.upper() == "DEX20_10"].iloc[-1]
    return "\n".join(
        [
            f"BETX at DEX20_11 = {float(d11['BETX']):.4g} m",
            f"  S = {float(d11['S']):.4g} m",
            f"BETY at DEX20_10 = {float(d10['BETY']):.4g} m",
            f"  S = {float(d10['S']):.4g} m",
        ]
    )


def family_strength_summary(twiss, families) -> str:
    quads = twiss[twiss["KEYWORD"] == "QUADRUPOLE"].copy()
    strength_by_name: dict[str, float] = {}
    for _, row in quads.iterrows():
        length = float(row["L"])
        if length == 0:
            continue
        strength_by_name[str(row["NAME"]).upper()] = float(row["K1L"]) / length * BRHO_TM

    lines = ["Varied quadrupole family gradients:"]
    for family, element_names, cap in families:
        strengths = []
        for element_name in element_names:
            value = strength_by_name.get(element_name.upper())
            if value is not None:
                strengths.append(value)
        if not strengths:
            continue
        rounded = defaultdict(int)
        for value in strengths:
            rounded[round(value, 3)] += 1
        pieces = []
        for value, count in rounded.items():
            label = f"{value:+.3f} T/m"
            if count > 1:
                label += f" x{count}"
            pieces.append(label)
        lines.append(f"{family}: {', '.join(pieces)}  | cap {cap:.3g} T/m")
    return "\n".join(lines)


def endpoint_summary(twiss) -> str:
    last = twiss.iloc[-1]
    return "\n".join(
        [
            f"End BETX={float(last['BETX']):.4g} m",
            f"End BETY={float(last['BETY']):.4g} m",
            f"End ALFX={float(last['ALFX']):.4g}",
            f"End ALFY={float(last['ALFY']):.4g}",
        ]
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else base_dir / "facet_local_beta_moderate_twiss.tfs"
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else base_dir / "facet_local_beta_moderate_plot.png"
    family_set = sys.argv[3].lower() if len(sys.argv) > 3 else "line4"
    input_path = input_path if input_path.is_absolute() else base_dir / input_path
    output_path = output_path if output_path.is_absolute() else base_dir / output_path
    families = FFD_FAMILIES if family_set == "ffd" else LINE4_FAMILIES

    twiss = read_tfs(input_path)

    fig, (survey_ax, beta_ax, alpha_ax, strengths_ax) = plt.subplots(
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
    beta_ax.legend(title="Optics", loc="upper right")
    beta_ax.text(
        0.02,
        0.95,
        local_beta_summary(twiss),
        transform=beta_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.88},
    )

    alpha_ax.plot(twiss["S"], twiss["ALFX"], color="#2563eb", linewidth=2.0, label="ALFX")
    alpha_ax.plot(twiss["S"], twiss["ALFY"], color="#9333ea", linewidth=2.0, label="ALFY")
    alpha_ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    alpha_ax.set_ylabel("Alpha")
    alpha_ax.grid(True, alpha=0.3)
    alpha_ax.legend(title="Twiss", loc="upper right")
    alpha_ax.text(
        0.02,
        0.95,
        endpoint_summary(twiss),
        transform=alpha_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.88},
    )

    strengths_ax.axis("off")
    strengths_ax.text(
        0.01,
        0.98,
        family_strength_summary(twiss, families),
        transform=strengths_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.92},
    )
    strengths_ax.set_xlabel("Longitudinal position S [m]")

    fig.suptitle(input_path.stem.replace("_", " ").upper())
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
