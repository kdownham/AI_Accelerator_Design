#!/usr/bin/env python3
"""Compare original 10 TeV BDS Twiss with the BDS portion of the combined lattice."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt

from plot_10tev_twiss import read_tfs


def main() -> None:
    bds = read_tfs(Path("bds_10TeV.twiss"))
    combined = read_tfs(Path("combined_upright_10TeV.twiss"))
    offset = 543.995489
    combined_bds = combined[combined["S"] >= offset - 1.0e-9].copy().reset_index(drop=True)
    combined_bds["S_BDS"] = combined_bds["S"] - offset
    bds = bds.iloc[: len(combined_bds)].reset_index(drop=True)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10.5), sharex=True, constrained_layout=True)
    beta_ax, beta_diff_ax, alpha_ax, disp_ax = axes

    beta_ax.plot(bds["S"], bds["BETX"], color="#007c89", linewidth=2.0, label="BDS BETX")
    beta_ax.plot(bds["S"], bds["BETY"], color="#c2410c", linewidth=2.0, label="BDS BETY")
    beta_ax.plot(combined_bds["S_BDS"], combined_bds["BETX"], color="#004f59", linewidth=1.1, linestyle="--", label="Combined BETX")
    beta_ax.plot(combined_bds["S_BDS"], combined_bds["BETY"], color="#7c2d12", linewidth=1.1, linestyle="--", label="Combined BETY")
    beta_ax.set_ylabel("Beta [m]")
    beta_ax.grid(True, alpha=0.3)
    beta_ax.legend(ncol=2, loc="upper left")

    beta_diff_ax.plot(bds["S"], combined_bds["BETX"] - bds["BETX"], color="#007c89", linewidth=1.7, label="Delta BETX")
    beta_diff_ax.plot(bds["S"], combined_bds["BETY"] - bds["BETY"], color="#c2410c", linewidth=1.7, label="Delta BETY")
    beta_diff_ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.6)
    beta_diff_ax.set_ylabel("Beta diff [m]")
    beta_diff_ax.grid(True, alpha=0.3)
    beta_diff_ax.legend(loc="upper left")

    alpha_ax.plot(bds["S"], combined_bds["ALFX"] - bds["ALFX"], color="#2563eb", linewidth=1.7, label="Delta ALFX")
    alpha_ax.plot(bds["S"], combined_bds["ALFY"] - bds["ALFY"], color="#9333ea", linewidth=1.7, label="Delta ALFY")
    alpha_ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.6)
    alpha_ax.set_ylabel("Alpha diff")
    alpha_ax.grid(True, alpha=0.3)
    alpha_ax.legend(loc="upper left")

    disp_ax.plot(bds["S"], combined_bds["DX"] - bds["DX"], color="#166534", linewidth=1.7, label="Delta DX")
    disp_ax.plot(bds["S"], combined_bds["DY"] - bds["DY"], color="#b91c1c", linewidth=1.7, label="Delta DY")
    disp_ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.6)
    disp_ax.set_ylabel("Disp. diff [m]")
    disp_ax.set_xlabel("BDS-local S [m]")
    disp_ax.grid(True, alpha=0.3)
    disp_ax.legend(loc="upper left")

    fig.suptitle("Original BDS vs Combined Upright+BDS Twiss Comparison")
    out = Path("combined_vs_bds_twiss_comparison.png")
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
