#!/usr/bin/env python3
"""Plot GUINEA-PIG luminosity ratio versus grid cell count."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_TXT = ROOT / "guinea_10tev_ffs_quantum_sr_grid_scan.txt"
OUT_PNG = ROOT / "guinea_10tev_ffs_quantum_sr_grid_scan_ratio.png"


def load_scan(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows.append([float(value) for value in line.split()])
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return np.asarray(rows, dtype=float)


def main() -> None:
    data = load_scan(INPUT_TXT)
    n_cells = data[:, 0].astype(int)
    transverse_extent = data[:, 1]
    lumi_0 = data[:, -2]
    lumi_gp = data[:, -1]
    ratio = lumi_gp / lumi_0

    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    for index, extent in enumerate(sorted(set(transverse_extent))):
        mask = transverse_extent == extent
        order = np.argsort(n_cells[mask])
        ax.plot(
            n_cells[mask][order],
            ratio[mask][order],
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            color=colors[index % len(colors)],
            label=f"{extent:g} sigma",
        )

    ax.set_xlabel("Number of cells in each dimension")
    ax.set_ylabel(r"$\mathcal{L}_{GP}/\mathcal{L}_{0}$")
    ax.set_title("10 TeV FFS GUINEA-PIG grid scan")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Transverse extent", ncols=2, fontsize=9)

    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    print(f"wrote={OUT_PNG}")


if __name__ == "__main__":
    main()
