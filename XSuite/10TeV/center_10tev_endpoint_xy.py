#!/usr/bin/env python3
"""Create endpoint x/y-centered copies of the 10 TeV FFS tracking outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CASES = [
    (
        "no_sr",
        ROOT / "ffs_10tev_tracking_no_sr_start_end.npz",
        ROOT / "ffs_10tev_tracking_no_sr_start_end_xy_centered.npz",
    ),
    (
        "mean_sr",
        ROOT / "ffs_10tev_tracking_mean_sr_start_end.npz",
        ROOT / "ffs_10tev_tracking_mean_sr_start_end_xy_centered.npz",
    ),
    (
        "quantum_sr",
        ROOT / "ffs_10tev_tracking_quantum_sr_start_end.npz",
        ROOT / "ffs_10tev_tracking_quantum_sr_start_end_xy_centered.npz",
    ),
]


def center_file(label: str, source: Path, target: Path) -> None:
    with np.load(source, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    alive = np.asarray(arrays["output_state"]) > 0
    if not np.any(alive):
        raise RuntimeError(f"{source} has no live endpoint particles")

    x_mean = float(np.mean(np.asarray(arrays["output_x"], dtype=float)[alive]))
    y_mean = float(np.mean(np.asarray(arrays["output_y"], dtype=float)[alive]))

    arrays["output_x"] = np.asarray(arrays["output_x"], dtype=float) - x_mean
    arrays["output_y"] = np.asarray(arrays["output_y"], dtype=float) - y_mean
    arrays["xy_center_offset_x_m"] = np.asarray(x_mean)
    arrays["xy_center_offset_y_m"] = np.asarray(y_mean)
    arrays["xy_center_source_file"] = np.asarray(str(source.resolve()))

    np.savez_compressed(target, **arrays)

    centered_x_mean = float(np.mean(arrays["output_x"][alive]))
    centered_y_mean = float(np.mean(arrays["output_y"][alive]))
    print(
        f"{label}: wrote={target} "
        f"offset_x_m={x_mean:.12e} offset_y_m={y_mean:.12e} "
        f"centered_mean_x_m={centered_x_mean:.3e} centered_mean_y_m={centered_y_mean:.3e}"
    )


def main() -> None:
    for label, source, target in CASES:
        center_file(label, source, target)


if __name__ == "__main__":
    main()
