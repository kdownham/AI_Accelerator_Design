#!/usr/bin/env python3
"""Convert no-SR and mean-SR 100k endpoints to named GUINEA-PIG beam files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare_10tev_ffs_guinea_pig import ROOT, endpoint_beam


RUN_DIR = ROOT / "guinea-pig" / "runs" / "10tev_ffs_xsuite_quantum_sr_xy_centered"
CASES = [
    (
        "no_sr",
        ROOT / "ffs_10tev_tracking_100k_normal_zeta_uniform_energy_no_sr_start_end.npz",
        RUN_DIR / "electron_100k_normal_zeta_uniform_energy_no_sr.ini",
        RUN_DIR / "positron_100k_normal_zeta_uniform_energy_no_sr.ini",
        RUN_DIR / "conversion_manifest_100k_normal_zeta_uniform_energy_no_sr.json",
    ),
    (
        "mean_sr",
        ROOT / "ffs_10tev_tracking_100k_normal_zeta_uniform_energy_mean_sr_start_end.npz",
        RUN_DIR / "electron_100k_normal_zeta_uniform_energy_mean_sr.ini",
        RUN_DIR / "positron_100k_normal_zeta_uniform_energy_mean_sr.ini",
        RUN_DIR / "conversion_manifest_100k_normal_zeta_uniform_energy_mean_sr.json",
    ),
]


def write_case(
    label: str,
    source: Path,
    electron_file: Path,
    positron_file: Path,
    manifest_file: Path,
) -> None:
    beam, metadata = endpoint_beam(source)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    np.savetxt(electron_file, beam, fmt="%.18f")

    positron_beam = beam.copy()
    positron_beam[:, 3] *= -1.0
    positron_beam = positron_beam[np.argsort(positron_beam[:, 3], kind="stable")]
    np.savetxt(positron_file, positron_beam, fmt="%.18f")

    metadata["conversion"]["note"] = (
        "Wrote named electron/positron .ini files only; no .acc file was generated or modified."
    )
    manifest_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii")

    print(f"{label}: source={source}")
    print(f"{label}: surviving_particles={metadata['surviving_particles']}")
    print(f"{label}: electron_file={electron_file}")
    print(f"{label}: positron_file={positron_file}")
    print(f"{label}: manifest_file={manifest_file}")


def main() -> None:
    for case in CASES:
        write_case(*case)


if __name__ == "__main__":
    main()
