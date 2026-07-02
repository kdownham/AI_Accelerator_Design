#!/usr/bin/env python3
"""Convert the 100k normal-zeta uniform-energy quantum SR endpoint to GUINEA-PIG beams."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from prepare_10tev_ffs_guinea_pig import ROOT, endpoint_beam


SOURCE = ROOT / "ffs_10tev_tracking_100k_normal_zeta_uniform_energy_quantum_sr_start_end.npz"
RUN_DIR = ROOT / "guinea-pig" / "runs" / "10tev_ffs_xsuite_quantum_sr_xy_centered"
BACKUP_SUFFIX = "before_100k_normal_zeta_uniform_energy"


def backup_if_present(path: Path) -> None:
    if path.exists():
        backup = path.with_name(f"{path.stem}_{BACKUP_SUFFIX}{path.suffix}")
        shutil.copy2(path, backup)
        print(f"backup={backup}")


def write_beams() -> None:
    beam, metadata = endpoint_beam(SOURCE)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    electron_file = RUN_DIR / "electron.ini"
    positron_file = RUN_DIR / "positron.ini"
    manifest_file = RUN_DIR / "conversion_manifest.json"

    for path in [electron_file, positron_file, manifest_file]:
        backup_if_present(path)

    np.savetxt(electron_file, beam, fmt="%.18f")

    positron_beam = beam.copy()
    positron_beam[:, 3] *= -1.0
    positron_beam = positron_beam[np.argsort(positron_beam[:, 3], kind="stable")]
    np.savetxt(positron_file, positron_beam, fmt="%.18f")

    metadata["conversion"]["note"] = (
        "Wrote electron.ini and positron.ini only; no .acc file was generated or modified."
    )
    manifest_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    print(f"source={SOURCE}")
    print(f"run_dir={RUN_DIR}")
    print(f"surviving_particles={metadata['surviving_particles']}")
    print(f"electron_file={electron_file}")
    print(f"positron_file={positron_file}")
    print(f"manifest_file={manifest_file}")


def main() -> None:
    write_beams()


if __name__ == "__main__":
    main()
