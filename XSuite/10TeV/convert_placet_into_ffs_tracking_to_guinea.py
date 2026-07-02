#!/usr/bin/env python3
"""Write GUINEA-PIG beams from PLACET-derived Xsuite tracking endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare_10tev_ffs_guinea_pig import endpoint_beam


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "guinea-pig" / "runs" / "10tev_ffs_xsuite_quantum_sr_xy_centered"
PREFIX = ROOT / "ffs_10tev_tracking_placet_into_ffs"
MODES = ("no_sr", "mean_sr", "quantum_sr")


def write_mode(mode: str) -> tuple[Path, Path, Path, int]:
    source = PREFIX.with_name(PREFIX.name + f"_{mode}_start_end.npz")
    if not source.exists():
        raise FileNotFoundError(source)

    beam, metadata = endpoint_beam(source)
    electron_file = RUN_DIR / f"electron_placet_into_ffs_tracked_{mode}.ini"
    positron_file = RUN_DIR / f"positron_placet_into_ffs_tracked_{mode}.ini"
    manifest_file = RUN_DIR / f"conversion_manifest_placet_into_ffs_tracked_{mode}.json"

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(electron_file, beam, fmt="%.18f")

    positron_beam = beam.copy()
    positron_beam[:, 3] *= -1.0
    positron_beam = positron_beam[np.argsort(positron_beam[:, 3], kind="stable")]
    np.savetxt(positron_file, positron_beam, fmt="%.18f")

    metadata["output_files"] = {
        "electron": str(electron_file.resolve()),
        "positron": str(positron_file.resolve()),
        "accelerator_file": "not generated or modified by this conversion",
    }
    manifest_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return electron_file, positron_file, manifest_file, int(metadata["surviving_particles"])


def main() -> None:
    for mode in MODES:
        electron_file, positron_file, manifest_file, particles = write_mode(mode)
        print(f"mode={mode}")
        print(f"  particles={particles}")
        print(f"  electron_file={electron_file}")
        print(f"  positron_file={positron_file}")
        print(f"  manifest_file={manifest_file}")


if __name__ == "__main__":
    main()
