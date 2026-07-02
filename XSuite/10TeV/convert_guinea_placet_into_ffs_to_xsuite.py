#!/usr/bin/env python3
"""Convert GUINEA-PIG/PLACET beam files into Xsuite-ready particle arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ELECTRON_MASS_EV = 0.510998950e6
P0C_EV = 5.0e12
BUNCH_INTENSITY = 3.72e-9 / 1.6e-19
FILES = [
    (
        "electron",
        -1,
        ROOT / "electron_placet_into_FFS.ini",
        ROOT / "electron_placet_into_FFS_xsuite_particles.npz",
    ),
    (
        "positron",
        1,
        ROOT / "positron_placet_into_FFS.ini",
        ROOT / "positron_placet_into_FFS_xsuite_particles.npz",
    ),
]


def load_guinea_file(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"{path} should have six columns, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains non-finite values")
    return data


def convert(data: np.ndarray, q0: int) -> dict[str, np.ndarray]:
    energy_GeV = data[:, 0]
    x_um = data[:, 1]
    y_um = data[:, 2]
    z_um = data[:, 3]
    xp_urad = data[:, 4]
    yp_urad = data[:, 5]

    energy_eV = energy_GeV * 1e9
    momentum_eV = np.sqrt(np.maximum(energy_eV**2 - ELECTRON_MASS_EV**2, 0.0))
    momentum_ratio = momentum_eV / P0C_EV
    delta = momentum_ratio - 1.0

    slope_x = xp_urad * 1e-6
    slope_y = yp_urad * 1e-6
    pz_ratio = momentum_ratio / np.sqrt(1.0 + slope_x**2 + slope_y**2)
    px = slope_x * pz_ratio
    py = slope_y * pz_ratio

    n_particles = data.shape[0]
    return {
        "x": x_um * 1e-6,
        "px": px,
        "y": y_um * 1e-6,
        "py": py,
        "zeta": -z_um * 1e-6,
        "delta": delta,
        "weight": np.full(n_particles, BUNCH_INTENSITY / n_particles),
        "particle_id": np.arange(n_particles, dtype=np.int64),
        "state": np.ones(n_particles, dtype=np.int64),
        "q0": np.asarray(q0),
        "p0c_eV": np.asarray(P0C_EV),
        "mass0_eV": np.asarray(ELECTRON_MASS_EV),
        "bunch_intensity": np.asarray(BUNCH_INTENSITY),
        "source_energy_GeV": energy_GeV,
        "source_x_um": x_um,
        "source_y_um": y_um,
        "source_z_um": z_um,
        "source_xp_urad": xp_urad,
        "source_yp_urad": yp_urad,
    }


def stats(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    return {
        "n_particles": int(arrays["x"].size),
        "q0": int(arrays["q0"]),
        "mean_x_m": float(np.mean(arrays["x"])),
        "rms_x_m": float(np.std(arrays["x"])),
        "mean_y_m": float(np.mean(arrays["y"])),
        "rms_y_m": float(np.std(arrays["y"])),
        "mean_zeta_m": float(np.mean(arrays["zeta"])),
        "rms_zeta_m": float(np.std(arrays["zeta"])),
        "mean_delta": float(np.mean(arrays["delta"])),
        "rms_delta": float(np.std(arrays["delta"])),
        "mean_px": float(np.mean(arrays["px"])),
        "rms_px": float(np.std(arrays["px"])),
        "mean_py": float(np.mean(arrays["py"])),
        "rms_py": float(np.std(arrays["py"])),
        "total_weight": float(np.sum(arrays["weight"])),
    }


def main() -> None:
    manifest: dict[str, object] = {
        "format": "npz arrays readable by numpy and usable to construct xpart.Particles",
        "units": {
            "x_y_zeta": "m",
            "px_py": "normalized momenta relative to p0c",
            "delta": "p/p0c - 1",
            "p0c_eV": "eV",
            "mass0_eV": "eV",
        },
        "conversion": {
            "source_columns": "energy [GeV], x [um], y [um], z [um], x' [urad], y' [urad]",
            "zeta": "zeta = -z_um * 1e-6",
            "delta": "sqrt((energy_eV)^2 - mass0^2) / p0c_eV - 1",
            "px_py": "from slopes x'=px/pz, y'=py/pz",
        },
        "outputs": {},
    }

    for label, q0, source, target in FILES:
        source_data = load_guinea_file(source)
        arrays = convert(source_data, q0)
        np.savez_compressed(target, **arrays)
        manifest["outputs"][label] = {
            "source": str(source.resolve()),
            "target": str(target.resolve()),
            "stats": stats(arrays),
        }
        print(f"{label}: source={source}")
        print(f"{label}: wrote={target}")
        print(f"{label}: n_particles={arrays['x'].size}")

    manifest_path = ROOT / "placet_into_FFS_xsuite_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
