#!/usr/bin/env python3
"""Convert a tracked 10 TeV FFS endpoint bunch into GUINEA-PIG input files.

The installed GUINEA-PIG version reads external beam files with columns:
energy [GeV], x [um], y [um], z [um], x' [urad], y' [urad].
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "ffs_10tev_tracking_no_sr_start_end.npz"
DEFAULT_RUN_DIR = ROOT / "guinea-pig" / "runs" / "10tev_ffs_xsuite_no_sr"
ELECTRON_MASS_EV = 0.510998950e6


def rms(values: np.ndarray) -> float:
    return float(np.std(values))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def endpoint_beam(source: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(source) as data:
        required = [
            "output_x",
            "output_px",
            "output_y",
            "output_py",
            "output_zeta",
            "output_delta",
            "output_state",
            "output_weight",
            "p0c_eV",
            "mass0_eV",
            "bunch_intensity",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{source} is missing required arrays: {', '.join(missing)}")

        alive = np.asarray(data["output_state"]) > 0
        if not np.any(alive):
            raise RuntimeError("No live particles are available at the FFS endpoint")

        x = np.asarray(data["output_x"], dtype=float)[alive]
        px = np.asarray(data["output_px"], dtype=float)[alive]
        y = np.asarray(data["output_y"], dtype=float)[alive]
        py = np.asarray(data["output_py"], dtype=float)[alive]
        zeta = np.asarray(data["output_zeta"], dtype=float)[alive]
        delta = np.asarray(data["output_delta"], dtype=float)[alive]
        weights = np.asarray(data["output_weight"], dtype=float)[alive]
        p0c_eV = float(data["p0c_eV"])
        mass0_eV = float(data["mass0_eV"])
        bunch_intensity = float(data["bunch_intensity"])
        mode = str(data["mode"].item())
        rng_seed = int(data["rng_seed"])

    if not np.allclose(weights, weights[0], rtol=1e-12, atol=0.0):
        raise ValueError("GUINEA-PIG external beams require uniform macro-particle weights")

    momentum_ratio = 1.0 + delta
    pz_ratio_sq = momentum_ratio**2 - px**2 - py**2
    if np.any(pz_ratio_sq <= 0.0):
        raise ValueError("Encountered a particle without real longitudinal momentum")
    pz_ratio = np.sqrt(pz_ratio_sq)

    # Xsuite px/py are normalized canonical momenta. Convert to geometric
    # slopes p_x/p_z and p_y/p_z for the GUINEA-PIG external-beam format.
    xp_rad = px / pz_ratio
    yp_rad = py / pz_ratio
    energy_GeV = np.sqrt((p0c_eV * momentum_ratio) ** 2 + mass0_eV**2) * 1e-9

    # In Xsuite zeta is positive toward the bunch head; GUINEA-PIG uses
    # negative z for the head. Sort by GUINEA-PIG z as required by its reader.
    guinea_z_um = -zeta * 1e6
    beam = np.column_stack(
        [energy_GeV, x * 1e6, y * 1e6, guinea_z_um, xp_rad * 1e6, yp_rad * 1e6]
    )
    order = np.argsort(beam[:, 3], kind="stable")
    beam = beam[order]

    if not np.all(np.isfinite(beam)):
        raise ValueError("Non-finite values encountered while preparing GUINEA-PIG input")
    if np.any(np.diff(beam[:, 3]) < 0.0):
        raise AssertionError("GUINEA-PIG input was not sorted by z")

    metadata: dict[str, object] = {
        "source_file": str(source.resolve()),
        "source_sha256": sha256(source),
        "tracking_mode": mode,
        "tracking_rng_seed": rng_seed,
        "source_particles": int(alive.size),
        "surviving_particles": int(alive.sum()),
        "macro_particle_weight_electrons": float(weights[0]),
        "bunch_population_electrons": bunch_intensity,
        "p0c_eV": p0c_eV,
        "mass0_eV": mass0_eV,
        "conversion": {
            "energy": "sqrt((p0c*(1+delta))^2 + mass0^2) in GeV",
            "position": "x/y in m to um; z=-zeta in um",
            "angle": "x'=px/sqrt((1+delta)^2-px^2-py^2), likewise y', in urad",
            "ordering": "ascending GUINEA-PIG z",
            "counter_beam": "identical endpoint ensemble with z mirrored to align GUINEA-PIG's reversed collision slices",
        },
        "endpoint_rms": {
            "x_m": rms(x),
            "y_m": rms(y),
            "zeta_m": rms(zeta),
            "xp_rad": rms(xp_rad),
            "yp_rad": rms(yp_rad),
            "delta": rms(delta),
        },
    }
    return beam, metadata


def accelerator_input(metadata: dict[str, object], beam: np.ndarray) -> str:
    endpoint_rms = metadata["endpoint_rms"]
    assert isinstance(endpoint_rms, dict)
    particle_count = int(metadata["surviving_particles"])
    bunch_population = float(metadata["bunch_population_electrons"])
    energy_GeV = float(np.mean(beam[:, 0]))

    # The endpoint Twiss values are from the validated Xsuite FFS conversion.
    # GUINEA-PIG derives grid extents from the loaded particles, not these values.
    return f"""$ACCELERATOR:: ffs10tev
{{energy={energy_GeV:.12g};particles={bunch_population / 1e10:.12g};
beta_x=7.88371497873;beta_y=0.18019910366;emitt_x=0.660;emitt_y=0.020;
sigma_z={float(endpoint_rms['zeta_m']) * 1e6:.12g};espread={float(endpoint_rms['delta']):.12g};
dist_x=0;dist_z=0;f_rep=1.0;n_b=1;waist_x=0;waist_y=0;
offset_x.1=0;offset_y.1=0;offset_x.2=0;offset_y.2=0;}}

$PARAMETERS:: xsuite_endpoint
{{n_x=32;n_y=32;n_z=32;n_t=1;n_m={particle_count};
automatic_grid_sizing=0;cuts_from_loaded_beam=1;
cut_x_factor=5.0;cut_y_factor=5.0;cut_z_factor=5.0;
force_symmetric=0;charge_sign=-1.0;load_beam=3;electron_ratio=1.0;
do_photons=0;store_photons=0;do_coherent=0;do_pairs=0;track_pairs=0;store_pairs=0;
do_compt=0;do_hadrons=0;store_hadrons=0;do_muons=0;track_muons=0;store_muons=0;
do_lumi=1;num_lumi=100000;ecm_min=9000.0;hist_ee_bins=1000;hist_ee_max=11000.0;
grids=7;rndm_load=0;rndm_save=0;rndm_seed={metadata['tracking_rng_seed']};store_beam=0;}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    source = args.source.resolve()
    run_dir = args.run_dir.resolve()
    beam, metadata = endpoint_beam(source)
    run_dir.mkdir(parents=True, exist_ok=True)

    electron_file = run_dir / "electron.ini"
    positron_file = run_dir / "positron.ini"
    np.savetxt(electron_file, beam, fmt="%.12e")

    # GUINEA-PIG collides beam-1 slice i with beam-2 slice n-i-1. Mirror
    # the counter-beam longitudinal coordinate so matching endpoint samples
    # meet at the IP, then re-sort as required by the external-beam reader.
    positron_beam = beam.copy()
    positron_beam[:, 3] *= -1.0
    positron_beam = positron_beam[np.argsort(positron_beam[:, 3], kind="stable")]
    np.savetxt(positron_file, positron_beam, fmt="%.12e")
    (run_dir / "guinea_10tev_ffs.acc").write_text(accelerator_input(metadata, beam), encoding="ascii")
    (run_dir / "conversion_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )

    print(f"source={source}")
    print(f"run_dir={run_dir}")
    print(f"surviving_particles={metadata['surviving_particles']}")
    print(f"beam_population={metadata['bunch_population_electrons']:.12g}")
    print(f"electron_file={electron_file}")
    print(f"positron_file={positron_file}")
    print(f"acc_file={run_dir / 'guinea_10tev_ffs.acc'}")


if __name__ == "__main__":
    main()
