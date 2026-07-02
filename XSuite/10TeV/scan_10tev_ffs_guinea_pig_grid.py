#!/usr/bin/env python3
"""Scan GUINEA-PIG grid settings for the 10 TeV FFS quantum-SR endpoint beam."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from prepare_10tev_ffs_guinea_pig import ROOT, endpoint_beam


DEFAULT_SOURCE = ROOT / "ffs_10tev_tracking_quantum_sr_start_end.npz"
DEFAULT_SCAN_DIR = ROOT / "guinea-pig" / "runs" / "10tev_ffs_quantum_sr_grid_scan"
DEFAULT_OUTPUT = ROOT / "guinea_10tev_ffs_quantum_sr_grid_scan.txt"
GUINEA_BIN = ROOT / "guinea-pig" / "install" / "bin" / "guinea"
LOCAL_FFTW_LIB = ROOT / "guinea-pig" / "local-fftw" / "usr" / "lib" / "x86_64-linux-gnu"
CELL_COUNTS = list(range(20, 201, 30))
TRANSVERSE_EXTENTS = [6.0, 12.0, 24.0, 48.0, 64.0, 96.0, 128.0]
LONGITUDINAL_EXTENT = 6.0
LUMI_EE_RE = re.compile(r"^lumi_ee=([0-9.eE+-]+);", re.MULTILINE)


def rms(values: np.ndarray) -> float:
    return float(np.std(values))


def geometric_luminosity(population: float, sigma_x_m: float, sigma_y_m: float) -> float:
    return population * population / (4.0 * math.pi * sigma_x_m * sigma_y_m)


def accelerator_input(
    *,
    metadata: dict[str, object],
    beam: np.ndarray,
    cell_count: int,
    transverse_extent: float,
) -> str:
    endpoint_rms = metadata["endpoint_rms"]
    assert isinstance(endpoint_rms, dict)
    particle_count = int(metadata["surviving_particles"])
    bunch_population = float(metadata["bunch_population_electrons"])
    energy_GeV = float(np.mean(beam[:, 0]))

    if particle_count != 10000:
        raise ValueError(f"Expected 10000 live macroparticles, found {particle_count}")

    return f"""$ACCELERATOR:: ffs10tev
{{energy={energy_GeV:.12g};particles={bunch_population / 1e10:.12g};
beta_x=7.88371497873;beta_y=0.18019910366;emitt_x=0.660;emitt_y=0.020;
sigma_z={float(endpoint_rms['zeta_m']) * 1e6:.12g};espread={float(endpoint_rms['delta']):.12g};
dist_x=0;dist_z=0;f_rep=1.0;n_b=1;waist_x=0;waist_y=0;
offset_x.1=0;offset_y.1=0;offset_x.2=0;offset_y.2=0;}}

$PARAMETERS:: xsuite_endpoint
{{n_x={cell_count};n_y={cell_count};n_z={cell_count};n_t=1;n_m={particle_count};
automatic_grid_sizing=0;cuts_from_loaded_beam=1;integration_method=2;
cut_x_factor={transverse_extent:.12g};cut_y_factor={transverse_extent:.12g};cut_z_factor={LONGITUDINAL_EXTENT:.12g};
force_symmetric=0;charge_sign=-1.0;load_beam=3;electron_ratio=1.0;
do_photons=0;store_photons=0;do_coherent=0;do_pairs=0;track_pairs=0;store_pairs=0;
do_compt=0;do_hadrons=0;store_hadrons=0;do_muons=0;track_muons=0;store_muons=0;
do_lumi=1;num_lumi=100000;ecm_min=9000.0;hist_ee_bins=1000;hist_ee_max=11000.0;
grids=7;rndm_load=0;rndm_save=0;rndm_seed={metadata['tracking_rng_seed']};store_beam=0;}}
"""


def parse_luminosity(output: str) -> float:
    match = LUMI_EE_RE.search(output)
    if not match:
        raise RuntimeError("GUINEA-PIG output did not contain lumi_ee")
    return float(match.group(1))


def write_loaded_beams(scan_dir: Path, beam: np.ndarray) -> tuple[Path, Path]:
    scan_dir.mkdir(parents=True, exist_ok=True)
    electron_file = scan_dir / "electron.ini"
    positron_file = scan_dir / "positron.ini"
    np.savetxt(electron_file, beam, fmt="%.12e")

    positron_beam = beam.copy()
    positron_beam[:, 3] *= -1.0
    positron_beam = positron_beam[np.argsort(positron_beam[:, 3], kind="stable")]
    np.savetxt(positron_file, positron_beam, fmt="%.12e")
    return electron_file, positron_file


def write_header(path: Path) -> None:
    path.write_text(
        "# GUINEA-PIG 10 TeV FFS quantum-SR endpoint grid scan\n"
        "# f_rep=1 and n_b=1; luminosities are per bunch crossing in m^-2\n"
        "# n_macroparticles=10000; cut_z_factor=6.0; transverse_extent is cut_x/y_factor in RMS units\n"
        "# columns: n_cells transverse_extent_sigma sigma_x_ip_m sigma_y_ip_m lumi_geometric_per_bx_m^-2 lumi_guinea_lumi_ee_per_bx_m^-2\n",
        encoding="ascii",
    )


def sorted_data_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        lines.append((int(fields[0]), float(fields[1]), line))
    return [line for _, _, line in sorted(lines, key=lambda row: (row[0], row[1]))]


def sort_output_table(path: Path) -> None:
    if not path.exists():
        return
    header = [
        line
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith("#")
    ]
    lines = sorted_data_lines(path)
    path.write_text("\n".join(header + lines) + "\n", encoding="ascii")


def existing_points(path: Path) -> set[tuple[int, float]]:
    if not path.exists():
        return set()
    points: set[tuple[int, float]] = set()
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        points.add((int(fields[0]), float(fields[1])))
    return points


def run_point(
    *,
    scan_dir: Path,
    electron_file: Path,
    positron_file: Path,
    metadata: dict[str, object],
    beam: np.ndarray,
    cell_count: int,
    transverse_extent: float,
) -> float:
    run_dir = scan_dir / f"cells_{cell_count:03d}_extent_{transverse_extent:g}".replace(".", "p")
    run_dir.mkdir(parents=True, exist_ok=True)
    acc_file = run_dir / "guinea_10tev_ffs.acc"
    acc_file.write_text(
        accelerator_input(
            metadata=metadata,
            beam=beam,
            cell_count=cell_count,
            transverse_extent=transverse_extent,
        ),
        encoding="ascii",
    )
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{LOCAL_FFTW_LIB}:{env['LD_LIBRARY_PATH']}"
        if env.get("LD_LIBRARY_PATH")
        else str(LOCAL_FFTW_LIB)
    )
    result = subprocess.run(
        [
            str(GUINEA_BIN),
            "--acc_file=guinea_10tev_ffs.acc",
            f"--el_file={electron_file}",
            f"--pos_file={positron_file}",
            "ffs10tev",
            "xsuite_endpoint",
            "luminosity.out",
        ],
        cwd=run_dir,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    (run_dir / "guinea_run.log").write_text(result.stdout, encoding="ascii")
    if result.returncode != 0:
        raise RuntimeError(
            f"GUINEA-PIG failed for cells={cell_count}, "
            f"transverse_extent={transverse_extent:g}; see {run_dir / 'guinea_run.log'}"
        )
    luminosity_file = run_dir / "luminosity.out"
    if not luminosity_file.exists():
        raise RuntimeError(f"Missing GUINEA-PIG luminosity output: {luminosity_file}")
    return parse_luminosity(luminosity_file.read_text(encoding="ascii", errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fresh", action="store_true", help="replace prior scan outputs")
    args = parser.parse_args()

    if not GUINEA_BIN.is_file():
        raise FileNotFoundError(f"GUINEA-PIG executable not found: {GUINEA_BIN}")

    source = args.source.resolve()
    scan_dir = args.scan_dir.resolve()
    output = args.output.resolve()

    if args.fresh:
        if scan_dir.exists():
            shutil.rmtree(scan_dir)
        if output.exists():
            output.unlink()

    beam, metadata = endpoint_beam(source)
    if metadata["tracking_mode"] != "quantum":
        raise ValueError(f"Expected quantum SR tracking source, got {metadata['tracking_mode']!r}")
    electron_file, positron_file = write_loaded_beams(scan_dir, beam)

    endpoint_rms = metadata["endpoint_rms"]
    assert isinstance(endpoint_rms, dict)
    sigma_x_m = float(endpoint_rms["x_m"])
    sigma_y_m = float(endpoint_rms["y_m"])
    lumi_geom = geometric_luminosity(
        float(metadata["bunch_population_electrons"]), sigma_x_m, sigma_y_m
    )

    if not output.exists() or args.fresh:
        write_header(output)

    done = existing_points(output)
    total = len(CELL_COUNTS) * len(TRANSVERSE_EXTENTS)
    completed = len(done)
    print(f"source={source}")
    print(f"scan_dir={scan_dir}")
    print(f"output={output}")
    print(f"sigma_x_ip_m={sigma_x_m:.12e}")
    print(f"sigma_y_ip_m={sigma_y_m:.12e}")
    print(f"lumi_geometric_per_bx_m^-2={lumi_geom:.12e}")
    print(f"completed={completed}/{total}")

    pending = [
        (cell_count, transverse_extent)
        for cell_count in CELL_COUNTS
        for transverse_extent in TRANSVERSE_EXTENTS
        if (cell_count, transverse_extent) not in done
    ]
    if not pending:
        sort_output_table(output)
        print("scan already complete")
        return

    workers = max(1, args.workers)
    print(f"pending={len(pending)} workers={workers}")
    with output.open("a", encoding="ascii") as stream:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_point,
                    scan_dir=scan_dir,
                    electron_file=electron_file,
                    positron_file=positron_file,
                    metadata=metadata,
                    beam=beam,
                    cell_count=cell_count,
                    transverse_extent=transverse_extent,
                ): (cell_count, transverse_extent)
                for cell_count, transverse_extent in pending
            }
            for future in as_completed(futures):
                cell_count, transverse_extent = futures[future]
                lumi_guinea = future.result()
                completed += 1
                stream.write(
                    f"{cell_count:d} {transverse_extent:.12g} "
                    f"{sigma_x_m:.12e} {sigma_y_m:.12e} "
                    f"{lumi_geom:.12e} {lumi_guinea:.12e}\n"
                )
                stream.flush()
                print(
                    f"completed={completed}/{total} cells={cell_count} "
                    f"transverse_extent={transverse_extent:g} lumi_ee={lumi_guinea:.12e}",
                    flush=True,
                )
    sort_output_table(output)


if __name__ == "__main__":
    main()
