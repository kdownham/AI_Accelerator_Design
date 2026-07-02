---
name: xsuite-10tev-flatbeam-ffs
description: Use when evaluating, testing, matching, tracking, comparing, or plotting the 10 TeV flat-beam final-focus-system lattice in /home/keegan/Codex/XSuite/10TeV with Xsuite.
---

# Xsuite 10 TeV Flat-Beam FFS

## Scope

Use this skill for the lattice placed in `/home/keegan/Codex/XSuite/10TeV`.

Use the parent Xsuite environment:

```bash
cd /home/keegan/Codex/XSuite/10TeV
../.venv/bin/python --version
../.venv/bin/python -m pytest -q
```

Do not assume any prior lattice's files, Twiss values, particle parameters, beam energy, or magnet strengths apply here. Read the actual 10 TeV lattice, incoming Twiss, beam energy, emittances, particle species, and requested constraints before building or running anything.

## Input Discovery And Preservation

1. Inventory the current directory before editing or generating files.
2. Preserve all source lattice files. Create translated, matched, tracked, and plotted outputs under new descriptive prefixes.
3. Identify the source format and the intended active line or sequence before conversion.
4. If importing MAD-X, preserve element lengths, strengths, bends, tilts, markers, and the active sequence/line. Convert unsupported inert placeholders to length-preserving drifts or markers only after verifying that they have no active optical effect.
5. Compare an imported Xsuite Twiss result against a trusted source Twiss table whenever one is available. Report maximum differences for `s`, `betx`, `bety`, `alfx`, `alfy`, `dx`, and `dy` where applicable.

## Xsuite Lattice Evaluation

Default to `xo.ContextCpu()` unless the user asks for GPU tracking.

For an imported or generated line:

1. Set `line.particle_ref` from the actual particle species, `p0c`, `mass0`, and `q0`.
2. Build the tracker.
3. Run open-line Twiss using the supplied initial conditions.
4. Save a machine-readable Twiss result and a plot with a survey/element band, beta functions, alpha functions when handoff alpha matters, and dispersion only when requested or physically relevant.
5. Report line length, endpoint Twiss, peak beta, and any tracking/import warnings or exceptions.

Never call a translated lattice validated only because it imports. Validate its optics against the source reference or known endpoint conditions.

## Matching And Retargeting

Use Xsuite's native `line.match(...)` first for strength or position changes that can be stated as Twiss constraints.

- Use `xt.Vary(...)` with physical limits derived from the active strength caps.
- Use `xt.Target(...)` at named markers/elements or `xt.END`.
- Include endpoint `betx`, `bety`, `alfx`, and `alfy` when a clean handoff is required.
- Start with the smallest useful knob set. With four endpoint constraints, begin with four independent knobs rather than varying every quadrupole.
- Keep the current keeper unchanged; label candidates as baseline, trial, matched, aperture-reduced, or conservative-strength as appropriate.

For external optimization, retain endpoint constraints while optimizing peak beam size, aperture, or layout objectives. Verify every selected candidate with a final Xsuite Twiss and tracking pass.

## Physical Checks

Use the actual user-specified beam parameters. Do not infer emittance type or energy from another project.

Use these conversions where applicable:

```text
Brho [T*m] = p [GeV/c] / 0.299792458
quadrupole gradient G [T/m] = K1 [1/m^2] * Brho [T*m]
solenoid field B [T] = KS [1/m] * Brho [T*m]
sigma_x [m] = sqrt(BETX [m] * eps_x [m])
sigma_y [m] = sqrt(BETY [m] * eps_y [m])
```

Before presenting a lattice candidate, check:

- endpoint and requested marker `BETX/BETY/ALFX/ALFY`;
- peak `BETX/BETY`;
- peak RMS beam size in meters for the declared emittances;
- normal and skew quadrupole gradients in `T/m` and solenoid fields in Tesla;
- total length and last active element position;
- aperture margin if aperture data are available;
- optimizer status, particle losses, and warnings.

Do not describe a candidate as an improvement if it hits endpoint beta but worsens aperture, produces large endpoint alpha, violates strength limits, or introduces losses.

## Particle Tracking

For particle studies, generate distributions from the declared beam parameters and save the RNG seed.

For uncoupled transverse Courant-Snyder sampling, use independent samples by default:

```text
x  = sigma_x  * r1x
px = sigma_px * (-alpha_x * r1x + r2x)
y  = sigma_y  * r1y
py = sigma_py * (-alpha_y * r1y + r2y)
```

Do not reuse the same random samples between x and y unless the user explicitly requests transverse correlation. Check and report `corr(x,y)` for an intended independent distribution.

Choose storage by the requested analysis:

- For start/end evaluation, save start/end arrays, particle `state`, weights, beam parameters, and a concise summary only.
- For evolution animations, save compressed snapshots at each requested element with `element_names`, cumulative `s`, all tracked coordinates, and state. Provide a render-only path so GIF styling or speed changes do not rerun tracking.

## Current 10 TeV FFS Tracking Baseline

Use `track_10tev_ffs_sr.py` for the current FFS start/end tracking workflow:

```bash
cd /home/keegan/Codex/XSuite/10TeV
../.venv/bin/python track_10tev_ffs_sr.py
```

It uses the incoming Twiss resolved from `job_10TeV_updated.madx` through the converted FFS model:

```text
BETX=98.8079838793 m
BETY=26.7759808922 m
ALFX=ALFY=DX=DPX=DY=DPY=0
```

The current beam setup is:

```text
N=10000
energy=5e12 eV
bunch_intensity=3.72e-9 / 1.6e-19 electrons
normalized emittance x=660 nm, y=20 nm
sigma_z=44 um
sigma_delta=3e-3
```

The user explicitly requested shared `r1/r2` samples in x and y for this baseline. Preserve that correlated distribution for direct reproduction. Do not use this as the default distribution for a new study unless the user repeats that request; the generic default remains independent x/y sampling.

Run all three SR conditions from the identical seeded input distribution:

```text
none
mean
quantum
```

Expected output prefixes:

```text
ffs_10tev_tracking_no_sr_start_end.npz
ffs_10tev_tracking_mean_sr_start_end.npz
ffs_10tev_tracking_quantum_sr_start_end.npz
ffs_10tev_tracking_sr_phase_space.png
ffs_10tev_tracking_sr_summary.txt
```

The comparison plot must show input/output overlays for `x-px`, `y-py`, and `zeta-delta` under every SR condition, with physical RMS values and survivor counts printed in each panel.

## Backend Handoff Contract

The endpoint particle distribution is the future interface to the downstream component. Do not regenerate, rematch, or independently resample particles at that handoff.

For every start/end tracking output, persist at minimum:

```text
x, px, y, py, zeta, delta
state, weight, particle_id
p0c, mass0, q0
incoming Twiss, beam parameters, SR mode, RNG seed
```

When the backend is added:

1. Use only particles with positive `state` for the live-beam handoff, while retaining lost-particle records for accounting.
2. Preserve particle order/IDs so input-to-output correspondence is auditable.
3. Pass the final `x, px, y, py, zeta, delta`, particle reference, and weights directly to the next Xsuite component.
4. Keep the FFS endpoint output file immutable; write downstream results under a new prefix and record the exact parent input filename.
5. Validate the handoff with particle count, total surviving weight, centroids, RMS values, and phase-space plots on both sides of the interface.

## Synchrotron Radiation

Use Xsuite radiation modes only when requested:

```python
line.configure_radiation(model="mean")
line.configure_radiation(model="quantum")
```

For quantum radiation, seed the particle RNG when reproducibility is needed:

```python
particles._init_random_number_generator(seeds=np.arange(n_particles) + rng_seed)
```

When comparing no SR, mean SR, and quantum SR:

- use the same starting distribution and seed;
- save separate start/end outputs for each condition;
- report alive/lost counts, endpoint RMS values, and endpoint mean `delta`;
- create phase-space comparisons for `x-px`, `y-py`, and `zeta-delta` when longitudinal behavior matters;
- normalize comparison axes by RMS size only when the plot labels and annotations also report the physical RMS values.

## Plotting And Reporting

For optics plots, include a lattice survey band and show the panels needed for the decision: beta, alpha, dispersion, beam size, or strength summary. Avoid plots that conceal the current acceptance criteria.

For particle phase-space plots:

- label coordinates and units explicitly;
- use micrometers for `x`, `y`, `zeta`, microradians for `px`, `py`, and dimensionless units for `delta` unless the user specifies otherwise;
- include initial/final labels and survivor counts;
- annotate physical RMS values when axes are normalized;
- include element name and `s` in GIF frames.

For serious comparisons, generate a compact HTML summary containing the comparison plot, endpoint metrics table with sufficient precision to show mode/candidate differences, source/result filenames, and the assumptions used.

## Acceptance Checklist

Before reporting completion, include:

- source lattice and active line/sequence;
- Xsuite script and reproducible command;
- line length and element count;
- initial and endpoint Twiss values;
- peak beta and RMS beam-size metrics;
- magnet strength and aperture checks;
- particle count, distribution assumptions, surviving/lost counts, and RNG seed for tracking studies;
- radiation mode, if enabled;
- output paths for data, plots, GIFs, and HTML summaries;
- test command and result.

## Lessons Learned

- Validate source-to-Xsuite conversion numerically; import success alone is not validation.
- Treat beta, alpha, aperture, strength, and losses as a single acceptance problem rather than independent afterthoughts.
- Preserve keepers and raw source lattices. Make new branches for all trials and parameter scans.
- Separate simulation from rendering: snapshot files should support new plots, HTML, or animation timing without another tracking run.
- Keep input parameters local to this 10 TeV lattice. Do not copy beam, lattice, or radiation assumptions from another project.
