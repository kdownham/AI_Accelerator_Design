---
name: xsuite-lattice-setup
description: Use when working in /home/keegan/Codex/XSuite to install, verify, troubleshoot, match, retarget, compare, or plot Xsuite accelerator lattice designs, especially xtrack Twiss, line.match, endpoint beta/alpha matching, aperture checks, and physical magnet strength reporting.
---

# Xsuite Lattice Setup And Optics Workflow

## Scope

Use this skill for Xsuite setup and lattice-design work in `/home/keegan/Codex/XSuite`.

Default to the local virtual environment:

```bash
cd /home/keegan/Codex/XSuite
source .venv/bin/activate
```

Do not rely on the global conda install unless explicitly asked. In the observed global environment, importing `xtrack` failed because `xdeps` expected `scipy.optimize.direct`, which the global SciPy did not provide.

## Environment

The working local venv was installed with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel xsuite pytest
```

Known working package versions from the validation run:

```text
xsuite 0.50.1
xtrack 0.103.0
xpart 0.23.10
xobjects 0.6.1
xdeps 0.10.16
scipy 1.15.3
pytest 9.1.0
```

If pip cannot resolve PyPI from the sandbox, request network escalation for the pip install into `.venv`.

## Installation Validation

Run the local smoke test:

```bash
cd /home/keegan/Codex/XSuite
.venv/bin/python -m pytest -q
```

Expected current result:

```text
. [100%]
1 passed
```

The smoke test lives at `tests/test_xsuite_install.py`. It checks:

- `xtrack`, `xpart`, and dependencies import from `.venv`.
- A simple FODO-like line can be built.
- A tracker can be built.
- `line.twiss(method="4d", ...)` returns positive endpoint beta functions.
- `line.match(...)` can vary two quadrupoles to hit endpoint `BETX` and `BETY` targets.

When debugging without pytest, run:

```bash
cd /home/keegan/Codex/XSuite
.venv/bin/python -c 'import importlib.metadata as md; import xtrack; print(md.version("xsuite")); print(md.version("xtrack")); print(md.version("scipy")); print(xtrack.__file__)'
```

Confirm `xtrack.__file__` points inside:

```text
/home/keegan/Codex/XSuite/.venv/lib/python3.10/site-packages/
```

## Default Lattice Workflow

1. Preserve source and keeper designs. Write new attempts under descriptive prefixes; do not overwrite accepted baselines unless the user explicitly asks.
2. Read current metrics from generated Twiss tables, logs, JSON snapshots, or plots. Do not rely on memory.
3. Use Xsuite's native `line.match(...)` first for endpoint or marker optics constraints that can be expressed with variable strengths or positions.
4. Use external optimizers only for objectives that `line.match(...)` does not express cleanly, such as peak beam-size minimization, broad layout scans, aperture objectives, or combinatorial knob searches. Verify final candidates with Xsuite Twiss and, when appropriate, a polishing `line.match(...)`.
5. Treat endpoint `ALFX` and `ALFY` as first-class constraints whenever the user asks for a clean handoff. Do not wait for a correction after matching only `BETX/BETY`.
6. When aperture is the concern, optimize or compare RMS beam size directly rather than using beta functions as a proxy.
7. Present tradeoffs compactly: length, element count, endpoint optics, peak beta, peak beam size, max strengths, optimizer status, exceptions, and files produced.

## Xsuite Matching Pattern

For pure endpoint retargets:

- Start from the best stable candidate closest to the user's request.
- Keep the existing layout and length unless the user asks for geometry changes.
- Create Xsuite variables for matched strengths, for example `line.vars["kq005"]`, and assign element references to those variables.
- Use bounded `xt.Vary(...)` knobs when physical strength limits are active.
- Use `xt.Target(...)` constraints at named elements, markers, or `xt.END`.
- Include endpoint `betx`, `bety`, `alfx=0`, and `alfy=0` unless the user explicitly relaxes alpha.
- With four endpoint constraints, try a small useful knob set first. Search knob combinations when many quadrupoles exist, then choose by physical usefulness, not only numerical penalty.

Minimal form:

```python
line.match(
    method="4d",
    betx=betx0,
    alfx=alfx0,
    bety=bety0,
    alfy=alfy0,
    vary=[
        xt.Vary("kqf", step=1e-5, limits=(-kmax, kmax)),
        xt.Vary("kqd", step=1e-5, limits=(-kmax, kmax)),
    ],
    targets=[
        xt.Target("betx", target_betx, at=xt.END, tol=1e-8),
        xt.Target("bety", target_bety, at=xt.END, tol=1e-8),
        xt.Target("alfx", 0.0, at=xt.END, tol=1e-9),
        xt.Target("alfy", 0.0, at=xt.END, tol=1e-9),
    ],
)
```

For marker-local optics, put targets at the named marker/element, for example local `BETX` or `BETY` targets at review markers equivalent to the MadX `DEX20_10` and `DEX20_11` workflow.

## Physical Checks

Assume 5 TeV electrons only when that is the active project context or the user says so.

Use these conversions:

```text
Brho [T*m] = p [GeV/c] / 0.299792458
quadrupole gradient G [T/m] = K1 [1/m^2] * Brho [T*m]
solenoid field B [T] = KS [1/m] * Brho [T*m]
sigma_x [m] = sqrt(BETX [m] * eps_x [m])
sigma_y [m] = sqrt(BETY [m] * eps_y [m])
```

If the user gives emittance in nm without qualification, treat it as geometric RMS emittance and state that assumption. In this project history, `20 nm` geometric RMS emittance is a common aperture-review default.

Before calling a candidate useful, check:

- endpoint `BETX/BETY/ALFX/ALFY`;
- marker-local optics requested by the user;
- peak `BETX/BETY`;
- peak `sigma_x/sigma_y` in meters for the active emittance;
- max normal/skew quadrupole gradients in `T/m`;
- solenoid fields in Tesla, if any;
- total line length and last active element location;
- failed imports, exceptions, optimizer status, and test status.

Do not describe a lattice as good if it only hits endpoint beta while peak beam size, aperture, alpha handoff, or physical strengths are poor.

## Aperture And Peak Beam Size

When reducing maximum transverse beam size:

1. Start from the current best stable design, not from a random layout.
2. Keep endpoint beta/alpha constraints active.
3. Penalize peak `sigma_x` and `sigma_y` directly using the active emittance.
4. Add or retune focusing near high-beta regions before changing endpoint targets.
5. If strengths hit a cap, compare a high-performance candidate against a conservative-strength candidate.
6. Label unconverged or loose-endpoint candidates as trials, even if the aperture metric improves.

## MAD-X FACET Import Workflow

For the FACET work imported from `/home/keegan/Codex/MadX/FACET`, preserve the original MAD-X source and translate into Xsuite under `/home/keegan/Codex/XSuite`.

Current reproducible importer:

```bash
cd /home/keegan/Codex/XSuite
.venv/bin/python run_facet_madx_in_xsuite.py
```

Source and assumptions:

- Source lattice: `/home/keegan/Codex/MadX/FACET/flatGoldenLattice_line4_stub_10gev.madx`.
- Use `line_4` as the FACET line.
- Convert unsupported/non-focusing placeholders to length-preserving drifts or markers only when they are instrumentation/corrector placeholders.
- Preserve focusing and bending with Xsuite elements: `Drift`, `Marker`, `Quadrupole`, `Sextupole`, and `Bend`.
- Preserve `tilt` as `rot_s_rad` when Xsuite elements accept it.
- Use the FACET initial Twiss from the MAD-X source:

```text
BETX=29.515618, BETY=45.174946
ALFX=2.664132, ALFY=-3.582160
DX=DPX=DY=DPY=0
```

Check the Xsuite Twiss against the existing MAD-X `facet_twiss.tfs` for basic consistency before using it as a tracking model.

## Particle Tracking Workflows

Default to `xo.ContextCpu()` unless the user asks for GPU or another context.

Use descriptive output prefixes and avoid overwriting previous particle studies. Current workflows:

- `track_facet_particles_cpu.py`: one-pass Gaussian test distribution, start/end only.
- `track_facet_particles_element_gif.py`: element-by-element snapshots and GIF rendering for the simple Gaussian test distribution.
- `track_facet_realistic_independent_gif.py`: realistic FACET beam with independent x/y samples, element snapshots, and phase-space GIF.
- `track_facet_realistic_radiation_start_end.py`: realistic FACET beam with synchrotron radiation modes, start/end only.

When the user asks to save the beam at each element:

- Save a compressed NPZ with all six coordinates and `state` for every frame.
- Include `element_names`, cumulative `s`, RNG seed, and beam parameters.
- Keep a `--render-only` path for GIFs so frame rate/style changes do not rerun tracking.
- Put the element name and `s` position in every GIF frame.

When the user asks for start/end only:

- Do not save every element.
- Save start/end arrays only, plus a concise summary and optional static phase-space plot.

## Realistic FACET Beam Distribution

For the realistic FACET input distribution used in this project, use:

```text
N=10000
energy=10e9 eV
bunch_intensity=1.6e-9 / 1.6e-19 electrons
beta_x=29.51561770054117
beta_y=45.17494646985441
alpha_x=2.66413224525837
alpha_y=-3.58216043939153
normalized emittance x/y = 5e-6 m
sigma_z=20e-6 m
sigma_delta=1e-4
```

Compute geometric emittance as:

```text
gamma = energy / xt.ELECTRON_MASS_EV
physemit_x = 5e-6 / gamma
physemit_y = 5e-6 / gamma
sigma_x = sqrt(beta_x * physemit_x)
sigma_y = sqrt(beta_y * physemit_y)
sigma_px = sigma_x / beta_x
sigma_py = sigma_y / beta_y
```

Generate independent transverse samples:

```text
x  = sigma_x  * r1x
px = sigma_px * (-alpha_x * r1x + r2x)
y  = sigma_y  * r1y
py = sigma_py * (-alpha_y * r1y + r2y)
```

Do not reuse the same `r1/r2` for both planes unless the user explicitly asks for correlated x/y samples. Check and report the initial `corr(x,y)`; it should be small for independent samples. Strong `corr(x,px)` and `corr(y,py)` are expected because FACET starts with nonzero alpha.

## Synchrotron Radiation Tracking

Use Xsuite's native radiation configuration:

```python
line.configure_radiation(model="mean")
line.configure_radiation(model="quantum")
```

For quantum radiation, seed the particle RNG when reproducibility matters:

```python
particles._init_random_number_generator(seeds=np.arange(n_particles) + RNG_SEED)
```

For comparisons across no SR, mean SR, and quantum SR:

- Use the same initial realistic distribution and RNG seed.
- Save separate `*_start_end.npz` files for mean and quantum modes.
- Keep no-SR as a separate reference, either from start/end output or from first/last frames of the element-snapshot file.
- Report alive/lost counts, endpoint RMS values, and endpoint mean `delta`.
- Expect mean energy loss to appear as a negative shift in endpoint mean `delta`.

Current summary generator:

```bash
.venv/bin/python create_facet_sr_tracking_summary.py
```

## Phase-Space Plotting

For particle phase-space plots:

- Show `x-px`, `y-py`, and when relevant `zeta-delta`.
- Use micrometers for `x`, `y`, `zeta`; microradians for `px`, `py`; dimensionless units for `delta`.
- For start/end occupancy plots, include both transverse and longitudinal panels when longitudinal dynamics or SR are relevant.
- For SR comparisons, normalize axes by the RMS beam size in each dimension and print the physical RMS values on the plot.
- Use separate panels or rows for no SR, mean SR, and quantum SR.
- Keep enough significant digits in tables to see small differences between radiation modes.

Current plot/report scripts:

- `plot_facet_realistic_phase_space_endpoints.py`
- `plot_facet_sr_phase_space_rms_compare.py`
- `create_facet_sr_tracking_summary.py`

## Combined Lattices And Comparisons

When combining Xsuite lattice sections:

- Preserve the source section definitions and create a new combined line or serialized output.
- Avoid name collisions by prefixing generated element names when needed.
- Use the upstream section's initial Twiss parameters for the combined run when that section defines the physical incoming beam.
- Compare downstream subsections against a reference Twiss table by subtracting the upstream length offset from `s`.
- Report maximum absolute differences for `s`, `betx`, `bety`, `alfx`, `alfy`, `dx`, and `dy` when those columns are available.

The Xsuite analogue of the MadX combined-deck workflow is: build or load each `xt.Line`, concatenate or preserve sections in a new line object, compute Twiss from the upstream initial conditions, then compare the downstream slice against the reference Twiss DataFrame/Table.

## Plot Defaults

For optics review plots, default to:

- top lattice survey band or element-strength band;
- beta panel with `BETX/BETY`;
- alpha panel with final `BETX`, `BETY`, `ALFX`, and `ALFY` printed on the plot;
- RMS beam-size panel in meters when aperture matters;
- no dispersion panel unless the user asks for dispersion or it is central to the problem;
- strength annotations in physical units, grouping repeated or family magnets;
- marker annotations for user-named review points.

When the user asks to show a plot, open the PNG with a real image viewer such as `eog` if available.

## Output Discipline

For each serious candidate, save enough to reproduce it:

- a Python script or notebook-free `.py` generator/matcher;
- serialized lattice data if used, such as JSON from Xsuite line serialization;
- Twiss table output, preferably CSV/parquet plus any native Xsuite table export;
- match/optimization log or terminal output captured in a text file;
- PNG plot;
- optional HTML summary when documenting accepted design comparisons.

For final reporting, include:

- file prefix and whether it is a keeper or trial;
- total length and element counts;
- endpoint and requested marker optics;
- max beta and max RMS beam size;
- max gradients/fields in physical units and cap status;
- optimizer residual/status;
- test command run, especially `.venv/bin/python -m pytest -q`;
- tradeoff versus the previous keeper.

## Lessons Learned

- Check imports before assuming an existing installation is usable. A package can be present but broken by dependency mismatch.
- Keep Xsuite isolated in this project's `.venv`; this avoids disturbing the global conda environment.
- Functional validation needs more than `import xsuite`: test `xtrack.Line`, `build_tracker`, `twiss`, and `match`.
- For lattice design work, preserve a small endpoint-matching smoke test so optimizer regressions are caught immediately.
- Keep `.venv/`, `.pytest_cache/`, `__pycache__/`, and `*.pyc` ignored locally.
- Fold the MadX workflow lessons into Xsuite work immediately: match alphas with betas, report aperture and beam size early, convert normalized strengths to physical units, preserve keepers, and compare against reference optics before presenting a result.
- For FACET particle studies, default to realistic independent x/y sampling when using the supplied FACET beam parameters; save full element snapshots only when they are needed for animations, otherwise save start/end distributions.
- For synchrotron-radiation studies, compare no SR, mean SR, and quantum SR from the same input distribution; report endpoint RMS and mean `delta`, and create RMS-normalized plots plus HTML summaries when comparing modes.
