---
name: madx-10tev-bds-combiner
description: Use for work in /home/keegan/Codex/MadX/10TeV that combines the 10 TeV MAD-X job, FFS lattice, and optimized strengths into a single MAD-X deck, runs MAD-X, creates Twiss output, and plots beta/alpha/dispersion without changing the lattice definitions.
---

# MAD-X 10 TeV BDS Combiner

Use this skill for the 10 TeV BDS files in `/home/keegan/Codex/MadX/10TeV`.

## Source Files

The directory-specific source files are:

- `job_10TeV_updated.madx`: executable job deck with `CALL` statements and Twiss commands.
- `ffs_10TeV_updated.madx`: FFS/BDS lattice definitions.
- `optimized_strengths_updated.madx`: optimized strength assignments.

Preserve those source files. Write generated outputs under new descriptive names.

## Combining Workflow

1. Read the job deck and identify `CALL, FILE=...` statements.
2. Inline `ffs_10TeV_updated.madx` and `optimized_strengths_updated.madx` at the exact original `CALL` locations.
3. Do not alter element definitions, sequence structure, strengths, Twiss setup, or lattice parameters unless the user explicitly asks.
4. Save the combined deck as `combined_10TeV_updated.madx` or a similarly descriptive new filename.
5. Verify there are no remaining `CALL` statements before running MAD-X.

## Run And Plot

Run MAD-X with the local parent executable:

```bash
../madx combined_10TeV_updated.madx
```

Save a durable log:

```bash
combined_10TeV_updated.log
```

Expected primary Twiss output:

```text
bds_10TeV.twiss
```

Use `plot_10tev_twiss.py` to produce a PNG showing:

- lattice survey band;
- `BETX/BETY`;
- `ALFX/ALFY`;
- `DX/DY`.

Default plot:

```bash
python3 plot_10tev_twiss.py bds_10TeV.twiss bds_10TeV_twiss_plot.png
```

## Validation Checklist

Before reporting completion, check:

- MAD-X finished normally.
- `Number of warnings: 0`, or explicitly report the warning count.
- `bds_10TeV.twiss` exists.
- `bds_10TeV_twiss_plot.png` exists.
- Summary metrics from the Twiss file:
  - line length;
  - endpoint `BETX/BETY/ALFX/ALFY`;
  - max `BETX/BETY`;
  - max/min `DX`.

When the user asks to open a plot, use a real image viewer such as `eog`.
