---
name: madx-10tev-upright-bds-compare
description: Use for /home/keegan/Codex/MadX/10TeV work that prepends the upright endpoint transformer lattice to the 10 TeV BDS lattice, runs the combined MAD-X line from the upright initial Twiss parameters, and compares the BDS portion against bds_10TeV.twiss.
---

# MAD-X 10 TeV Upright+BDS Compare

Use this skill when combining the upright endpoint transformer with the 10 TeV BDS lattice in `/home/keegan/Codex/MadX/10TeV`.

## Inputs

Typical inputs:

- `combined_10TeV_updated.madx`: inlined 10 TeV BDS deck.
- `../upright_endpoint_98p808_26p776_madx_match.madx`: upright endpoint transformer and MATCH deck.
- `bds_10TeV.twiss`: reference BDS-only Twiss table.

Do not modify these source files. Create a new combined deck.

## Combination Workflow

1. Inline the upright endpoint lattice before the 10 TeV BDS definitions.
2. Keep the upright `MATCH` block if the source deck contains matched variables, so the source lattice is reproducible.
3. Use the upright initial Twiss parameters for the final combined Twiss:

```text
BETX=0.2351, ALFX=0
BETY=0.2351, ALFY=0
DX=0, DPX=0
DY=0, DPY=0
```

4. MAD-X cannot place a `SEQUENCE` name directly inside a `LINE`. If the upright source is a `SEQUENCE`, derive an equivalent line using the same markers, quadrupoles, and drift gaps:
   - prefix generated upright elements with `U_` to avoid name collisions;
   - preserve quadrupole lengths, positions, and `k1` expressions;
   - insert explicit drifts between sequence elements;
   - concatenate with `COMBINED_U10: LINE=(UPRIGHT_EQ, NEWBDS);`.
5. Run final Twiss over `COMBINED_U10` and write:

```text
combined_upright_10TeV.twiss
```

## Expected Outputs

Use these names unless the user asks otherwise:

- `combined_upright_10TeV.madx`
- `combined_upright_10TeV.log`
- `combined_upright_10TeV.twiss`
- `combined_upright_10TeV_twiss_plot.png`
- `combined_vs_bds_twiss_comparison.csv`
- `combined_vs_bds_twiss_comparison.png`

## Comparison Workflow

Compare the BDS portion of the combined Twiss against `bds_10TeV.twiss` by subtracting the upright section length from the combined `S` coordinate.

For the current upright endpoint transformer, the offset is:

```text
543.995489 m
```

Report maximum absolute differences for:

- `S`;
- `BETX`;
- `BETY`;
- `ALFX`;
- `ALFY`;
- `DX`;
- `DY`.

The BDS portion should reproduce the original BDS-only Twiss to numerical/TFS print precision when the handoff Twiss is unchanged.

## Plotting

Use:

```bash
python3 plot_10tev_twiss.py combined_upright_10TeV.twiss combined_upright_10TeV_twiss_plot.png
python3 plot_combined_vs_bds.py
```

The first plot shows the full upright+BDS optics. The second plot overlays the BDS-only optics against the BDS subsection of the combined lattice and plots differences.

## Validation Checklist

Before presenting the result, report:

- MAD-X final warning count.
- Upright MATCH final penalty if a MATCH block was run.
- Combined line length.
- Endpoint `BETX/BETY/ALFX/ALFY`.
- Peak `BETX/BETY`.
- BDS-local max absolute differences from `bds_10TeV.twiss`.
- Links to the combined deck, log, Twiss output, full plot, comparison CSV, and comparison plot.
