---
name: madx-upright-endpoint-retarget
description: Use when changing endpoint BETX/BETY targets for the upright-quadrupole-only round-to-flat MAD-X transformer, especially starting from upright_peak_reduced_10q_matched, using MAD-X MATCH with endpoint alpha constraints, preserving aperture metrics, saving static result decks, and creating HTML summaries.
---

# MAD-X Upright Endpoint Retarget

Use this skill for endpoint target changes on the upright-quadrupole-only round-to-flat transformer in `/home/keegan/Codex/MadX`.

## Default Starting Point

- Start from `upright_peak_reduced_10q_matched.*` unless the user names another keeper.
- Preserve the source files. Write new branches with a target-specific prefix such as `upright_endpoint_98p808_26p776`.
- Assume 5 TeV electrons and 20 nm geometric RMS emittance unless the user changes that context.

## Retarget Workflow

1. Use MAD-X `MATCH` first.
2. Keep the existing length and quadrupole positions for a pure endpoint retarget.
3. Constrain the endpoint to the requested `BETX`, requested `BETY`, `ALFX=0`, and `ALFY=0`.
4. Use bounded `VARY` knobs in physical units converted through `K1 = G[T/m] / Brho[T*m]`.
5. With four endpoint constraints, vary four quadrupole knobs at a time. Search combinations and then select the candidate that preserves physical usefulness:
   - zero MAD-X warnings;
   - tiny endpoint residual and clean `MATCH` final penalty;
   - max gradient still within the active cap, usually `160 T/m` unless changed;
   - max RMS beam size unchanged or improved;
   - smallest reasonable gradient changes.
6. Save both outputs:
   - `*_madx_match.madx`, `*_madx_match.log`, `*_madx_match_twiss.tfs`;
   - `*_madx_match_result.madx`, `*_madx_match_result.log`, `*_madx_match_result_twiss.tfs`.
7. Regenerate `*_madx_match_result_plot.png` with `plot_twiss.py`.

## Preferred Helper

Use `match_upright_endpoint_target.py` when possible. Example:

```bash
python3 match_upright_endpoint_target.py \
  --target-betx 98.80798387931307 \
  --target-bety 26.775980892186183 \
  --gradient-bound-tm 160 \
  --output-prefix upright_endpoint_98p808_26p776
```

Then plot:

```bash
python3 plot_twiss.py \
  upright_endpoint_98p808_26p776_madx_match_result_twiss.tfs \
  upright_endpoint_98p808_26p776_madx_match_result_plot.png \
  20
```

## Summary Requirements

For final reporting and HTML summaries, include:

- old keeper and new endpoint `BETX/BETY/ALFX/ALFY`;
- MAD-X `MATCH` final penalty and warning count;
- length, quadrupole count, maximum gradient in `T/m`;
- max `BETX/BETY` and max RMS beam sizes in meters;
- changed quadrupoles and gradient deltas in `T/m`;
- links to the reproducible match deck, static result deck, TFS, log, PNG plot, and HTML summary.

For the `98.80798387931307 / 26.775980892186183 m` retarget, the useful knob set was `q005`, `q006`, `q008`, and `q010`; it preserved the peak RMS size from the 10Q keeper.
