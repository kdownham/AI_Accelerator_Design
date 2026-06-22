---
name: madx-upright-transformer-iteration
description: Use for iterating the upright-quadrupole-only MAD-X flat-beam transformer designs in this workspace, especially preserving the current keeper design, retargeting endpoint beta/alpha with MAD-X MATCH, reducing maximum RMS beam size, varying lattice length or quadrupole positions, and comparing gradient-cap tradeoffs.
---

# MAD-X Upright Transformer Iteration

Use this skill for the upright-quadrupole-only design branch in `/home/keegan/Codex/MadX`.
Also use `skills/madx-flatbeam-optimizer/SKILL.md` for general physical checks and plot defaults.

## Current Keeper Designs

Preserve these unless the user explicitly asks to overwrite them:

- `upright_length_alpha_8q.*`: clean zero-alpha baseline.
- `upright_peak_reduced_8q.*`: smallest accepted 8Q peak beam size; one gradient near `170 T/m`.
- `upright_peak_reduced_8q_g160.*`: more conservative gradient-cap version near `160 T/m`.
- `upright_peak_reduced_10q.*`: aperture-promising trial, but endpoint match is looser than the 8Q keepers.
- `upright_peak_reduced_10q_matched.*`: current best 10Q aperture keeper, endpoint matched with max RMS beam size near `0.0235082355 m` at 20 nm and max gradient `160 T/m`.
- `upright_endpoint_98p808_26p776_*`: endpoint-retargeted reference from the 10Q keeper, matched to `BETX=98.80798387931307 m`, `BETY=26.775980892186183 m`, `ALFX=0`, `ALFY=0` with unchanged peak envelope.

Write new attempts under new descriptive prefixes. Never silently replace the keeper branch.

## Default Workflow

1. Start from the best stable candidate closest to the user’s stated goal.
2. Read metrics from TFS/log files, not from memory:
   - final `BETX`, `BETY`, `ALFX`, `ALFY`
   - max `BETX`, max `BETY`
   - max RMS `sigma_x`, `sigma_y` using 20 nm geometric RMS emittance unless changed
   - max quadrupole gradient in `T/m`
   - MAD-X warning count
3. If final alphas need improvement, vary length and gradients first.
   - Use a least-squares endpoint match against `BETX=80 m`, `BETY=30 m`, `ALFX=0`, `ALFY=0`.
   - Check whether simple final drift can zero both alphas. If the x/y required drift lengths differ substantially, optimize lattice length and gradients instead.
4. If the user changes only endpoint `BETX/BETY`, default to MAD-X `MATCH` from the best keeper before changing the layout.
   - Keep the existing positions and section length unless asked otherwise.
   - Constrain endpoint `BETX`, `BETY`, `ALFX=0`, and `ALFY=0`.
   - Vary the smallest useful quadrupole knob set first. With 10 quadrupoles and four endpoint constraints, search four-knob combinations rather than varying all ten at once.
   - Select by physical usefulness after convergence: prefer unchanged or reduced peak RMS size, unchanged max gradient cap, zero warnings, and small quadrupole changes. Do not pick solely by the smallest numerical penalty if it damages aperture.
   - Save both a reproducible `*_madx_match.madx` deck and a static `*_madx_match_result.madx` deck.
5. If maximum beam size needs improvement, optimize peak RMS beam size directly with equality constraints on final beta/alpha.
   - Do not accept lower peak size if endpoint `BETX/BETY/ALFX/ALFY` drift materially.
   - Compare a high-performance candidate against a conservative-gradient candidate.
6. If gradients hit the cap, try one of:
   - a lower gradient cap rerun,
   - additional upright quadrupole slots near high-beta regions,
   - position variation around the known stable layout.
7. Verify each serious candidate with MAD-X, regenerate the plot, and compare against the keeper in a compact table.

## Useful Scripts

- `match_upright_length_lsq.py`: rematches length and gradients to endpoint beta/alpha.
- `match_upright_endpoint_target.py`: MAD-X `MATCH` retarget of endpoint `BETX/BETY/ALFX/ALFY` from an upright keeper; searches valid four-knob sets, saves reproducible and static result decks.
- `optimize_upright_peak_size.py`: constrained peak RMS beam-size reduction.
- `optimize_upright_matrix_positions.py`: fast transfer-matrix exploration of length, position, and gradient changes before MAD-X verification.
- `plot_twiss.py`: standard plot with survey, beta, alpha, beam size, final endpoint annotation, and strength summary.

Prefer these scripts over hand-editing large MAD-X decks.

## Acceptance Checklist

Before presenting a candidate as an improvement, report:

- File prefix and whether it is a keeper or trial.
- Total lattice length and quadrupole count.
- Final `BETX/BETY` and `ALFX/ALFY`.
- Max RMS beam size in meters for the active emittance.
- Max gradient in `T/m` and whether it approaches `160-170 T/m`.
- MAD-X warning count.
- Tradeoff versus the current keeper.
- For endpoint retargets, include the exact target `BETX/BETY`, MAD-X final penalty, changed quadrupoles, gradient deltas in `T/m`, and whether max RMS beam size changed.

If a run stops by iteration limit but MAD-X verification is good, say that clearly: the candidate may still be useful, but the optimizer did not formally converge.
