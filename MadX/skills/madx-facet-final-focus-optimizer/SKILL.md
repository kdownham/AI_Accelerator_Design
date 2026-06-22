---
name: madx-facet-final-focus-optimizer
description: Use for FACET MAD-X final-focus and dump-line optics optimization in /home/keegan/Codex/MadX/FACET, especially changing Q5FF-Q0FF and Q0D-Q2D strengths, raising or lowering beta at DEX20 markers, preserving endpoint optics, using MAD-X MATCH by default, and producing comparison plots.
---

# MAD-X FACET Final-Focus Optimizer

Use this skill for FACET optics work in `/home/keegan/Codex/MadX/FACET`.

## Default Workflow

1. Preserve the original lattice and known-good candidates. Write new attempts under descriptive names.
2. Convert Bmad-exported `lcavity` elements to length-preserving drifts before running MAD-X.
3. Default to MAD-X's built-in `MATCH` command for strength matching:
   - initialize variables from the original lattice or current candidate;
   - set `VARY` bounds from the user's gradient caps using `K1 = G[T/m] / Brho[T*m]`;
   - put `CONSTRAINT` targets directly on requested markers such as `DEX20_10`, `DEX20_11`, and `#e`;
   - include enough constraints for the number of varied variables, adding local alpha constraints when needed.
4. Save both outputs:
   - `*_madx_match.madx`, `*_madx_match.log`, and `*_madx_match_twiss.tfs` for the reproducible MATCH run;
   - `*_madx_match_result.madx`, `*_madx_match_result.log`, and `*_madx_match_result_twiss.tfs` with matched strengths baked in.
5. Regenerate the plot with the FACET plotting script and include the lattice survey, beta panel, alpha panel, marker values, endpoint values, and strength table.

## Validation Checklist

Before presenting the result, report:

- MAD-X final penalty from the `MATCH` summary.
- MAD-X warning count.
- `BETX/BETY/ALFX/ALFY` at requested markers.
- endpoint `BETX/BETY/ALFX/ALFY`.
- varied magnet gradients in `T/m`, checked against absolute caps.
- files created for the MATCH deck, static result deck, TFS outputs, logs, and PNG plots.

If MAD-X says the match command was ignored because there are more variables than constraints, add physically meaningful constraints instead of silently accepting the unchanged lattice.

## FACET Local-Beta Context

For the current FACET final-focus work, the common varied families are:

- `Q5FF`, `Q4FF`, `Q3FF`, `Q2FF`, `Q1FF`, `Q0FF`
- `Q0D`, `Q1D`, `Q2D`

Common review markers:

- `DEX20_10`: often used for local `BETY`.
- `DEX20_11`: often used for local `BETX`.
- `#e`: endpoint constraints and comparison.

Use external optimizers only as helpers for broad searches or objectives that MAD-X `MATCH` cannot express directly, such as max-envelope or aperture objectives. Verify and polish final optics with MAD-X.
