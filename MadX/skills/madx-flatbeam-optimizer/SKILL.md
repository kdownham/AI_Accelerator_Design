---
name: madx-flatbeam-optimizer
description: Use for MAD-X flat-beam transformer lattice work, especially solenoid/skew quadrupole matching, beta-function control, beam-size/aperture analysis, physical-unit strength checks, and optics plots for 5 TeV electron beam studies.
---

# MAD-X Flat-Beam Optimizer

Use this skill when working on the `MadX` project’s 5 TeV round-to-flat electron beam lattices.

## Default Priorities

1. Optimize for physical usefulness, not only endpoint matching.
2. Report maximum transverse beam size early, not only beta functions.
3. Keep magnet strengths in achievable physical units on every candidate.
4. Make plots match the user’s current review needs without waiting for repeated corrections.
5. Preserve known-good designs under stable filenames before starting a new optimization branch.
6. When a design branch is accepted or used as a reference, create a compact HTML summary if the user asks for documentation or comparison.

## Physical Assumptions

- Beam momentum: 5 TeV electron beam unless the user says otherwise.
- Beam rigidity: `Brho [T*m] = p [GeV/c] / 0.299792458`.
- Solenoid field: `B [T] = KS [1/m] * Brho`.
- Quadrupole or skew quadrupole gradient: `G [T/m] = K1 [1/m^2] * Brho`.
- If the user gives emittance in nm without further qualification, treat it as geometric RMS emittance in both planes and state that assumption.
- RMS beam size: `sigma_x = sqrt(BETX * eps_x)`, `sigma_y = sqrt(BETY * eps_y)`.

## What To Check Before Presenting A Candidate

- Final `BETX`, `BETY`, `ALFX`, `ALFY`.
- Peak `BETX`, peak `BETY`.
- Final and peak `sigma_x`, `sigma_y` using the current emittance assumption.
- Solenoid field in Tesla.
- Normal and skew quadrupole gradients in T/m.
- Total length and last active element location.
- MAD-X warnings, especially mode flips.

Do not describe a lattice as good if it only hits endpoint beta while peak beam size or mode stability is bad.

## Immediate Habits From Prior Iterations

- When the user asks for a cleaner match, include final `ALFX` and `ALFY` in the objective immediately; do not wait for a separate correction.
- When endpoint alphas are nonzero, try varying lattice length and rematching gradients before adding unnecessary complexity. A final drift can help only if both planes want compatible drift lengths.
- When aperture is the concern, optimize RMS beam size directly with the current emittance assumption. Do not use beta matching as a proxy.
- When a solution pushes gradients near a cap, run a conservative variant with a lower gradient cap and compare the beam-size penalty.
- If adding quadrupoles, use them to reduce peak envelope and gradient pressure, then rematch endpoint `BETX/BETY/ALFX/ALFY`; do not accept a smaller beam size with a loose endpoint match unless it is explicitly labeled as a trial.
- Keep the previous best design intact. Write new branches under descriptive prefixes such as `*_baseline`, `*_peak_reduced`, `*_g160`, or `*_trial`.
- Present a compact comparison table before calling a candidate useful: length, quadrupole count, final beta/alpha, max RMS beam size, max gradient, and warnings.
- For endpoint retargeting, document changed magnets and gradient deltas rather than only the final optics.

## Plot Defaults

For the flat-beam matching plots, default to:

- Top lattice survey band.
- Beta-function panel.
- Alpha-function panel with final `BETX`, `BETY`, `ALFX`, and `ALFY` printed on the plot.
- RMS beam-size panel in meters.
- No dispersion panel unless the user explicitly asks for dispersion.
- Strength annotations in physical units, with repeated quadrupoles grouped.
- Use `20 nm` emittance if that is the active user context.
- For HTML summaries, embed or link the generated PNG and include previous-vs-new endpoint optics, peak beam size, gradient changes, MAD-X penalty, and warnings.

If the user asks to “show the plot,” use a real image viewer directly, preferably `eog`, rather than relying on a generic opener.

## Optimization Strategy

Default to MAD-X's built-in `MATCH` command for future optics optimizations when the task can be expressed as variable magnet strengths/positions with Twiss constraints. Use `VARY` bounds from the user's physical strength limits, put `CONSTRAINT` targets directly on named markers/elements and the endpoint, and save both:

- a reproducible `*_madx_match.madx` deck containing the `MATCH` block and log;
- a static `*_madx_match_result.madx` deck with the matched strengths baked in.

Only use scipy, transfer-matrix scans, or other external optimizers as a helper layer when MAD-X `MATCH` cannot represent the objective cleanly, such as peak beam-size minimization, aperture max-envelope objectives, combinatorial layout changes, or broad seed searches. When using an external helper, verify the final candidate with MAD-X and consider a follow-up `MATCH` pass to polish local/end Twiss constraints.

When trying to reduce maximum transverse beam size:

1. Start from the best stable candidate, not from a random layout.
2. Add or retune normal focusing around the high-beta region before changing the endpoint target.
3. Penalize peak beam size directly in the objective, not just peak beta.
4. Keep endpoint matching constraints active enough that the optimizer cannot solve aperture by missing the target.
5. Compare candidates in a compact table with length, endpoint optics, peak beam size, strengths, and warnings.

For shortening the lattice:

- Do not just truncate drift unless the endpoint optics remain acceptable.
- Try moving the endpoint closer to the last active focusing/skew elements and rematch.
- A shorter lattice is not an improvement if it increases peak beam size, produces large final alphas, or introduces mode flips.

## Current Project Baselines

- Cleaner baseline: `multifamily_skew_twiss.tfs`, 900 m, no MAD-X warnings.
- Shorter comparison: `multifamily_skew_720m_twiss.tfs`, 720 m, worse aperture and mode-flip warnings.
- Existing plotting script: `plot_twiss.py`.
- Existing generator: `generate_flatbeam_lattice.py`.
- Existing optimizer wrapper: `optimize_multifamily_skew.py`.

When in doubt, regenerate outputs from scripts rather than hand-editing MAD-X decks.
