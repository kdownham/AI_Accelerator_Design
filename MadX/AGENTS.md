# AGENTS.md for the MAD-X lattice workspace

- This directory is for MAD-X accelerator-lattice experiments and optics plotting.
- Treat numerical MAD-X matches as starting points, not accepted designs, until the magnet fields, gradients, apertures, beta functions, and section length are physically plausible.
- For 5 TeV electron-beam work, always convert normalized strengths to magnetic units using beam rigidity before judging a solution:
  - `B rho [T*m] = p [GeV/c] / 0.299792458`
  - solenoid field estimate: `B [T] = KS [1/m] * B rho [T*m]`
  - quadrupole gradient estimate: `G [T/m] = K1 [1/m^2] * B rho [T*m]`
- Prefer present-day achievable magnet strengths over purely mathematical matches. As a first-pass conceptual cap, keep solenoids near single-digit Tesla and quadrupole/skew-quadrupole gradients around order `100 T/m`; call out anything approaching or exceeding about `200 T/m`.
- Watch aperture implications. Large beta functions imply large beam envelopes for any nonzero emittance, so do not accept solutions with beta functions that grow by orders of magnitude unless the user explicitly asks for a mathematical toy model.
- When beta functions get too large, first add a modest number of normal focusing quadrupoles or a simple correction channel, then rematch the solenoid/skew section. Compare peak `BETX` and `BETY`, not just final endpoint values.
- Preserve the user’s target optics in summaries: starting round beam beta, final `BETX/BETY`, section length, magnet fields/gradients, peak beta functions, and any MAD-X warnings such as mode flips.
- Keep lattice files reproducible. Prefer generator scripts for repetitive marker/sliced-element lattices, and regenerate MAD-X input, TFS output, and plots from those scripts rather than hand-editing large repeated blocks.
- Include dense markers when plotting optics evolution, but avoid placing markers inside thick MAD-X elements unless the element is sliced so the sequence remains valid.
- Plots should show both optics curves and a top survey band for solenoids, normal quadrupoles, skew quadrupoles, and markers. Strength annotations should be in physical units, not only normalized MAD-X units.
- If a match only works because the fields are unrealistic, state that directly and keep searching for a more physical layout before presenting it as a useful design.
- For future flat-beam transformer optimization, consult `skills/madx-flatbeam-optimizer/SKILL.md` before editing or running scans. Apply the plot defaults there immediately: include RMS beam size in meters, omit dispersion unless requested, group repeated strengths, and compare maximum beam size before accepting a candidate.
