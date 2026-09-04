# Ni–Cr JANUS Reproduction: Hyperparameter / Provenance Instructions

## Current status

Do **not** describe the Ni–Cr hyperparameters as fully optimized or broadly swept.

The current configuration should be described as:

- largely inherited from the validated Cu–Ni reproduction,
- modified where the Ni–Cr setup requires it,
- with only limited local tuning performed so far,
- and with several Ni–Cr-specific prior/oracle choices still provisional.

---

## 1. Validated / inherited from Cu–Ni

Use the following Cu–Ni reproduced best configuration as the default starting point unless a Ni–Cr-specific reason requires changing it:

- PaiNN features: `64`
- interaction layers: `4`
- RBFs: `16`
- batch size: `96`
- optimizer: AdamW
- learning rate: `1e-4`
- weight decay: `1e-3`
- warmup: `5,000`
- minimum learning rate: `1e-5`
- replay capacity: `5,000`
- gradient updates per round: `500`
- gradient clip: `100`
- target-score clipping:
  - displacement `u`: `100`
  - volume `v`: `1000`
- rollout clipping:
  - velocity: `0.1`
  - score: `1000`
- continuous/discrete objectives: `TSM + SCE`
- discrete loss weight: `2`
- bf16: off

Treat these as **validated inherited defaults**, not as Ni–Cr-specific optima.

---

## 2. Ni–Cr-specific settings already fixed

The following changes/settings are specific to the Ni–Cr reproduction:

- FCC:
  - `N = 108`
  - `M = 108`
- BCC:
  - `N = 128`
  - `M = 128`
- discrete reveal:
  - one-site reveal
  - exact-`n` fixed-composition boundary quota
- fresh terminals:
  - `500 / round`
- temperature range:
  - `600–1500 K`
- retain the JANUS temperature scaling form
  \[
  g_c(T) \propto \sqrt{T/750}
  \]
  for each continuous channel \(c\).

These should be separated in provenance from inherited Cu–Ni defaults.

---

## 3. Diffusion scales \(g_u\) and \(g_v\): do **not** force a shared scale

The current smoke test selected a common multiplicative scale of `0.5` because it was substantially more stable than `1.0` in terminal energy, RMS displacement, and log-weight behavior, while `0.25` was a smaller but more conservative stable value.

However:

> **Do not assume that the same multiplicative scale must be used for the displacement channel \(u\) and the volume channel \(v\).**

It is mathematically valid to use separate channel-wise diffusion strengths,

\[
g_u(T)
=
c_u\,g_{u,\mathrm{base}}(T),
\qquad
g_v(T)
=
c_v\,g_{v,\mathrm{base}}(T),
\]

with

\[
c_u \neq c_v,
\]

provided that the implementation uses the corresponding channel-specific \(g_c\) **consistently everywhere** it appears.

In particular, if \(g_u\) and \(g_v\) are tuned separately, verify consistency in:

- forward / rollout SDEs,
- velocity and score parameterization,
- target-score labels,
- path-weight / log-weight terms,
- any \(g^2 s\) drift contribution,
- temperature scaling,
- clipping or normalization logic that depends on channel scale.

There is no mathematical requirement that the displacement and volume channels share one numerical diffusion scale. They are different coordinates with different physical and numerical scales.

### Recommended tuning

Do a small channel-wise local sweep rather than a broad global sweep.

For example, starting around the current stable common choice:

\[
(c_u,c_v)
\in
\{0.25,0.5,1.0\}
\times
\{0.25,0.5,1.0\},
\]

or a cheaper subset centered around `0.5`.

Prioritize diagnostics separately by channel:

**For \(u\):**
- RMS physical displacement
- displacement target magnitude
- displacement score/velocity clipping frequency
- atomic clashes / unstable relaxation
- contribution to log-weight variance

**For \(v\):**
- volume / log-volume distribution
- phase-dependent equilibrium volume
- volume target magnitude
- score/velocity clipping frequency
- contribution to log-weight variance

Also monitor jointly:
- terminal energy
- `std(log W)`
- ESS
- training loss stability
- phase-wise failure / collapse

Do not report `0.5` as a final Ni–Cr optimum unless this channel-wise local check has been performed.

---

## 4. Displacement prior width: still provisional

Current statement:

> “Ni–Cr phase-specific displacement prior width is inherited from Cu–Ni.”

This is too strong.

Use instead:

> **The Cu–Ni displacement width is only an initial fallback / inherited baseline. Ni–Cr FCC/BCC phase-specific calibration is still pending.**

The displacement prior width is system- and phase-dependent. A preferable Ni–Cr calibration is based on:

- phase-specific EAM Hessians / quasi-harmonic estimates, or
- a short, explicitly documented phase-restricted thermal calibration.

Do not present the inherited Cu–Ni width as paper-confirmed or Ni–Cr-optimal.

---

## 5. Volume prior: phase-specific calibration is acceptable, but mark it as reconstruction

A short FCC/BCC-specific volume calibration is reasonable.

However, distinguish:

- **paper-supported structure of the prior**, versus
- **our reconstruction of numerical width / calibration details**.

The exact log-volume prior width fitting recipe is not fully published.

Therefore record the selected phase-specific volume prior parameters and the calibration procedure explicitly in provenance.

Do not use held-out production reference data to tune this prior if the reproduction is intended to remain self-bootstrapped from the EAM/oracle.

---

## 6. Target EAM cutoff vs JANUS graph cutoff

Do **not** write:

> “We use 6.0 Å although the paper uses 5.0 / 5.3 Å.”

That wording incorrectly conflates two different cutoffs.

Current validated working convention:

- **target EAM energy / forces:** native potential range `6.0 Å`
- **JANUS neighbor graph cutoff:**
  - FCC: `5.0 Å`
  - BCC: `5.3 Å`

The cutoff diagnostic found that hard-truncating the target EAM at the paper-reported `5.0 / 5.3 Å` did **not** improve agreement with the published pure-phase energy anchors relative to the native `6.0 Å` oracle.

Therefore the correct provenance statement is:

> **The paper-reported 5.0 Å (FCC) / 5.3 Å (BCC) values are retained as JANUS neighbor-graph cutoffs. The target EAM oracle currently uses the potential's native 6.0 Å range, supported by the cutoff diagnostic. Author confirmation of the target-oracle cutoff convention remains pending.**

Do not call native `6.0 Å` merely a speculative candidate unless new evidence contradicts the existing diagnostic.

---

## 7. What has actually been tuned so far

At present, the Ni–Cr reproduction should be described as:

- architecture: inherited from validated Cu–Ni reproduction
- optimizer / training cadence: inherited
- replay / update cadence: inherited except for Ni–Cr paper-specific fresh-terminal count
- discrete process: changed to exact-`n` one-site reveal
- system size / temperatures: Ni–Cr paper settings
- continuous diffusion scale:
  - only a **small local stability test** has been performed
  - common scale `0.5` is the current working value
  - channel-wise \(g_u\) vs \(g_v\) tuning is still recommended
- prior widths: not yet fully Ni–Cr-calibrated
- target-oracle cutoff: native `6.0 Å` working convention supported by diagnostic

Do **not** claim a comprehensive hyperparameter sweep.

---

## 8. Requested next actions

Before treating the Ni–Cr configuration as production-final:

1. Calibrate or at least sanity-check phase-specific displacement prior widths.
2. Record the exact phase-specific volume-prior calibration procedure.
3. Run a small **separate \(g_u\) / \(g_v\) local sweep**.
4. Compare each candidate using:
   - terminal energy,
   - RMS physical displacement,
   - volume statistics,
   - `std(log W)`,
   - ESS,
   - clipping frequency,
   - training stability.
5. Preserve the distinction:
   - target EAM cutoff = native `6.0 Å`
   - graph cutoff = `5.0 Å` FCC / `5.3 Å` BCC.
6. Keep all remaining author-unconfirmed choices explicitly marked in provenance.
7. Do not broaden the sweep unnecessarily: start from the validated Cu–Ni configuration and tune only Ni–Cr-specific axes that show evidence of sensitivity.

---

## Short summary for reports

> The Ni–Cr configuration is not the result of a broad hyperparameter sweep. Architecture, optimizer, replay, and most training settings are inherited from the validated Cu–Ni reproduction. Ni–Cr-specific changes follow the reported system sizes, fixed-composition one-site reveal process, temperature range, and fresh-terminal cadence. A limited diffusion-scale smoke test currently favors a scale near 0.5, but \(g_u\) and \(g_v\) should be treated as independent channel-wise hyperparameters and locally tuned separately. Displacement and volume priors remain partially reconstruction-dependent. The target EAM oracle uses the native 6.0 Å potential range, while 5.0 Å (FCC) / 5.3 Å (BCC) are retained as JANUS neighbor-graph cutoffs.
