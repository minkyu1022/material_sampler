# Ni–Cr tasks for the JANUS reproduction server

## Goal

Extend the existing JANUS reproduction codebase from Cu–Ni to the **paper-faithful Ni–Cr baseline**.

This server should reproduce what JANUS actually did for Ni–Cr:

- separate FCC and BCC lattice-specific models;
- separate lattice-restricted reference MC controls;
- fixed-composition free-energy reconstruction;
- FCC/BCC free-energy alignment;
- JANUS Fig. 3b-style evaluation.

Do **not** implement the new 6-parameter flexible-cell target on this server.  
That belongs to the separate reference-MC server.

---

## 1. Use the same Ni–Cr interatomic potential as the paper

Use the Finnis–Sinclair-type Ni–Co–Cr potential used in JANUS, restricted to Ni and Cr.

Record:
- exact potential file/source;
- SHA256 hash;
- species ordering;
- units;
- cutoff.

Paper settings:

### FCC Ni–Cr
- conventional \(3\times3\times3\) FCC supercell;
- \(N=108\);
- cutoff 5.0 Å;
- \(P=0\).

### BCC Ni–Cr
- conventional \(4\times4\times4\) BCC supercell;
- \(N=128\);
- cutoff 5.3 Å;
- \(P=0\).

Cross-check energy, forces, virial/stress, and single-site substitution energies against an independent trusted implementation before production.

---

## 2. Reproduce the lattice-restricted fixed-composition reference MC

For each lattice family independently, implement the canonical composition ladder

\[
n=N_{\rm Cr}=0,\ldots,N.
\]

For each \((n,T)\):

- two walkers:
  - random solid solution,
  - phase-separated slab;
- 30,000 sweeps;
- 5,000 burn-in;
- collective displacement move;
- scalar log-volume move;
- fixed-composition Ni↔Cr pair exchange;
- temperature replica exchange at fixed composition.

Paper-stated temperature range:
- 7 isotherms spanning 600–1500 K per lattice.

Do not guess the following missing settings in any run labeled paper-faithful:

- exact 7 temperatures;
- pair-exchange attempts per sweep;
- exact canonical-ladder temperature-RE cadence.

Keep them configurable until Denis replies.

---

## 3. Reference-MC move definitions

### Collective displacement

\[
u' = u+\sigma_u\xi,
\qquad
\xi\sim\mathcal N(0,I).
\]

At \(P=0\),

\[
\log A=-\beta\Delta U.
\]

### Scalar log-volume

\[
v' = v+\sigma_v\xi,
\qquad
V=e^v.
\]

At \(P=0\),

\[
\log A=-\beta\Delta U+N\Delta v.
\]

### Fixed-composition pair exchange

Choose one Ni site and one Cr site uniformly and swap labels.

\[
\log A=-\beta\Delta U.
\]

### Step-size adaptation

During burn-in only:

- adapt displacement and volume proposal widths;
- target acceptance around 0.3;
- freeze proposal widths at production start;
- reset counters at production start.

### Temperature replica exchange

At fixed \(n\), use adjacent-temperature deterministic even–odd swaps.

At \(P=0\),

\[
\log A_{\rm swap}
=
(\beta_k-\beta_l)
\left[U(x_k)-U(x_l)\right].
\]

Keep the swap cadence configurable until confirmed.

---

## 4. Implement the composition-ladder free-energy estimator

For every neighboring pair \(n\leftrightarrow n+1\):

- from rung \(n\), evaluate Ni→Cr single-site substitution energies;
- from rung \(n+1\), evaluate Cr→Ni single-site substitution energies;
- keep atomic coordinates and cell fixed during these virtual substitutions;
- implement the paper's two-sided BAR estimator including combinatorial factors.

Save per edge:

- BAR estimate for \(G(n+1)-G(n)\);
- forward one-sided estimate;
- reverse one-sided estimate;
- forward/reverse discrepancy;
- overlap diagnostics;
- effective sample counts.

Then accumulate

\[
G(n)-G(0)
=
\sum_{m=0}^{n-1}
\left[G(m+1)-G(m)\right].
\]

Unit-test the BAR machinery on toy systems before using Ni–Cr data.

Do not replace this with naive integration of \(\Delta\mu(x)\).

---

## 5. FCC/BCC absolute alignment

The FCC and BCC ladders have unrelated additive free-energy constants.

Prepare and validate Frenkel–Ladd absolute references for:

- Ni-FCC;
- Cr-FCC;
- Ni-BCC;
- Cr-BCC.

Use an Einstein-crystal reference and thermodynamic integration as in the paper.

The purpose is only to determine the additive offsets needed to put

\[
G_{\rm FCC}(x,T)
\]

and

\[
G_{\rm BCC}(x,T)
\]

on the same absolute scale.

Do not compare the FCC and BCC branches thermodynamically before this alignment is validated.

---

## 6. Reproduce the paper's Ni–Cr observables

The main target is JANUS Fig. 3b-style evaluation.

At minimum reproduce:

1. \(G_{\rm mix}(x)\) at 1200 K for the FCC branch;
2. \(G_{\rm mix}(x)\) at 1200 K for the BCC branch;
3. their relative free-energy competition after absolute alignment;
4. the Ni-rich FCC / Cr-rich BCC coexistence compositions;
5. the resulting phase boundary/binodal over temperature.

Also report oracle-evaluation counts separately for:

- reference MC;
- JANUS training;
- JANUS inference/free-energy estimation.

---

## 7. JANUS fixed-composition model

After Denis replies, extend the existing JANUS code to the paper's fixed-composition Ni–Cr setting.

One JANUS model per lattice family:

- one FCC model;
- one BCC model.

Each model should amortize across:

- temperature;
- fixed composition \(n/N\).

The species channel must end with **exactly \(n\) Cr atoms** in every terminal sample.

Do not finalize this implementation until the missing constrained-reveal rule is confirmed.

---

## 8. Missing author-side details that block exact JANUS training

Wait for Denis on:

### Reference MC
- exact 7 temperatures;
- pair-exchange attempts per sweep;
- canonical-ladder temperature-RE cadence.

### JANUS fixed-composition training
- exact constrained masked-reveal algorithm enforcing exactly \(n\) Cr atoms;
- Ni–Cr continuous diffusion strengths;
- interpolant / gamma setting;
- clipping convention;
- actual \(T,n\) condition-sampling distribution.

Until those arrive:

- implement interfaces;
- write unit tests;
- run only engineering smoke tests;
- do not label the result an exact paper reproduction.

---

## 9. Important scope boundary

Do **not** implement on this server:

- a common FCC↔BCC BCT cell;
- a 6-parameter cell head;
- one-model FCC/BCC structural transitions;
- Bain-path umbrella sampling;
- HMC for the new flexible-cell reference;
- REUS for the new flexible-cell target.

Those are handled by the separate reference-MC server.

This server's job is:

\[
\boxed{
\text{paper-faithful JANUS Ni–Cr baseline}
}
\]

not the new method.

---

## 10. Deliverables

Return:

1. Ni–Cr potential/oracle validation report;
2. FCC reference-ladder control results;
3. BCC reference-ladder control results;
4. BAR edge diagnostics;
5. Frenkel–Ladd alignment report;
6. reconstructed \(G_{\rm FCC}(x,T)\) and \(G_{\rm BCC}(x,T)\);
7. Fig. 3b-style phase-boundary comparison;
8. exact list of any remaining author-dependent blockers;
9. JANUS FCC/BCC training configs used once the missing settings are confirmed.

Do not close the task merely because code runs.  
Close it only when the Ni–Cr paper baseline is quantitatively reproduced or a specific unresolved blocker is isolated.
