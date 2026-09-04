# Ni–Cr Unified BCT2D Model — Training and Final Evaluation Guideline

## Goal

Your role is to train and evaluate the **unified neural sampler**.

Do not run production reference MC.

Do not wait for reference MC before training.

Do not use production reference-MC samples as training data.

The training algorithm remains self-bootstrapped.

---

## 1. Training loop

Production training should follow:

model rollout
→ EAM target / score / path-weight evaluation
→ replay buffer
→ model update
→ new rollout

The replay buffer contains the model’s own generated states / trajectories, not ground-truth reference-MC samples.

Reference MC is independent and used only after or alongside training for evaluation.

---

## 2. Fixed model definition

Use the current unified BCT2D design:

- N = 128
- common BCT reference sites
- exact-n fixed-cardinality discrete species channel
- fractional displacement u
- 2D cell variables a,c
- normalized cell latent
- linear interpolant
- gamma = 0
- M = 100
- g^2(T) = 0.02^2 * T/750 as the baseline
- corrected target-density/Jacobian treatment
- corrected forward/backward path probabilities
- float64 log-weight accumulation

Do not regress these validated choices silently.

---

## 3. Training diagnostics

Track continuously:

- training loss
- replay-buffer size
- valid-domain fraction
- exact terminal Cr count
- RMS fractional u
- site-anchoring ratio
- q_Bain or c/a coverage
- BCC-mode coverage
- FCC-mode coverage
- std(logW)
- ESS per condition
- energy distribution
- cell distribution

Never pool ESS across different (T,n) conditions.

---

## 4. Structural mode-collapse gate

The model must not be declared successful if it learns only one structural basin.

At each relevant (T,x_Cr), track:

- fraction of samples in BCC basin
- fraction in FCC basin
- intermediate-BCT probability
- q_Bain histogram

If one basin is systematically missing, first use model-side fixes:

- broader cell prior
- symmetric BCC/FCC mixture prior
- balanced seeding
- replay-buffer balancing
- curriculum

Do not inject reference-MC samples into training to repair mode collapse.

---

## 5. Final evaluation against independent reference

Once validated reference outputs arrive, compare model versus reference on the same conditions.

### Primary headline metrics

1. G_mix(x_Cr,T)
2. P(c/a | x_Cr,T) or P(q_Bain | x_Cr,T)
3. BCC/FCC basin populations
4. phase boundary / coexistence region

These are the main scientific results.

The key new result beyond JANUS is direct matching of the structural two-mode distribution in one common state space.

---

## 6. Distribution-level comparisons

For each condition compare:

- q_Bain / c/a distribution
- energy/atom distribution
- volume/atom distribution
- atomic RMS displacement distribution
- chemical-order distribution
- RDF / species-resolved pair correlation when available

Use quantitative distribution distances where appropriate, for example:

- Wasserstein distance
- histogram / CDF error
- basin-population absolute error

Do not rely only on visual agreement.

---

## 7. Sampler-quality metrics

Report sampler quality separately from physical accuracy:

- ESS
- std(logW)
- valid fraction
- oracle NFE per effective sample
- rollout cost
- throughput / wall-clock
- success rate across random seeds
- BCC/FCC mode coverage across seeds

This is important because the reference MC requires enhanced sampling, while the neural sampler is intended to amortize the difficult multimodal target.

---

## 8. JANUS comparison

The final comparison should make the methodological difference explicit.

JANUS Ni–Cr:

- separate FCC and BCC models
- separate free-energy branches
- later thermodynamic alignment

Unified BCT2D model:

- one N=128 common BCT state space
- one sampler
- BCC, BCT intermediate, FCC-equivalent all represented directly
- direct P(c/a) / basin-population evaluation

Do not claim superiority only from one phase-boundary plot.

---

## 9. Final figures / tables

At minimum prepare:

### Figure A
G_mix vs x_Cr at representative temperatures:
- reference
- unified model
- JANUS baseline where available

### Figure B
P(c/a | T,x_Cr) or P(q_Bain | T,x_Cr) at representative conditions:
- reference
- unified model

### Figure C
Phase diagram / coexistence boundary:
- reference
- unified model
- JANUS baseline where available

### Table
Sampler-quality metrics:
- ESS
- std(logW)
- valid fraction
- NFE / effective sample
- wall-clock / throughput
- mode coverage

Optional supplementary figures:
- energy distributions
- volume distributions
- atomic RMS displacement
- chemical-order metrics
- RDF / pair correlations

---

## 10. Final success condition

Do not declare success from smoke tests or terminal exact-n alone.

Success requires:

- exact composition
- valid local u regime
- both BCC and FCC structural modes represented where the target requires them
- acceptable q_Bain / c/a distribution agreement
- acceptable thermodynamic agreement
- phase-boundary agreement
- non-catastrophic path-weight degeneracy
- reproducibility across seeds

Reference-MC completion is not a blocker for training, but it is required for the final scientific accuracy claim.
