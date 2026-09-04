# JANUS Ni–Cr Model Reproduction: Hamiltonian-Correction, Provenance Audit, and Retraining Validation Prompt

We have a major update from the reference-MC reproduction that changes how the current neural-model results must be interpreted.

Do **not** continue treating the existing Ni–Cr checkpoints as paper-faithful JANUS reproduction until their Hamiltonian provenance is audited.

The immediate goals are:

1. determine exactly which Hamiltonian every existing checkpoint was trained and evaluated against;
2. rebuild the model pipeline so that **all Hamiltonian-dependent components are internally consistent with the current paper-faithful working target**;
3. re-evaluate whether the severe path-weight collapse and long-training degradation remain after this correction;
4. revalidate, but do **not blindly retune**, the continuous diffusion strengths \(g_u\) and \(g_v\).

---

# 0. New reference-side result that must be incorporated

Reference-MC diagnostics have established the following.

## Native-6.0 reference run

The completed reference ladders used the native 6.0 Å cutoff of the selected Ni–Co–Cr EAM file.

Those trajectories are internally valid for the Hamiltonian they actually sampled.

## Paper-reported interaction ranges

JANUS SI states:

- Ni–Cr FCC: 5.0 Å interaction cutoff
- Ni–Cr BCC: 5.3 Å interaction cutoff

Therefore the original reproduction and the paper were not using the same target Hamiltonian.

## Free-energy evidence

Native-6.0 -> abrupt candidate-cutoff free-energy perturbation gives, at 1200 K:

- FCC visible-branch RMSE:
  35.68 -> 7.71 meV/atom
- BCC visible-branch RMSE:
  7.10 -> 0.46 meV/atom

Thus the cutoff mismatch removes most of the Fig. 3b discrepancy at the **free-energy level**, not merely at the mean-energy level.

The exact author cutoff operation is not yet confirmed. The current working candidate is:

- FCC: abrupt/header 5.0 Å
- BCC: abrupt/header 5.3 Å

This is a **provisional reproduction Hamiltonian**, not yet claimed to be the exact author artifact.

## Reference-side caveat

Some FEP rungs have poor overlap and the first direct candidate pilots were not equilibrated, so exact quantitative reproduction is not yet certified.

However, there is already enough evidence to conclude:

> Do not continue using native 6.0 Å as the paper-faithful Ni–Cr target.

Treat old native-6.0 results as control-Hamiltonian results.

---

# 1. Existing neural-model observation that must be reinterpreted

At 1200 K, 256 unweighted samples/condition showed strong checkpoint dependence.

## FCC

- old120:
  - std(log W): about 207–254
  - RMS displacement \(u\): about 0.037
  - volume/atom: about 12.5–13.2
- round24:
  - std(log W): about 12.6–15.9
  - RMS \(u\): about 0.019–0.021
  - volume/atom: about 11.0–12.0
- new120:
  - std(log W): about 147–194
  - RMS \(u\): about 0.035–0.038
  - volume/atom: about 12.8–13.6

## BCC

- old120:
  - std(log W): about 343–486
  - RMS \(u\): about 0.042–0.043
  - volume/atom: about 14.5–15.2
- round24:
  - std(log W): about 24–34
  - RMS \(u\): about 0.018–0.019
  - volume/atom: about 11.6–11.8
- new120:
  - std(log W): about 147–354
  - RMS \(u\): about 0.028–0.029
  - volume/atom: about 12.5

This establishes a real **long-training degradation / non-monotonicity**: the late checkpoints are dramatically worse proposals than round24 under the current diagnostic.

However:

> ESS \(\approx 1\) at round24 is **not by itself evidence of a path-weight implementation bug**.

Even std(log W) of 13–34 is already enormous enough for importance weights to collapse severely with only 256 samples.

Therefore do not classify `(path-weight bug)` merely from low ESS. The implementation must be tested independently.

---

# STEP 1. Build a checkpoint-by-checkpoint Hamiltonian provenance table

Before any new training, audit `old120`, `round24`, `new120`, and every currently running/recent Ni–Cr model.

For each FCC/BCC checkpoint identify exactly which cutoff/Hamiltonian was used for each of the following:

1. terminal energy labels;
2. force labels for displacement score training;
3. virial / volume-score labels;
4. all single-site substitution energies used for discrete soft labels;
5. energy used during self-generated rollout labeling;
6. terminal target-density term in path-weight evaluation;
7. any energy term used in free-energy inference;
8. network graph cutoff;
9. volume-prior calibration;
10. displacement-prior / Hessian calibration;
11. transition-line estimate \(\Delta\mu_c(T)\);
12. composition-conditioned prior parameters;
13. any cached/replay-buffer labels inherited from an older Hamiltonian;
14. any reference values copied from paper or another run.

Return:

| checkpoint | component | FCC cutoff/Hamiltonian | BCC cutoff/Hamiltonian | source/config/code | internally consistent? | paper-working-target consistent? |
|---|---|---|---|---|---|---|

Do not infer from filenames. Trace actual configs, potential objects, stored metadata, and code paths.

---

# STEP 2. Classify the existing checkpoints correctly

After the provenance audit, assign each checkpoint to one of:

### A. Self-consistent native-6.0 control
Training labels, priors/calibration, graph and inference all consistently correspond to native 6.0 Å.

This checkpoint is **not paper-faithful**, but remains useful as a control for training stability.

### B. Mixed-Hamiltonian checkpoint
Different parts of the pipeline use incompatible Hamiltonians/cutoffs, e.g.:

- graph cutoff 5.0/5.3 but labels from 6.0;
- priors calibrated under one Hamiltonian and labels under another;
- replay buffer contains stale labels from another Hamiltonian;
- path-weight target uses a different cutoff from training.

These checkpoints are invalid for clean methodological diagnosis until the mismatch is removed.

### C. Candidate paper-Hamiltonian checkpoint
All Hamiltonian-dependent pieces use the current provisional candidate:

- FCC 5.0 Å
- BCC 5.3 Å

Only category C should be used to judge the current paper-faithful reproduction.

---

# STEP 3. Make the entire candidate-Hamiltonian pipeline internally consistent

For the next candidate model, do not change only the terminal energy oracle.

Recompute or regenerate **all quantities that depend on the target Hamiltonian**.

At minimum:

## Labels

Use the same candidate Hamiltonian for:

- \(U\),
- forces \(-\nabla_u U\),
- virial / volume derivative,
- all single-site substitution energies,
- heat-bath soft labels.

## Replay buffer

Do not reuse stale oracle labels computed under 6.0 Å.

Either:

1. discard the old replay buffer, or
2. re-evaluate every retained terminal with the candidate Hamiltonian and rebuild every energy-dependent label.

State explicitly which option is used.

## Physics-informed priors

Recalibrate under the candidate Hamiltonian:

- conditioned volume prior \(\hat V_{\rm atom}(c,T)\),
- excess-volume term,
- thermal-expansion fit,
- displacement prior width / Hessian-derived calibration.

Do not silently reuse 6.0-derived calibration.

## Thermodynamic conditioning

Re-estimate any Hamiltonian-derived transition-line or chemical-potential conditioning quantities, including:

\[
\Delta\mu_c(T)
\]

if the implementation derives it from potential energies / pilot data.

## Graph

Use the phase-appropriate interaction range consistently:

- FCC graph cutoff 5.0 Å
- BCC graph cutoff 5.3 Å

unless the exact author convention later demonstrates otherwise.

## Path weights

The target terminal density and every Hamiltonian-dependent term in inference must use the same candidate Hamiltonian as training.

No 6.0 target term may remain in a 5.0/5.3 candidate evaluation.

---

# STEP 4. Do not assume the current \(g\) scales remain optimal; audit them first

JANUS treats the continuous diffusion strength as a free inference/training parameter.

The paper uses **channel-specific diffusion strengths** \(g_c(t)\), \(c\in\{u,v\}\), with temperature scaling

\[
g_c(t;T)=g_c(t)\sqrt{T/T_{\rm ref}}.
\]

A change in cutoff does **not mathematically require** a different \(g\) merely to preserve the target marginal in the ideal continuous theory.

Therefore:

> Do not automatically change \(g_u\) or \(g_v\) just because the cutoff changed.

However, the target Hamiltonian change alters:

- force magnitudes,
- virial/volume-score magnitudes,
- equilibrium displacement scale,
- equilibrium volume scale,
- prior calibration,
- learned score accuracy,

and \(g\) directly affects finite-step stochastic integration and the variance of continuous forward/backward path weights.

So the existing \(g_u,g_v\) values must be **revalidated** after Hamiltonian correction.

---

# STEP 5. Audit the current \(g_u\) and \(g_v\) implementation

Report separately for displacement and volume:

1. exact functional schedule \(g_u(t)\);
2. exact functional schedule \(g_v(t)\);
3. amplitude / scale parameters;
4. temperature scaling;
5. \(T_{\rm ref}\);
6. whether FCC and BCC share values;
7. whether old120 / round24 / new120 used the same values;
8. how these values were originally selected;
9. whether they were inherited from Cu–Ni or another system;
10. all places where \(g_c\) appears:
    - forward SDE,
    - score-corrected drift \(g_c^2s_c\),
    - stochastic variance,
    - backward transition,
    - Gaussian path-weight density.

Verify that the same \(g_c\) convention is used consistently in all these locations.

Return:

| channel | schedule | amplitude | T scaling | origin | training use | inference use | path-weight use | consistent? |
|---|---|---:|---|---|---|---|---|---|

Do not force \(g_u=g_v\). They are separate physical channels and should be diagnosed separately.

---

# STEP 6. Revalidate \(g_u,g_v\) under the corrected Hamiltonian before expensive long training

First train a **short candidate-Hamiltonian model** using the current \(g_u,g_v\) values as the baseline.

Use early checkpoints, not only a final long run.

Then evaluate at representative 1200 K compositions:

\[
x_{\rm Cr}\approx0.25,\ 0.50,\ 0.75
\]

for both FCC and BCC.

Measure:

1. raw terminal energy distribution;
2. volume distribution;
3. RMS displacement;
4. local chemical-order statistic;
5. std(log W);
6. ESS;
7. log-weight decomposition:
   - target/prior term,
   - discrete term,
   - displacement forward/backward term,
   - volume forward/backward term;
8. numerical integration stability;
9. fraction of pathological configurations, if any.

Compare against the candidate-Hamiltonian direct/reference data wherever validated.

---

# STEP 7. Only tune \(g\) if the corrected-Hamiltonian baseline shows a reason to

If the current \(g_u,g_v\) work well after Hamiltonian correction, **keep them**.

Do not tune just to change a hyperparameter.

Trigger a \(g\)-scale search only if one or more of the following occurs:

- continuous path-weight terms dominate log-weight variance;
- displacement trajectories are too noisy/unstable;
- volume trajectories are too noisy/unstable;
- raw terminal \(u\) or \(V\) is poor despite reasonable learned fields;
- a channel has dramatically worse forward/backward consistency;
- ESS is strongly sensitive to continuous path contribution.

If tuning is triggered, tune \(g_u\) and \(g_v\) **independently**.

Use multiplicative perturbations around the current baseline schedule rather than inventing a new schedule immediately.

For example, a small diagnostic grid may use relative amplitudes such as:

\[
m_c\in\{0.5,\ 1.0,\ 1.5,\ 2.0\},
\qquad
g_c^{\rm test}(t)=m_c g_c^{\rm baseline}(t),
\]

but reduce the grid if compute is expensive.

This grid is only a suggested diagnostic; do not treat these exact multipliers as paper values.

### Selection criterion

Do **not** choose \(g\) by fitting Fig. 3b.

Choose based on sampler quality:

1. improved raw terminal agreement with the candidate target;
2. lower continuous path-weight variance;
3. improved ESS;
4. stable Euler–Maruyama trajectories;
5. no degradation of volume/displacement observables;
6. forward/backward consistency.

Then lock \(g_u,g_v\) before the long training comparison.

---

# STEP 8. Diagnose long-training degradation under the corrected Hamiltonian

The old results show that round120 can be much worse than round24.

We must determine whether this degradation persists after the Hamiltonian pipeline is corrected.

For the candidate-Hamiltonian training, save/evaluate at least:

- very early checkpoint,
- around round 12,
- around round 24,
- around round 48,
- around round 72,
- around round 96,
- around round 120,

or the closest available schedule.

For each checkpoint evaluate the same fixed diagnostic set.

Track:

\[
\operatorname{std}(\log W),
\quad
{\rm ESS},
\quad
U,
\quad
V,
\quad
{\rm RMS}(u),
\quad
{\rm SRO},
\]

and log-weight components.

Do not select the final checkpoint simply because it is the latest.

Implement a checkpoint-selection criterion based on held-out/self-generated sampler diagnostics.

The main question is:

> Does the candidate-Hamiltonian model improve toward the target and then degrade with continued outer-loop/replay training?

If yes, diagnose replay-buffer age, outer-loop update size, buffer replacement ratio, model drift, and stale/off-policy terminal distribution.

---

# STEP 9. Explicitly compare old native-6.0 controls against corrected candidate models

Use the exact same:

- 1200 K conditions,
- compositions,
- 256-sample diagnostic first,
- integration grid,
- seeds where possible,
- plotting code,
- path-weight decomposition code.

Compare:

1. native-6.0 round24;
2. native-6.0 round120;
3. candidate-Hamiltonian early/best checkpoint;
4. candidate-Hamiltonian round120 if trained that far.

This comparison should answer separately:

### Question A
How much of the Fig. 3b / raw-terminal discrepancy was caused by the Hamiltonian mismatch?

### Question B
How much of the ESS collapse remains even after fixing the Hamiltonian?

### Question C
Does long-training degradation persist?

Do not merge these into a single “model failed” label.

---

# STEP 10. Reassess the path-weight bug hypothesis only after Hamiltonian correction

Do continue controlled continuous-only / discrete-only tests, but interpret them properly.

Low ESS alone is not evidence of an implementation bug.

The path-weight implementation becomes a primary suspect only if, under a corrected and reasonably accurate proposal:

- analytically controlled tests fail;
- continuous Gaussian ratios disagree with independent calculations;
- discrete autoregressive probability is inconsistent;
- near-target samples still produce implausibly huge path-weight variance;
- changing integration resolution produces behavior inconsistent with the equations.

Run:

1. continuous-only controlled test;
2. discrete-only controlled test;
3. near-identity / known-target test if available;
4. float64 numerical-stability check;
5. forward/backward density-ratio unit tests.

Report pass/fail independently from model quality.

---

# STEP 11. Final free-energy evaluation only after sampler diagnostics pass

Do not jump directly from retraining to Fig. 3b.

First require:

1. reasonable raw terminal agreement with candidate reference;
2. stable checkpoint behavior;
3. understood log-weight variance;
4. no known path-weight implementation failure.

Then run the neural fixed-composition weighted-BAR pipeline.

For every rung report:

- within-rung ESS;
- std(log W);
- forward/reverse one-sided edge estimates;
- BAR estimate;
- edge overlap;
- uncertainty;
- reliability flag.

Only then reconstruct candidate-model \(G_{\rm mix}\).

---

# STEP 12. Respect the provisional status of the abrupt cutoff

For now use:

- FCC 5.0 Å abrupt candidate;
- BCC 5.3 Å abrupt candidate;

as the **working reproduction Hamiltonian**.

Label this clearly in configs and reports.

If the reference agent later establishes that the paper used shifted/tapered/re-tabulated cutoffs, update:

- oracle,
- labels,
- priors,
- transition-line calibration,
- graph if needed,
- path weights,

and revalidate accordingly.

Do not silently mix results across cutoff conventions.

---

# Required deliverable

Return one model-reproduction diagnostic report containing:

## A. Hamiltonian provenance table

For every old/new checkpoint and every Hamiltonian-dependent component.

## B. Corrected pipeline checklist

| component | native old setting | candidate setting | regenerated/recalibrated? | evidence |
|---|---|---|---|---|

Include:

- energy,
- force,
- virial,
- substitution labels,
- replay buffer,
- graph cutoff,
- volume prior,
- displacement prior,
- transition line,
- path-weight target.

## C. \(g_u,g_v\) audit

Exact schedules, amplitudes, temperature scaling, origin, and code locations.

State whether retuning is actually necessary.

## D. If \(g\) tuning is triggered

Provide a compact table:

| g_u multiplier | g_v multiplier | terminal mismatch | std logW_u | std logW_v | total std logW | ESS | stability |
|---:|---:|---:|---:|---:|---:|---:|---|

Do not tune against the paper free-energy curve.

## E. Checkpoint trajectory

Plot diagnostics versus training round for candidate-Hamiltonian training.

## F. Old-vs-corrected comparison

At 1200 K and matched compositions show:

- energy,
- volume,
- RMS u,
- SRO,
- std(log W),
- ESS,
- path-weight component variance.

## G. Final root-cause assessment

Separate conclusions into:

1. wrong-Hamiltonian effect;
2. mixed-Hamiltonian pipeline effect, if present;
3. long-training/replay degradation;
4. continuous \(g\)-scale issue, if present;
5. path-weight implementation issue, if independently demonstrated;
6. residual unresolved issues.

Do not call an estimator bug merely because ESS is low.

---

# Immediate priority order

1. Hamiltonian provenance audit.
2. Make the candidate 5.0/5.3 pipeline internally consistent.
3. Recompute priors/calibration/replay labels under that Hamiltonian.
4. Audit current \(g_u,g_v\).
5. Run short corrected-Hamiltonian training with current \(g\) baseline.
6. Decide from diagnostics whether \(g\) needs retuning.
7. Track early-to-late checkpoints to test long-training degradation.
8. Only then continue expensive long training and weighted-BAR reproduction.

Do not spend another long training run before Steps 1–6 are complete.
