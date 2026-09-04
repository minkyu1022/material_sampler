# JANUS Ni–Cr Reproduction Agent — Final Plan v2
## Author-confirmed constrained unmasking + Cu–Ni reproduced setup inheritance + modular code/tuning workflow

This document supersedes earlier reproduction-agent instructions where they conflict.

The **Cu–Ni JANUS baseline has already been reproduced** and should now be treated as the **working engineering baseline** for Ni–Cr.

The immediate goal is:

1. audit/refactor/modularize the current codebase without breaking the existing Cu–Ni reproduction;
2. inherit the reproduced Cu–Ni architecture/training/sampler settings wherever they are system-independent;
3. change only the parts that must differ for Ni–Cr;
4. reproduce the JANUS Ni–Cr FCC/BCC fixed-composition models using the **author-confirmed \(M=N\), one-site-at-a-time, boundary-only quota rule**;
5. tune only where Ni–Cr diagnostics show a problem;
6. after the paper-faithful reproduction works, optionally compare the Denis quota rule with our quota-aware DP decoder;
7. keep the unified BCT/Bain-path method frozen until the JANUS Ni–Cr baseline is established.

The independent reference-MC track should continue separately. Do not redesign that pipeline from this task.

---

# 0. Provenance labels

Every nontrivial implementation choice should be tagged internally as one of:

- `PAPER_CONFIRMED`
- `AUTHOR_CONFIRMED`
- `CUNI_REPRODUCED_BASELINE`
- `PUBLIC_CODE_REFERENCE`
- `PROVISIONAL_RECONSTRUCTION`
- `OUR_METHOD`

Do not silently promote a provisional or inherited setting to paper-confirmed.

---

# 1. Current facts that are now settled

## 1.1 Cu–Ni unconstrained discrete rollout

Denis Blessing confirmed:

> “M=100 is only for unconstrained.”

For Cu–Ni:

\[
N=108,\qquad M=100.
\]

He also confirmed that for the unconstrained sampler:

- multiple sites can be revealed in one numerical step;
- the final step reveals all sites that are still masked.

This agrees with the published JANUS tau-leap description.

Therefore the current Cu–Ni baseline is:

```text
Cu–Ni
N=108
M=100
unconstrained SGC
tau-leap discrete unmasking
multiple reveals allowed
final step clears all remaining masks
```

This has already been reproduced. Do **not** redo it unless a new correctness bug is found.

---

## 1.2 Ni–Cr fixed-composition rollout

Denis confirmed:

> “For constrained we set M=N.”

Therefore:

### Ni–Cr FCC

\[
N=108,\qquad M=108.
\]

### Ni–Cr BCC

\[
N=128,\qquad M=128.
\]

The constrained rollout is one-site-at-a-time.

---

## 1.3 Ni–Cr quota rule

Denis also confirmed:

> “we enforced only when necessary”

when asked whether quota was enforced only at the boundary or whether the categorical was renormalized at every step.

Therefore the JANUS reproduction rule is:

At reveal step \(k\), track

- target Cr count
  \[
  n=N_{\rm Cr},
  \]
- already revealed Cr count
  \[
  c_k,
  \]
- still-masked site count
  \[
  m_k,
  \]
- remaining Cr quota
  \[
  r_k=n-c_k.
  \]

If

\[
0<r_k<m_k,
\]

sample from the **raw model categorical**.

If

\[
r_k=0,
\]

force Ni:

\[
P({\rm Ni})=1,\qquad P({\rm Cr})=0.
\]

If

\[
r_k=m_k,
\]

force Cr:

\[
P({\rm Cr})=1,\qquad P({\rm Ni})=0.
\]

Even after the quota becomes deterministic, continue one-site-at-a-time until all \(N\) sites are revealed.

This is the **author-confirmed reproduction rule**.

Do not replace it with our DP decoder in the main baseline.

---

# 2. High-level execution order

Proceed in this order:

1. snapshot/tag current repo;
2. regression-test the existing Cu–Ni baseline;
3. audit/refactor/modularize code;
4. preserve the Cu–Ni best reproduced config as the default inheritance source;
5. implement author-confirmed Ni–Cr fixed-composition discrete sampling;
6. run small FCC/BCC smoke tests using **Cu–Ni reproduced settings inherited by default**;
7. only tune parameters that show a clear Ni–Cr-specific problem;
8. train full FCC and BCC models;
9. reproduce JANUS free energies / phase boundary against independent reference MC;
10. optionally compare Denis boundary-only quota vs our quota-aware DP;
11. only then return to BCT/Bain.

---

# 3. Code refactor requirements

Do not rewrite working scientific code just for aesthetics.

Refactor where it improves:

- correctness;
- modularity;
- reproducibility;
- future loss combinations;
- sampler swapping;
- performance;
- debugging.

The existing Cu–Ni baseline should remain numerically unchanged after the refactor.

---

# 4. Desired code organization

Adapt to the current repo rather than forcing exact names, but aim for this conceptual split:

```text
src/
  models/
    painn_janus.py
    heads/
      continuous.py
      species.py

  losses/
    continuous/
      tsm.py
      registry.py
    discrete/
      sce.py
      wdce.py
      registry.py

  samplers/
    continuous/
      euler_maruyama.py
      registry.py

    discrete/
      janus_tau_leap.py
      sequential_random_order.py
      fixed_composition_boundary_quota.py
      fixed_composition_dp.py
      registry.py

    hybrid/
      janus_hybrid.py

  targets/
    cuni.py
    nicr_fcc.py
    nicr_bcc.py

  oracle/
    eam.py
    substitution.py

  training/
    fixed_point.py
    replay_buffer.py

  evaluation/
    path_weights.py
    cuni_metrics.py
    nicr_metrics.py
    free_energy.py

configs/
  cuni/
  nicr_fcc/
  nicr_bcc/

scripts/
  train_cuni_janus.sh
  eval_cuni_janus.sh
  train_nicr_fcc_janus.sh
  train_nicr_bcc_janus.sh
  eval_nicr_janus.sh
```

---

# 5. Losses must be modular and config-driven

This is a hard requirement.

Use config such as:

```yaml
loss:
  cont_loss: tsm
  disc_loss: sce

  cont_weight_u: 1.0
  cont_weight_v: 1.0
  disc_weight: 2.0
```

Desired pattern:

```python
cont_loss_fn = CONT_LOSS_REGISTRY[cfg.loss.cont_loss]
disc_loss_fn = DISC_LOSS_REGISTRY[cfg.loss.disc_loss]
```

Each objective should live in a separate module with a common API.

---

# 6. Primary JANUS loss configuration

## Continuous

Use JANUS generalized target-score matching + velocity regression.

For linear interpolation

\[
\alpha(t)=t,
\]

use

\[
c(t)=\frac{t^2}{t^2+(1-t)^2}.
\]

Config:

```yaml
cont_loss: tsm
```

## Discrete

Use JANUS soft cross entropy with heat-bath / Boltzmann single-site labels.

Config:

```yaml
disc_loss: sce
disc_weight: 2.0
```

Primary reproduction:

```text
TSM + SCE
```

---

# 7. Future loss combinations must be easy

The trainer should support future combinations without source edits, for example:

```yaml
loss:
  cont_loss: future_cont_loss
  disc_loss: wdce
```

Implement `wdce` only if it is clean and low-cost to add.

Do **not** use WDCE for the JANUS baseline.

---

# 8. Samplers must be modular

Use config-driven selection.

Example:

```yaml
sampler:
  continuous:
    type: euler_maruyama

  discrete:
    type: fixed_composition_boundary_quota
```

Required discrete modules:

1. `janus_tau_leap`
2. `sequential_random_order`
3. `fixed_composition_boundary_quota`
4. `fixed_composition_dp`

Main usage:

### Cu–Ni
```yaml
discrete.type: janus_tau_leap
```

### Ni–Cr reproduction
```yaml
discrete.type: fixed_composition_boundary_quota
```

### Later ablation
```yaml
discrete.type: fixed_composition_dp
```

---

# 9. Single-shell-file execution

Each major training/evaluation should run from one shell file.

Required interface:

```bash
bash scripts/train_cuni_janus.sh
bash scripts/eval_cuni_janus.sh

bash scripts/train_nicr_fcc_janus.sh
bash scripts/train_nicr_bcc_janus.sh
bash scripts/eval_nicr_janus.sh
```

Shell scripts should be thin wrappers around configs.

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m train \
  --config configs/nicr_fcc/janus_fixed_composition.yaml \
  "$@"
```

Do not put scientific logic inside shell scripts.

---

# 10. Cu–Ni reproduced config is the default inheritance source

This is the main tuning principle.

Before making new Ni–Cr hyperparameter choices, load the **actual best Cu–Ni reproduced config** and copy all settings that are not inherently system-specific.

Do not restart tuning from generic paper defaults when we already have a working reproduced configuration.

Create an explicit inheritance mechanism if possible:

```yaml
defaults:
  - ../cuni/reproduced_best
```

or equivalent config composition.

Then override only Ni–Cr-specific settings.

---

# 11. Settings that should be inherited from Cu–Ni by default

Unless there is a strong system-specific reason otherwise, inherit:

- model architecture;
- hidden dimensions;
- PaiNN depth;
- RBF configuration;
- time embedding;
- continuous head parameterization;
- species head parameterization;
- continuous loss implementation;
- discrete loss implementation;
- loss weights if already validated;
- optimizer type;
- learning rate;
- scheduler;
- gradient clipping setting;
- EMA setting;
- batch size;
- replay buffer strategy;
- replay buffer size;
- replay sampling rule;
- inner updates / round;
- target-score stabilization;
- path-weight precision;
- forward/backward drift implementation;
- continuous diffusion-strength implementation;
- checkpoint cadence;
- logging cadence;
- mixed precision policy;
- any clipping that was actually required for the reproduced Cu–Ni run.

Do not silently change these while moving to Ni–Cr.

---

# 12. Settings that necessarily change for Ni–Cr

These must be overridden:

## Conditioning

Cu–Ni:
\[
(T,\Delta\mu)
\]

Ni–Cr free-energy/binodal route:
\[
(T,c_0),
\qquad
c_0=\frac{N_{\rm Cr}}{N}.
\]

## Discrete sampler

Cu–Ni:
```text
M=100 tau-leap
```

Ni–Cr:
```text
M=N one-site sequential
boundary-only quota
```

## Lattice / N

FCC:
\[
N=108.
\]

BCC:
\[
N=128.
\]

## Fresh terminals / round

Use the paper-reported Ni–Cr value as the initial override:

```yaml
fresh_terminals_per_round: 500
```

rather than Cu–Ni's 1000.

## System-specific priors / potential / cutoff

Use Ni–Cr-specific calibration.

Do not inherit Cu–Ni physical prior parameters blindly.

---

# 13. Ni–Cr FCC/BCC models are separate

Train two independent lattice-specific models.

### FCC

\[
N=108,\qquad M=108.
\]

Reference lattice:
- \(3\times3\times3\) conventional FCC.

### BCC

\[
N=128,\qquad M=128.
\]

Reference lattice:
- \(4\times4\times4\) conventional BCC.

Do not use one model with an FCC/BCC token.

---

# 14. Ni–Cr condition

Let

\[
n=N_{\rm Cr},
\qquad
c_0=\frac{n}{N}.
\]

One amortized model per lattice should cover all rungs.

### FCC

\[
n=0,\ldots,108.
\]

### BCC

\[
n=0,\ldots,128.
\]

Terminal states must satisfy:

\[
N_{\rm Cr}=n
\]

exactly.

---

# 15. Author-confirmed Ni–Cr discrete rollout

At rollout start:

1. all sites are MASK;
2. construct the one-site reveal order using the existing sequential/random-order implementation;
3. run exactly \(N\) reveal steps;
4. reveal one site per step;
5. track quota throughout.

At step \(k\):

\[
r_k=n-c_k,
\]

where \(c_k\) is the number of revealed Cr atoms.

Let \(m_k\) be the remaining masked-site count.

### Interior

If

\[
0<r_k<m_k,
\]

sample from raw network categorical:

\[
q_{\theta,i}({\rm Ni}),
\qquad
q_{\theta,i}({\rm Cr}).
\]

### Cr quota already full

If

\[
r_k=0,
\]

force Ni.

### All remaining sites must be Cr

If

\[
r_k=m_k,
\]

force Cr.

Continue the remaining deterministic reveal events one-by-one until all \(N\) sites are unmasked.

---

# 16. Training masking does not change

For buffered clean terminal \(a_1\):

1. sample \(t\);
2. independently mask every site with probability
   \[
   1-\alpha_a(t);
   \]
3. use
   \[
   \alpha_a(t)=t.
   \]

Do not switch the main reproduction to fixed-\(m\) masking.

---

# 17. Path probability audit

The discrete log probability must use the probability of the **actual sampling kernel**.

For an interior step:

\[
0<r_k<m_k,
\]

use:

\[
\log p_k
=
\log q_{\theta,i_k}(a_{i_k}).
\]

For a forced deterministic step:

\[
r_k=0
\quad\text{or}\quad
r_k=m_k,
\]

the actual allowed token probability is 1:

\[
\log p_k=0.
\]

Therefore:

\[
\log q_{\rm disc}
=
\sum_{k:\,0<r_k<m_k}
\log q_{\theta,i_k}(a_{i_k}).
\]

Audit this explicitly.

Use float64 accumulation.

---

# 18. Required constrained-sampler tests

Test:

- \(n=0\);
- \(n=1\);
- \(n=N/4\);
- \(n=N/2\);
- \(n=3N/4\);
- \(n=N-1\);
- \(n=N\).

Check:

1. one reveal/step;
2. exactly \(N\) steps;
3. revealed site never changes;
4. terminal \(N_{\rm Cr}=n\);
5. invalid quota state raises;
6. forced token contributes log probability 0;
7. no NaN/inf;
8. deterministic seed reproducibility;
9. toy path probability agrees with hand calculation.

---

# 19. Ni–Cr tuning philosophy

Do **not** begin with a broad hyperparameter sweep.

The default initial Ni–Cr run should be:

\[
\boxed{\text{Cu–Ni reproduced best config}}
\]

plus only the required Ni–Cr changes.

The first question is:

> Does the Cu–Ni working setup transfer cleanly to Ni–Cr once the condition/sampler/lattice are changed?

Only tune if diagnostics identify a concrete failure mode.

---

# 20. Stage A — direct inheritance smoke test

For FCC and BCC separately:

Use the inherited Cu–Ni settings and only override:

- lattice;
- \(N\);
- \(M=N\);
- fixed-composition condition;
- quota sampler;
- Ni–Cr potential/prior/cutoff;
- fresh terminals/round.

Use sparse compositions:

\[
x_{\rm Cr}\in
\{0,\;0.25,\;0.5,\;0.75,\;1\}
\]

and a small temperature subset.

Run a short smoke experiment.

Monitor:

- training loss;
- continuous loss components;
- discrete SCE;
- gradient norm;
- terminal energy;
- displacement;
- volume;
- quota forcing frequency;
- first forced-step location;
- path-weight mean/std;
- ESS;
- invalid fraction;
- NaNs/infs.

If healthy, proceed without tuning.

---

# 21. Stage B — targeted tuning only when needed

Tune one axis at a time.

Priority order:

## 21.1 Numerical/correctness first

Before changing optimization:

- confirm target score sign;
- confirm backward drift sign;
- confirm path probability;
- confirm quota logic;
- confirm prior calibration;
- confirm cutoff/potential;
- confirm replay condition binding.

Do not “tune around” a correctness bug.

## 21.2 Learning rate

Start from the reproduced Cu–Ni lr.

If unstable, run a small local screen around it.

Example only:

```text
0.5x
1.0x
2.0x
```

of the Cu–Ni value.

Do not run a broad order-of-magnitude sweep first.

## 21.3 Batch size

Start from the Cu–Ni reproduced batch size.

Change only if:

- gradient noise is clearly excessive;
- GPU memory permits a useful increase;
- throughput is poor.

## 21.4 Replay buffer

Start from Cu–Ni reproduced:
- buffer size;
- replay sampling;
- update cadence.

Tune only if:
- stale-buffer behavior appears;
- new terminals are not propagating into training;
- composition coverage is poor.

## 21.5 Fresh terminals / round

Start from JANUS Ni–Cr paper value 500.

Increase only if:
- buffer diversity is inadequate;
- outer-loop fixed point moves too slowly;
- composition/temperature coverage is insufficient.

## 21.6 Discrete loss weight

Start from:

\[
\lambda_{\rm disc}=2.
\]

Tune only if:
- species head clearly lags continuous heads;
- or discrete loss overwhelms the shared trunk.

Use diagnostics rather than raw loss magnitudes alone.

## 21.7 Continuous diffusion strength

Start from the working Cu–Ni implementation/scaling.

Tune only if:
- path-weight variance is very large;
- continuous transport is unstable;
- terminal displacement/volume coverage is poor.

Keep \(u\) and \(v\) changes separated where possible.

## 21.8 Gradient clipping / EMA / target clipping

Inherit Cu–Ni settings.

Do not turn these on simply because they exist.

If introduced, record exactly:
- why;
- threshold;
- effect on training stability;
- effect on ESS/physics.

---


# 21.9 Diffusion-strength scale \(g(t)\): explicit stability knob

Preserve the **temperature-conditioned form** of the JANUS diffusion strength.

Use a parameterization such as

\[
g_c(t;T)
=
s_{g,c}\,\bar g_c(t)\sqrt{\frac{T}{T_{\rm ref}}},
\qquad
c\in\{u,v\},
\]

where:

- \(\bar g_c(t)\) is the existing/base time-dependent diffusion schedule;
- \(\sqrt{T/T_{\rm ref}}\) is the desired temperature scaling and should be preserved;
- \(s_{g,c}\) is a configurable multiplicative scale used only to control the overall stochastic strength.

The default should inherit the **working Cu–Ni reproduced value**:

```yaml
sampler:
  diffusion:
    temperature_scaling: sqrt_T_over_Tref

    u:
      scale: <inherit_from_cuni>

    v:
      scale: <inherit_from_cuni>
```

## When to lower the scale

If either

- final inference rollouts, or
- self-bootstrapped buffer-generation rollouts

show numerical/physical explosion, for example:

- rapidly diverging displacement magnitude;
- unrealistic volume excursions;
- non-finite terminal energies;
- NaN/Inf states;
- extremely large \(\operatorname{std}(\log W)\);
- ESS collapse associated with continuous-path instability;
- unstable score-corrected drift;

then **one of the first targeted tuning directions should be to reduce the overall \(g(t)\) scale while keeping the temperature dependence unchanged**.

Do not remove the \(T\)-conditioning merely to stabilize the run.

Conceptually:

\[
s_g:
1.0
\rightarrow
0.5
\rightarrow
0.25
\]

is a reasonable local stability screen if the inherited value is clearly too aggressive.  
Use a small local screen rather than a broad sweep.

The exact candidate values should be centered around the reproduced Cu–Ni value rather than assumed universally.

## Why this can stabilize the rollout

The JANUS continuous generative SDE has the form

\[
dx_t^c
=
\left[
b_\theta^c
+
g_c(t;T)^2s_\theta^c
\right]dt
+
\sqrt{2}\,g_c(t;T)\,dW_t.
\]

Therefore multiplying \(g_c\) by a scale \(s_g<1\):

- decreases the stochastic noise amplitude linearly;
- decreases the score-correction term \(g_c^2s_\theta\) quadratically.

So if instability is caused by an overly aggressive stochastic/score-corrected rollout, lowering \(s_g\) can materially improve stability.

## Important constraint

Do **not** tune \(g\) only on training loss.

For every \(g\)-scale candidate compare:

- terminal energy;
- displacement;
- volume;
- invalid/non-finite fraction;
- \(\operatorname{std}(\log W)\);
- ESS;
- physical observable agreement with reference;
- mode/composition coverage.

A smaller \(g\) may stabilize trajectories but can also reduce exploration/mixing.

Therefore choose the **largest stable \(g\)-scale that still gives good ESS and physical coverage**, rather than simply minimizing \(g\).

## Channel-specific tuning

Start by inheriting the Cu–Ni relation between the displacement and volume channels.

Do not immediately introduce independent \(s_{g,u}\) and \(s_{g,v}\) sweeps.

If diagnostics clearly localize the instability:

- displacement explosion only → tune \(s_{g,u}\);
- volume explosion only → tune \(s_{g,v}\);
- both channels unstable → first try a shared multiplicative reduction.

## Consistency across training and inference

The chosen diffusion configuration must be stored with the checkpoint and used explicitly in:

- buffer-generation rollouts;
- evaluation/inference rollouts;
- forward/backward path-weight calculations.

Do not silently train buffers with one \(g\)-scale and evaluate with another unless that is an explicit ablation.


# 23. Tuning acceptance criteria

Do not select a hyperparameter based on training loss alone.

Use:

### Sampler quality
- ESS;
- \(\operatorname{std}(\log W)\);
- finite path-weight fraction;
- terminal validity.

### Physical quality
- energy/atom;
- volume/atom;
- displacement;
- chemical ordering;
- agreement with reference MC where available.

### Discrete quality
- exact composition;
- quota forcing frequency;
- first forced-step distribution;
- categorical entropy/calibration.

### Optimization quality
- loss stability;
- gradient norm;
- seed-to-seed variance;
- no collapse.

### Efficiency
- network evaluations;
- oracle NFE;
- wall time;
- memory.

---

# 24. Important forcing diagnostics

Because Denis's implementation only forces when necessary, log:

- number of forced reveals / trajectory;
- fraction of forced reveals;
- first step where forcing begins;
- remaining fraction of trajectory at first force;
- whether forcing is to Ni or Cr;
- dependence on composition;
- dependence on temperature;
- dependence on training round;
- FCC vs BCC.

This is one of the most important diagnostics for understanding the constrained decoder.

If the network learns the composition condition well, forcing should generally become less severe.

---

# 25. Full Ni–Cr training gate

Proceed to full training only when FCC/BCC smoke tests show:

- stable training;
- exact composition;
- finite path weights;
- reasonable ESS;
- no systematic energy drift;
- reasonable displacement/volume;
- no severe forced-tail pathology;
- no correctness issues.

---

# 26. Full amortized Ni–Cr training

Train:

### FCC
one model over
\[
n=0,\ldots,108.
\]

### BCC
one model over
\[
n=0,\ldots,128.
\]

Use the inherited Cu–Ni training infrastructure, with only validated Ni–Cr-specific overrides.

Do not train one model per composition.

---

# 27. Final JANUS reproduction targets

Compare against independent reference MC.

Required outputs:

- \(G_{\rm mix}^{FCC}(x,T)\);
- \(G_{\rm mix}^{BCC}(x,T)\);
- neighboring-rung free-energy estimates;
- FCC/BCC absolute alignment;
- Ni-rich FCC stability;
- Cr-rich BCC stability;
- coexistence boundary;
- uncertainty / seed consistency.

---

# 28. Optional later ablation: quota-aware DP

Only after the Denis reproduction is stable.

Tag:

```text
OUR_METHOD
```

Define network odds:

\[
w_i=
\frac{q_i({\rm Cr})}
{q_i({\rm Ni})}.
\]

For masked set \(S\), define the fixed-cardinality partition function:

\[
Z_r(S)
=
\sum_{\substack{A\subseteq S\\|A|=r}}
\prod_{i\in A}w_i.
\]

Compute it using:

\[
F(j,k)
=
F(j-1,k)
+
w_jF(j-1,k-1).
\]

Then condition each reveal on the exact remaining quota.

Compare:

```text
Denis boundary-only quota
vs
quota-aware DP
```

using:

- forcing frequency;
- ESS;
- log-weight variance;
- energy;
- ordering;
- free-energy error;
- phase-boundary error;
- runtime.

Do not replace the reproduction baseline with DP.

---

# 29. Public-code references

Use as engineering references only.

### MDNS
https://github.com/yuchen-zhu-zyc/MDNS

Useful for:
- sequential one-site decoding;
- random permutations;
- masking/replay implementation;
- modular discrete sampler code.

### MetaDNS
https://github.com/xiaochendu/metadns

Useful for:
- alloy discrete-sampling engineering;
- batching;
- scripts/config organization;
- replay mechanics.

### PDNS
https://github.com/AlexandreGUO2001/PDNS

Useful for:
- modular discrete utilities;
- alternative masking helpers.

Do not import their objectives into the JANUS baseline unless explicitly performing an ablation.

---

# 30. Performance audit

Before long jobs, profile:

- model forward time;
- EAM energy/force/virial time;
- substitution-label time;
- sequential reveal loop;
- replay-buffer overhead;
- path-weight bookkeeping;
- quota logic;
- CPU↔GPU transfer.

Optimize real bottlenecks first.

Priority:

1. batched substitution labels;
2. no redundant model forward;
3. efficient replay tensors;
4. vectorized masking;
5. cached static lattice info where valid;
6. no Python loop around expensive oracle calls;
7. float64 only where needed:
   - path-weight accumulation;
   - free-energy computation;
   - DP ablation.

---

# 31. Run provenance

Every run must save:

- resolved config;
- inherited Cu–Ni parent config/version;
- all Ni–Cr overrides;
- git commit;
- dirty-tree state;
- random seed;
- system/lattice;
- potential/hash;
- cutoff;
- architecture;
- loss names;
- sampler names;
- \(N\);
- \(M\);
- composition mode;
- optimizer;
- replay settings;
- precision settings;
- checkpoint parent;
- start/end time;
- hardware.

---

# 32. Reference MC

Do not modify the independent reference-MC track from this task.

Use its outputs for evaluation.

Known ambiguities there remain separate:
- canonical pair-exchange attempts/sweep;
- exact seven temperatures;
- exact canonical T-RE cadence.

---

# 33. BCT / Bain-path branch

Preserve all BCT/Bain code, checkpoints, tests, and notes.

For now:

```text
NO NEW BCT TRAINING
NO NEW BCT REFERENCE PRODUCTION
NO BCT-vs-JANUS CLAIMS
```

Return to BCT/Bain only after the JANUS Ni–Cr baseline is convincingly reproduced.

---

# 34. Immediate action list

Execute now:

1. snapshot/tag repo;
2. identify and freeze the current best Cu–Ni reproduced config;
3. run Cu–Ni regression tests;
4. refactor loss registry;
5. refactor sampler registry;
6. clean configs and shell entrypoints;
7. preserve the Cu–Ni tau-leap baseline unchanged;
8. implement `fixed_composition_boundary_quota`;
9. add exact quota/path-probability unit tests;
10. build Ni–Cr configs by **inheriting the Cu–Ni reproduced config**;
11. override only Ni–Cr-specific settings;
12. run FCC smoke;
13. run BCC smoke;
14. tune only if diagnostics identify a specific problem;
15. train full FCC model;
16. train full BCC model;
17. reproduce free energies / phase boundary;
18. optionally compare against quota-aware DP;
19. only then return to BCT/Bain.

Do not wait for another message between engineering stages unless a genuine correctness/scientific ambiguity blocks progress.
