# JANUS Ni–Cr Reproduction Agent — Final Updated Plan
## Author-confirmed discrete implementation, code refactor, modular losses/samplers, and Ni–Cr-first execution

This document supersedes earlier reproduction instructions where they conflict.

The **Cu–Ni JANUS baseline is already reproduced correctly enough to serve as the starting point**. Do **not** restart Cu–Ni from scratch unless a new correctness bug is found.

The immediate objective is now:

1. audit/refactor/modularize the current codebase;
2. preserve the existing Cu–Ni implementation as the validated unconstrained baseline;
3. implement the **author-confirmed fixed-composition Ni–Cr discrete rollout**;
4. reproduce the separate FCC and BCC JANUS Ni–Cr models;
5. only after the paper-faithful Ni–Cr reproduction works, optionally compare the Denis quota rule against our more principled quota-aware DP decoder;
6. keep the unified BCT/Bain-path method frozen for now.

The independent reference-MC track should continue separately. Do not redesign that pipeline from this reproduction task.

---

# 0. Source-of-truth labels

Every important implementation choice must be tagged internally as one of:

- `PAPER_CONFIRMED`
- `AUTHOR_CONFIRMED`
- `PUBLIC_CODE_REFERENCE`
- `PROVISIONAL_RECONSTRUCTION`
- `OUR_METHOD`

Do not silently promote an implementation choice into `PAPER_CONFIRMED`.

---

# 1. Author-confirmed discrete implementation

Denis Blessing has now clarified the key ambiguities.

## 1.1 Unconstrained alloy rollout

Denis confirmed:

> “M=100 is only for unconstrained.”

For Cu–Ni:

\[
N=108,\qquad M=100.
\]

He also confirmed that, with \(N=108>M=100\):

- **multiple sites may be revealed in one step**;
- the **final step reveals all sites that are still masked**.

This matches the published JANUS tau-leap description.

Therefore:

```text
Cu–Ni unconstrained reproduction
= M=100
= tau-leap / multiple-reveal discrete integrator
```

This has already been implemented and reproduced.

### Action

Do **not** redo Cu–Ni training as the primary task.

Preserve:

- existing Cu–Ni checkpoints;
- existing configs;
- existing evaluation outputs;
- existing path-weight implementation;
- existing tau-leap sampler.

Only rerun Cu–Ni if the code audit discovers a real correctness bug.

---

## 1.2 Fixed-composition rollout

Denis confirmed:

> “For constrained we set M=N.”

Therefore the fixed-composition JANUS route uses:

### Ni–Cr FCC

\[
N=108,\qquad M=N=108.
\]

### Ni–Cr BCC

\[
N=128,\qquad M=N=128.
\]

The constrained rollout is therefore naturally interpreted as **one reveal per numerical step**.

---

## 1.3 Quota enforcement

Denis also confirmed:

> “we enforced only when necessary”

and explicitly said they did **not** renormalize the categorical probabilities at every step.

Thus the paper/private implementation to reproduce is:

### State bookkeeping

At step \(k\), track:

- target Cr count:
  \[
  n=N_{\rm Cr};
  \]

- already revealed Cr count:
  \[
  c_k;
  \]

- number of still-masked sites:
  \[
  m_k;
  \]

- remaining Cr quota:
  \[
  r_k=n-c_k.
  \]

Always require

\[
0\le r_k\le m_k.
\]

### Interior case

If

\[
0<r_k<m_k,
\]

use the network's **raw categorical probability**:

\[
q_{\theta,i}({\rm Ni}),
\qquad
q_{\theta,i}({\rm Cr})
\]

without quota-aware renormalization.

### Boundary case 1

If

\[
r_k=0,
\]

force

\[
P({\rm Cr})=0,\qquad P({\rm Ni})=1.
\]

### Boundary case 2

If

\[
r_k=m_k,
\]

force

\[
P({\rm Cr})=1,\qquad P({\rm Ni})=0.
\]

This is the **paper-faithful / author-confirmed reproduction rule**.

Call this implementation something explicit such as:

```yaml
sampler:
  discrete:
    type: fixed_composition_boundary_quota
```

Do **not** replace it with our DP decoder in the main reproduction.

---

# 2. Cu–Ni baseline: preserve, do not redo

The current Cu–Ni reproduction should remain the unconstrained reference implementation.

Required characteristics:

```yaml
system: cuni
lattice: fcc
N: 108

condition:
  mode: sgc
  variables: [T, delta_mu]

sampler:
  discrete:
    type: janus_tau_leap
  continuous_steps: 100
  discrete_steps: 100
```

The discrete process should remain:

\[
p_n
=
\frac{\alpha_a(t_{n+1})-\alpha_a(t_n)}
{1-\alpha_a(t_n)}
\]

with

\[
\alpha_a(t)=t.
\]

Each still-masked site is independently revealed with probability \(p_n\).

Multiple reveals in one step are allowed.

The final step has \(p_{M-1}=1\), so all remaining masks are removed.

### Do not change

- Cu–Ni physical target;
- Cu–Ni conditioning;
- Cu–Ni tau-leap sampler;
- Cu–Ni replay logic;
- Cu–Ni training objective.

Use this implementation as the regression test for all refactoring.

---

# 3. Immediate priority: code audit and refactor

Before launching Ni–Cr training, inspect the current repository.

The goal is not cosmetic rewriting.

Refactor only where it improves:

- correctness;
- modularity;
- future objective combinations;
- sampler swapping;
- experiment reproducibility;
- performance;
- maintainability.

The Cu–Ni baseline must still reproduce the same outputs after refactoring.

---

# 4. Desired code architecture

Adapt to the existing repository rather than forcing exact names, but aim for this conceptual separation:

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
      event_scheduler.py

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

# 5. Losses must be modular and config-selectable

This is a hard requirement.

The trainer must not hard-code one objective combination.

Use config such as:

```yaml
loss:
  cont_loss: tsm
  disc_loss: sce

  cont_weight_u: 1.0
  cont_weight_v: 1.0
  disc_weight: 2.0
```

Desired registry pattern:

```python
cont_loss_fn = CONT_LOSS_REGISTRY[cfg.loss.cont_loss]
disc_loss_fn = DISC_LOSS_REGISTRY[cfg.loss.disc_loss]
```

Each objective should live in its own module with a common API.

---

# 6. Primary JANUS losses

## 6.1 Continuous loss: `tsm`

Use the JANUS continuous objective:

- velocity regression;
- generalized target-score matching.

For linear interpolation

\[
\alpha(t)=t,
\]

use

\[
c(t)=
\frac{t^2}
{t^2+(1-t)^2}.
\]

The target is the bounded combination of:

- terminal target score;
- prior score.

Do not reintroduce a naive singular \(1/t\) target if the generalized target-score formulation is already implemented.

Config:

```yaml
loss:
  cont_loss: tsm
```

---

## 6.2 Discrete loss: `sce`

Use JANUS soft cross entropy with the heat-bath / single-site Boltzmann conditional.

Config:

```yaml
loss:
  disc_loss: sce
  disc_weight: 2.0
```

The primary reproduction remains:

```text
TSM + SCE
```

---

# 7. Future loss extensibility

The code must make future combinations easy.

For example:

```yaml
loss:
  cont_loss: future_continuous_objective
  disc_loss: wdce
```

should not require modifying the trainer.

Implement `wdce` only if it is clean and low-cost to add.

MDNS/MetaDNS are useful references for WDCE engineering, but **do not switch the JANUS reproduction to WDCE**.

---

# 8. Samplers must be modular

Use config-driven sampler selection.

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

The main reproduction uses:

- Cu–Ni:
  `janus_tau_leap`
- Ni–Cr:
  `fixed_composition_boundary_quota`

The DP version is an ablation / improvement, not the reproduction baseline.

---

# 9. Single-shell-file execution

Each major method/system should be runnable from one shell script.

Required interface:

```bash
bash scripts/train_cuni_janus.sh
bash scripts/eval_cuni_janus.sh

bash scripts/train_nicr_fcc_janus.sh
bash scripts/train_nicr_bcc_janus.sh
bash scripts/eval_nicr_janus.sh
```

Shell files should be thin wrappers.

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m train \
  --config configs/nicr_fcc/janus_fixed_composition.yaml \
  "$@"
```

Do not put scientific logic in shell files.

---

# 10. Run provenance

Every run must save:

- full resolved config;
- git commit hash;
- dirty-tree status;
- random seed;
- system;
- lattice;
- potential identifier/hash;
- cutoff convention;
- model architecture;
- continuous loss;
- discrete loss;
- continuous sampler;
- discrete sampler;
- composition mode;
- \(M\);
- path-weight dtype;
- replay settings;
- optimizer settings;
- checkpoint parent/provenance;
- timestamps;
- hardware info.

---

# 11. Ni–Cr model structure

Train two **separate** lattice-specific models.

Do not use an FCC/BCC condition token.

## FCC model

\[
N=108.
\]

Reference lattice:

- \(3\times3\times3\) conventional FCC.

Rollout:

\[
M=N=108.
\]

## BCC model

\[
N=128.
\]

Reference lattice:

- \(4\times4\times4\) conventional BCC.

Rollout:

\[
M=N=128.
\]

---

# 12. Ni–Cr conditioning

For the JANUS free-energy/binodal route, use the paper's fixed-composition sampler.

Let

\[
n=N_{\rm Cr},
\qquad
c_0=\frac{n}{N}.
\]

Condition the lattice-specific model on:

- temperature;
- rung composition \(c_0\).

The terminal configuration must satisfy:

\[
N_{\rm Cr}=n.
\]

One amortized model per lattice should span all composition rungs.

### FCC

\[
n=0,\ldots,108.
\]

### BCC

\[
n=0,\ldots,128.
\]

---

# 13. Ni–Cr discrete rollout: exact reproduction rule

At rollout start:

1. initialize all species sites as MASK;
2. generate the reveal order according to the existing sequential implementation / JANUS random-order convention;
3. perform exactly \(N\) reveal steps;
4. reveal one site at each step;
5. track the remaining Cr quota.

At step \(k\):

\[
r_k=n-c_k,
\]

where \(c_k\) is the number of Cr sites already revealed.

Let \(m_k\) be the number of masked sites.

Use:

\[
0<r_k<m_k
\]

→ sample from raw network categorical.

Use:

\[
r_k=0
\]

→ force Ni.

Use:

\[
r_k=m_k
\]

→ force Cr.

Update quota after each reveal.

At terminal:

\[
N_{\rm Cr}=n
\]

must hold exactly.

---

# 14. Training masking remains Bernoulli

Do not change the inner JANUS masking rule.

For a buffered clean terminal \(a_1\):

1. sample \(t\);
2. independently mask every site with probability
   \[
   1-\alpha_a(t);
   \]
3. use
   \[
   \alpha_a(t)=t.
   \]

Do **not** switch the main reproduction to fixed-\(m\) masking.

---

# 15. Path probability for the author-confirmed quota rule

The discrete log probability must use the **actual probability used by the sampler**.

At an interior step:

\[
0<r_k<m_k,
\]

if site \(i\) is sampled from raw categorical:

\[
p_k(a_i)
=
q_{\theta,i}(a_i).
\]

At a forced boundary step:

\[
r_k=0
\]

or

\[
r_k=m_k,
\]

the allowed species has probability 1 under the actual constrained sampling kernel:

\[
\log p_k = 0.
\]

Therefore:

\[
\log q_{\rm disc}
=
\sum_{k:\,0<r_k<m_k}
\log q_{\theta,i_k}(a_{i_k}).
\]

Do not accidentally add the raw network probability at a step where the sampler actually forced the token deterministically.

Audit this carefully.

Use float64 accumulation.

---

# 16. Required quota tests

Unit tests must include:

- \(n=0\);
- \(n=1\);
- \(n=N/4\);
- \(n=N/2\);
- \(n=3N/4\);
- \(n=N-1\);
- \(n=N\).

Check:

1. exactly one reveal/step;
2. all sites revealed after \(N\) steps;
3. no revealed site changes again;
4. terminal \(N_{\rm Cr}=n\);
5. invalid quota states raise immediately;
6. forced-step probability is handled correctly in logq;
7. no NaNs/infs;
8. deterministic seed reproducibility;
9. path probability on toy examples matches manual calculation.

---

# 17. Ni–Cr training progression

Do not immediately launch the full production training.

## Phase A — FCC smoke

Use a sparse composition set:

- 0
- 0.25
- 0.5
- 0.75
- 1.0

and a small temperature subset.

Check:

- exact composition;
- energy;
- displacement;
- volume;
- replay behavior;
- path weights;
- ESS;
- quota forcing frequency.

## Phase B — BCC smoke

Repeat the same checks.

## Phase C — full amortized training

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

Do not train one model per composition.

---

# 18. Important diagnostic: forcing frequency

Because Denis's implementation only enforces the quota when necessary, log:

- fraction of reveal steps that are forced;
- distribution of the first forced-step index;
- forced fraction in the final 10% of the rollout;
- dependence on:
  - temperature,
  - composition,
  - training round,
  - FCC vs BCC.

This is important scientifically.

If the trained model naturally learns the fixed-composition condition well, forcing should become rare except near the end.

If forcing remains frequent, the raw categorical head is poorly calibrated to the composition condition.

---

# 19. Optional methodological ablation: quota-aware DP

After the Denis reproduction is stable, implement our improved constrained decoder.

Do **not** use this as the main baseline.

Tag it:

```text
OUR_METHOD
```

Suppose the network outputs:

\[
q_i({\rm Ni}),
\qquad
q_i({\rm Cr}).
\]

Define:

\[
w_i=
\frac{q_i({\rm Cr})}
{q_i({\rm Ni})}.
\]

For remaining masked set \(S\), define:

\[
Z_r(S)
=
\sum_{\substack{A\subseteq S\\|A|=r}}
\prod_{i\in A} w_i.
\]

Compute with the recurrence:

\[
F(j,k)
=
F(j-1,k)
+
w_jF(j-1,k-1).
\]

This is an \(O(mr)\) dynamic-programming evaluation of the fixed-cardinality partition function.

For next reveal site \(i\):

\[
P(a_i={\rm Cr}\mid r)
=
\frac{
w_i Z_{r-1}(S\setminus\{i\})
}{
Z_r(S\setminus\{i\})
+
w_i Z_{r-1}(S\setminus\{i\})
}.
\]

This conditions the network's factorized logits on the exact remaining Cr count at **every** reveal step.

---

# 20. DP vs Denis rule comparison

After paper reproduction:

compare

```text
Denis boundary-only quota
vs
quota-aware DP
```

using identical model architecture/training budget.

Metrics:

- terminal exact-count validity;
- forcing frequency;
- terminal energy;
- ordering/SRO;
- \(\operatorname{std}(\log W)\);
- ESS;
- free-energy error;
- phase-boundary error;
- network evaluations;
- DP overhead;
- wall-clock/sample.

Do not claim DP is better before measuring.

---

# 21. Cu–Ni sequential ablations are now optional

Because Denis confirmed the unconstrained Cu–Ni implementation, the Cu–Ni `M=N` sequential variants are **not required for reproduction**.

Do not spend substantial compute on them now.

If desired later, they can be studied as pure algorithmic ablations:

1. `M=108` one-site sequential shared grid;
2. continuous `M_c=100`, discrete `M_d=108` decoupled clocks.

But these are **lower priority than Ni–Cr reproduction**.

---

# 22. Hyperparameter / stability structure

Keep all stabilization knobs separate and configurable.

Do not use one generic “clipping” option.

Example:

```yaml
optim:
  type: adam
  lr: 3e-3

  grad_clip:
    enabled: false
    max_norm: 1.0

  ema:
    enabled: false
    decay: 0.9999

loss:
  target_clip:
    enabled: false

  wdce_weight_clip:
    enabled: false
    max_value: 1e5

state:
  physical_clip:
    enabled: ...
```

For the JANUS baseline:

- start from paper-style Adam lr \(3\times10^{-3}\);
- use TSM + SCE;
- do not add target clipping without evidence;
- do not replace SCE with WDCE.

---

# 23. Performance audit

Before long Ni–Cr training, profile:

- network forward time;
- EAM energy/force/virial time;
- substitution-label time;
- replay-buffer overhead;
- sequential reveal-loop overhead;
- path-weight bookkeeping;
- forced-quota logic overhead;
- CPU↔GPU transfer.

Priority optimizations:

1. batch substitution-energy labels;
2. avoid duplicate network evaluations;
3. keep replay tensors efficient;
4. vectorize training masks;
5. cache static lattice information where valid;
6. no Python loop around expensive oracle calls;
7. float64 only where needed:
   - path-weight accumulation;
   - DP;
   - free-energy calculations.

Do not sacrifice correctness for throughput.

---

# 24. Reference MC

The independent reference-MC agent should continue its current JANUS-style FCC/BCC canonical-ladder work.

Do not modify that pipeline from this reproduction task.

Known reference ambiguities remain separate:

- exact canonical pair-exchange attempts/sweep;
- exact seven Ni–Cr temperatures;
- exact canonical temperature-RE cadence.

The neural reproduction agent should consume reference outputs for evaluation.

---

# 25. BCT / Bain-path branch

Preserve all current BCT/Bain code, checkpoints, tests, and notes.

But for now:

```text
NO NEW BCT TRAINING
NO NEW BCT REFERENCE PRODUCTION
NO BCT-vs-JANUS CLAIMS
```

The BCT/Bain method remains the planned methodological extension after the JANUS baseline is established.

---

# 26. Final Ni–Cr completion gate

Do not declare success just because training finishes.

Require:

1. exact fixed-composition terminal counts;
2. stable FCC training;
3. stable BCC training;
4. finite path weights;
5. reasonable ESS/log-weight variance;
6. correct physical observables;
7. neighboring-rung free-energy estimation;
8. \(G_{\rm mix}^{FCC}(x,T)\);
9. \(G_{\rm mix}^{BCC}(x,T)\);
10. correct FCC/BCC relative alignment;
11. Ni-rich FCC stability;
12. Cr-rich BCC stability;
13. coexistence boundary in agreement with the independent reference;
14. multi-seed reproducibility;
15. explicit provenance of all author-confirmed vs paper-confirmed choices.

---

# 27. Immediate action order

Proceed in this order:

1. snapshot/tag the current repository;
2. verify existing Cu–Ni tau-leap baseline still passes regression tests;
3. audit/refactor code architecture;
4. modularize continuous/discrete loss registries;
5. modularize continuous/discrete sampler registries;
6. ensure single `.sh` entrypoint per method/system;
7. implement `fixed_composition_boundary_quota`;
8. unit-test exact-count/path-probability behavior;
9. profile runtime;
10. run Ni–Cr FCC smoke;
11. run Ni–Cr BCC smoke;
12. train full FCC fixed-composition JANUS model;
13. train full BCC fixed-composition JANUS model;
14. reproduce free-energy/phase-boundary results against independent reference MC;
15. only then implement/compare the DP quota-aware decoder;
16. only after JANUS baseline completion, return to BCT/Bain.

Do not wait for another message between engineering stages unless there is a genuine scientific ambiguity or correctness blocker.
