# JANUS core reproduction specification

This document defines the two experiments selected for this reproduction:

1. the `16 x 16` two-dimensional ferromagnetic Ising model; and
2. the fcc Cu-Ni crystal/alloy experiment at `N=108`, including the reported `N=256`
   size-transfer test.

It separates settings stated by the paper from reconstruction choices. The local primary sources are
`reference/JANUS/arXiv-2608.19116v1.tar.gz::{main.tex,SI.tex}` and
`reference/JANUS/janus_paper.pdf`.

## 1. What JANUS learns

JANUS learns a conditional generator for equilibrium states, amortized over thermodynamic
conditions. It starts from a simple prior and repeatedly trains on terminal configurations generated
by the current model. The interatomic Hamiltonian or potential labels those terminals; no reference
MCMC samples are used as training targets.

The state has three possible channels:

- discrete site identities `a`;
- continuous fractional displacements `u`; and
- continuous log-volume `v = log(V)`.

The Ising experiment uses only `a`. Cu-Ni activates all three channels.

### Discrete channel

Every site starts as `MASK`. At interpolation time `t`, a terminal token remains visible with
probability `t` and is masked with probability `1-t`. The model predicts the terminal-site posterior
`q_theta(a_i | a_t, x_t, T, delta_mu)`. Its target is the exact single-site heat-bath distribution

```text
rho_i(b) proportional to exp(beta * [mu_b - U(a with site i replaced by b, u, v)]).
```

At inference, masked sites are revealed monotonically. For a time grid, a still-masked site is
revealed with probability

```text
p_n = [alpha(t_{n+1}) - alpha(t_n)] / [1 - alpha(t_n)],
```

and its token is drawn from `q_theta`. A revealed token is never masked again. The paper uses the
linear schedule `alpha(t)=t`.

### Continuous channels

For Cu-Ni, JANUS linearly interpolates fresh prior and terminal continuous states:

```text
x_t = (1-t) x_0 + t x_1,  x in {u, v}.
```

It regresses velocity heads `b_u,b_v` and score heads `s_u,s_v`. Sampling uses

```text
dx = [b_theta + g(t)^2 s_theta] dt + sqrt(2 g(t)^2) dW.
```

The bounded score target blends the exact prior score and terminal force/virial score with

```text
c(t) = t^2 / [t^2 + (1-t)^2].
```

For `v=log(V)`, the NPT density and Metropolis acceptance include the required `N v` Jacobian.

## 2. Experiment A: 16 x 16 ferromagnetic Ising model

### 2.1 Task and target distribution

Each of the `N=256` periodic square-lattice sites has a binary species/token, equivalently a spin
`sigma_i in {-1,+1}`. With `epsilon_AA=epsilon_BB=0` and `epsilon_AB=2`, the model maps to the
ferromagnetic Ising Hamiltonian with coupling `J=1` and field `h=delta_mu/2`:

```text
H(sigma) = -J sum_<ij> sigma_i sigma_j - (delta_mu/2) sum_i sigma_i.
```

The critical temperature at zero field is

```text
k_B T_c / J = 2 / log(1 + sqrt(2)) = 2.2692.
```

The task is not classification. It is conditional equilibrium generation: one model must generate
correct spin configurations throughout the temperature-field plane. Below `T_c` and at zero field,
the distribution is bimodal between positive and negative magnetization; above `T_c`, it is
disordered. Non-zero field biases one phase.

Paper sources: `main.tex`, Fig. 2a discussion; `SI.tex`, lattice-model paragraph around lines
`571-578` in the extracted source.

### 2.2 JANUS training used by the paper

- Prior: every site is `MASK`.
- Network: periodic fully convolutional model, `4-6` circular `3x3` layers, `64-96` channels,
  approximately `0.2-0.8M` parameters.
- Conditioning: time, temperature and chemical-potential difference; global spatial means are
  re-injected at each layer.
- Output: a zero-initialized `1x1` species-logit head. Zero initialization initially gives fair-coin
  completions.
- Generation: `128` reveal blocks during training.
- Outer loop: several hundred rounds, with `512` generated chains per round.
- Labels: exact Ising heat-bath probabilities are cheap and recomputed on the fly.
- Optimizer: Adam, learning rate `3e-3`.

One round is:

1. draw `(T, delta_mu)` conditions;
2. generate terminal lattices with the current masked model;
3. draw `t`, independently re-mask terminal sites with probability `1-t`;
4. calculate exact single-site heat-bath probabilities from the Ising Hamiltonian; and
5. minimize masked-site soft cross-entropy.

Reference samples never enter this loss. They are validation data only.

The Supplement's unqualified conditioning rule samples **inverse temperature uniformly** over the
training window. This applies to the Ising model as well as the alloys; uniform sampling in `T` is
not paper-faithful. For Ising, the exact coexistence line is `delta_mu=0`. The Supplement does not
state the precise Ising field mixture or whether this line receives an atomic sampling mass; that
part remains a reconstruction choice.

### 2.3 Published reference protocol

- lattice: `L=16`;
- temperatures: `11` values over `[1.5, 3.2]`, including the critical region;
- non-negative fields: `11` values of `delta_mu` over `[0, 0.4]`;
- negative fields: obtained by exact spin/particle-hole symmetry;
- sampler: ghost-spin Wolff cluster Monte Carlo;
- per grid state: `24` independent chains, `3,000` cluster steps per chain;
- burn-in: first `600` cluster steps;
- correctness check: exact enumeration on `4x4` lattices.

The source does not list the eleven individual field values. Fig. 2a shows that the displayed grid
is nonuniform (the `0.02` tick is two columns from zero, not five percent of a linear axis). The
current explicit reconstruction is
`[0,.01,.02,.04,.06,.08,.10,.15,.20,.30,.40]`, mirrored to negative fields, and plotted with equal
column widths as in the paper. This list must remain labeled as figure-reconstructed unless the
authors provide the original values.

### 2.4 Metrics and figures to reproduce

The required Fig. 2a outputs are:

1. **Spin-up population map**
   - `P(sigma=+1)` over `(T, delta_mu)`;
   - JANUS and ghost-Wolff reference shown on the same grid;
   - negative-field half obtained by symmetry.
2. **Zero-field absolute magnetization curve**
   - `E[|m|]`, where `m=N^{-1} sum_i sigma_i`;
   - plotted against `T` at `delta_mu=0`;
   - must show the transition around `T_c=2.2692` and agree with Wolff within uncertainty.
3. **Representative configurations**
   - samples below, near and above `T_c`, and optionally under positive/negative field;
   - used to verify ordered domains, critical structure and disordered states qualitatively.

Reproduction diagnostics, even when not explicitly drawn in the paper, must include:

- mean and maximum absolute spin-up population error over the grid;
- zero-field `|m|` error per temperature, with special reporting near `T_c`;
- uncertainty across chains/seeds;
- exact `4x4` observable agreement for both JANUS importance-weighting checks and Wolff validation;
- training loss and checkpoint/seed provenance.

### 2.5 Status of the first local run

The first saved run did use `L=16`, `128` reveal blocks, `512` chains, `300` outer rounds and the
published Wolff grid. It is a completed baseline execution, not a successful reproduction:

- overall mean spin-up population error: `0.05201`;
- at `delta_mu=0, T=2.18`: JANUS `|m|=0.527`, Wolff `0.814`;
- at `delta_mu=0, T=2.35`: JANUS `|m|=0.445`, Wolff `0.594`.

The critical-region discrepancy is too large to claim Fig. 2a reproduction.

## 3. Experiment B: fcc Cu-Ni crystal/alloy

### 3.1 Task and target distribution

The model generates a substitutional fcc alloy in the isobaric semi-grand-canonical ensemble. A
sample contains:

- Cu/Ni identity on every lattice site;
- atomic displacement from the reference fcc lattice; and
- isotropic cell volume.

It is conditioned on temperature `T` and chemical-potential difference
`delta_mu = mu_Cu - mu_Ni` at pressure `P=0`. The target in log-volume coordinates is proportional to

```text
exp{-beta [U(a,u,v) - delta_mu N_Cu]} * exp(N v).
```

This is the core hybrid JANUS test because the discrete, vector and scalar-volume channels are all
active simultaneously.

### 3.2 Published physical and model settings

- structure: `3x3x3` conventional fcc supercell, `N=108`;
- size transfer: apply the trained model to `N=256` without retraining;
- pressure: `P=0`;
- temperature window: `600-1200 K`;
- condition centre:

```text
delta_mu_hat(T) = 0.893 eV - 5.4e-5 eV/K * T;
```

- training offsets: mixture of wide `+/-0.30 eV` and narrow `+/-0.06 eV` windows around the centre;
- potential: Fischer-Schmitz-Eich Cu-Ni EAM from NIST;
- local file: `potentials/cu_ni/Cu_Ni_Fischer_2018.eam.alloy`;
- local SHA-256:
  `ee585bf9884a4ac2548abb395e81a12a941505983aaa058ef546929df74cf240`;
- interaction cutoff used by JANUS network: `5.0 Angstrom`;
- network: four PaiNN-style interaction layers, `64` scalar and `64` vector channels, `16` Gaussian
  radial basis functions, approximately `2.7e5` parameters;
- heads: `b_u,s_u,b_v,s_v` and per-site Cu/Ni logits;
- discrete loss weight: `lambda=2`; continuous displacement and volume channels equally weighted;
- rollout steps: `M=100`;
- optimizer: Adam, learning rate `3e-3`.

The paper reports `100-120` rounds, `1,000` fresh Cu-Ni terminals per round, replay capacity
`2,000-5,000`, `500` gradient steps per round and batch size `64-96`. Its stated total oracle budget
is `1.25e5`. The consistent maximal-budget reconstruction is an initial buffer of `5,000` plus
`120 x 1,000` fresh terminals.

### 3.3 Prior calibration

The prior is fitted using only the EAM, not reference MCMC labels used for evaluation.

- Composition estimate used to condition the **continuous** priors (species starts all-mask and the
  zero-initialized species head initially gives fair-coin completion):

```text
c_0 = sigmoid([delta_mu - delta_mu_hat(T)] / [k_B T]).
```

- Mean atomic volume: fit relaxed random alloys at compositions
  `{0, 1/4, 1/2, 3/4, 1}` plus short fixed-composition NPT runs to a pure-element interpolation,
  regular-solution excess volume and linear thermal expansion.
- Displacement width: calculate the EAM Hessian around relaxed configurations and fit the isotropic
  quasi-harmonic width.
- Log-volume width: estimate from the calibrated thermal volume fluctuations.

The numerical fitted widths are not published; independently fitting them with the cited EAM is a
reconstruction step, not an added source of reference data. Two ambiguities must remain explicit:

1. Eq. S34 fixes `sigma_u(T)=sigma_u_ref sqrt(T/T_ref)`, while the following calibration paragraph
   says to fit an exponent `p`. The primary equation-faithful run must use `p=0.5`; the fitted-`p`
   version is a separately reported calibration ablation.
2. The SI does not give a numerical `sigma_v` or a complete fitting recipe. Our reconstruction uses
   the equal-state RMS of within-state `log(V)` standard deviations from the short fixed-composition
   NPT calibration grid. Its value, state weighting and sensitivity scaling must be saved and swept;
   it cannot be presented as an author-provided setting.

The numerical diffusion strengths `g_u(t)` and `g_v(t)` are likewise unpublished. Our
implementation exposes constant base strengths, applies the published `sqrt(T/T_ref)` scaling,
and sweeps them as reconstruction choices. `g=0` remains the paper's exact deterministic
probability-flow baseline.

### 3.4 Published reference protocol

For `N=108`:

- `15` temperatures from `500-1200 K`;
- `33` chemical potentials from `0.600-1.150 eV`, with `10 meV` spacing through the composition
  crossover and `25 meV` in pinned wings;
- two walkers per state, initialized from all-Ni and all-Cu;
- `40,000` sweeps, first `2,000` discarded;
- six site-flip attempts per sweep;
- displacement and log-volume moves each sweep;
- burn-in-only step adaptation toward acceptance approximately `0.3`, then frozen;
- no replica exchange for the production `N=108` Cu-Ni run.

For `N=256` transfer:

- `7` temperatures and `11` chemical potentials;
- temperature exchange every other sweep;
- `24,000` sweeps, first `12,000` discarded.

The paper does not list the exact eleven `N=256` chemical-potential values.

### 3.5 Metrics and figures to reproduce

The required Fig. 2b-e outputs are:

1. **Equilibrium Cu concentration**
   - `E[x_Cu]` over `(T, delta_mu)`;
   - JANUS versus semi-grand reference;
   - include the `x_Cu=0.5` contour/crossover.
2. **Mean atomic displacement**
   - `E[|u|]` at `600, 900, 1200 K`;
   - JANUS curves/points against reference values.
3. **Atomic volume**
   - `E[V/N]` at `600, 900, 1200 K`;
   - JANUS against reference.
4. **Partial radial distribution functions**
   - `g_Cu-Cu(r)` and `g_Ni-Ni(r)` at `800 K` for representative compositions;
   - JANUS and reference on identical bins/normalization.
5. **Mixing free energy**
   - `G_mix(x,T)` derived from importance/path weights or equivalent validated reweighting;
   - report uncertainty and effective sample size.
6. **Size transfer**
   - compare the same trained model at `N=108` and `N=256` against their references;
   - report composition/thermodynamic error and throughput.
7. **Efficiency**
   - count target energy-force-virial-substitution oracle calls exactly;
   - compare against reference sampling under the paper's accounting convention.

## 4. Published settings versus reconstruction choices

| Item | Paper status | Reproduction policy |
| --- | --- | --- |
| Ising Hamiltonian/grid/Wolff protocol | Published | Match exactly |
| Ising outer rounds | Only “several hundred” | Choose using validation convergence and report |
| Ising gradient updates per round | Not published | Treat as a tunable implementation parameter; report explicitly |
| Ising temperature sampling | Uniform in inverse temperature is stated without an alloy-only restriction | Sample `beta=1/T` uniformly |
| Ising field sampling | Field window and coexistence line are stated; exact training mixture is absent | Report the chosen field law and any explicit mass at `delta_mu=0` |
| Cu-Ni potential | Citation published, exact implementation not named in text | Use current corrected NIST `ipr3`, checksum it and report provenance |
| Cu-Ni prior widths | Fitting procedure published, numbers absent | Refit from the fixed EAM and save calibration artifacts |
| Cu-Ni displacement exponent | Eq. S34 says `p=0.5`; prose says fit `p` | Primary `p=0.5`, fitted `p` as an ablation |
| Cu-Ni `sigma_v` | Value and exact recipe absent | Document RMS log-volume-fluctuation reconstruction and sweep its scale |
| `g_u(t),g_v(t)` base schedules | Not published | Compare simple documented candidates using held-out reference/ESS |
| Adam betas, clipping, EMA, seed/dtype | Not published | Choose conservatively and report |
| N=256 exact chemical-potential list | Not published | Keep explicit; do not claim exact grid identity without author data |

## 5. Are more gradient updates or changed condition sampling part of JANUS?

The distinction is important:

- **Repeated gradient updates are part of JANUS.** The algorithm explicitly performs a fixed number
  of regression steps per outer generate-label round, and the alloy setup states `500` updates per
  round.
- **The paper does not publish the number of gradient updates per Ising round.** The first local run
  used `10`; increasing that number is filling an omitted optimization hyperparameter, not changing
  the JANUS objective. It must be reported as a reconstruction choice and justified by convergence.
- **Uniform inverse-temperature sampling is stated by the paper and must be used for Ising.** The
  earlier local run's uniform-`T` sampling was incorrect.
- **Critical-region oversampling beyond uniform inverse temperature is not stated.** The precise
  Ising field-sampling mixture is also not published. Deliberately oversampling a temperature band,
  or assigning extra probability to `delta_mu=0`, must be reported as a reconstruction choice or
  ablation rather than a quoted paper setting.

Therefore the next faithful Ising run must sample uniformly in inverse temperature, use a documented
field law, and increase/validate the omitted update budget. It must use an acceptance criterion based
on the held-out Wolff grid, especially zero-field `|m|` near `T_c`. Additional temperature-focused
sampling should be tested only as a clearly labeled deviation.

### Explicit BMS-style damping

The reference BMS repository supports a damped fixed-point variant
`L = L_matching + eta ||u_theta-u_previous||^2`, but its supplied configuration defaults to
`damping=0`; `eta=10` is documented as an optional damped variant. JANUS does not state an explicit
previous-model penalty or damping coefficient in either the main text or Supplement. Its replay
buffer mixes terminals from current and earlier rounds and can soften fixed-point updates, but this
is not the same objective. Primary JANUS reproduction therefore uses explicit `eta=0` and the paper
replay buffer. An `eta=10` run may be useful only as a clearly labeled BMS-inspired ablation.

## 6. Completion criteria

An experiment is not complete merely because its process exits.

Ising completion requires:

- exact `4x4` reference validation;
- all required Fig. 2a artifacts;
- uncertainty-aware agreement over the grid;
- explicit critical-region `|m|` acceptance;
- checkpoint, resolved configuration, environment and W&B history.

Cu-Ni completion requires:

- EAM oracle cross-check against ASE for energy, forces and stress;
- saved prior-calibration fits;
- converged two-walker reference grids;
- all Fig. 2b-e metrics and plots with uncertainty/ESS;
- `N=256` size-transfer evaluation;
- exact oracle-call accounting and reproducible checkpoints/configuration.
