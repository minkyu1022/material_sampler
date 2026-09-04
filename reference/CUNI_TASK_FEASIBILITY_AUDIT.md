# Cu–Ni continuous CSP and joint realCSP: feasibility audit

## Verdict

Both projects are implementable, but they are not equally routine.

- **Continuous-only CSP + BMS:** feasible with high confidence and maximal reuse of Crystallite/JANUS.
- **Continuous-only CSP + PDNS:** implementable, but the exact target distribution must be fixed first because the required reference terminal-density term is not available from Crystallite for free.
- **Joint masked species + continuous geometry + BMS/soft-CE:** feasible research implementation; the released Crystallite type diffusion must be replaced rather than lightly configured.
- **Joint continuous+discrete PDNS:** theoretically plausible, but neither released codebase supplies the coupled algorithm. It requires a new joint path-weight derivation and cannot honestly be described as “use PDNS code as-is.”

## A. Shared relaxed dataset

### Required construction

1. Locate every retained Cu–Ni reference-MC structure and its state/trajectory provenance.
2. Relax with one frozen oracle/configuration, recording potential hash, cutoff, cell/position relaxation flags, optimizer, force/stress tolerances and convergence status.
3. Store both the finite-temperature parent and relaxed child IDs. Never overwrite the MC dataset.
4. Canonicalize only the cell/coordinate representation needed by Crystallite; do not independently reorder species and coordinates.
5. Deduplicate relaxed structures within each exact composition `n_Cu` using progressively stronger checks:
   energy/volume fingerprint, StructureMatcher, then structural descriptors for diversity statistics.
6. Split train/validation/test by parent trajectory/state block before any random sample-level split, because adjacent MC frames and relaxed descendants are correlated.

“Relax everything” does not guarantee a useful pretraining dataset. Many finite-temperature frames may collapse to the same local minimum. The post-relaxation unique count, basin population imbalance and train/test leakage risk must be measured before training.

### Composition and diversity report

For every `n_Cu` group report:

- raw frame count, converged relaxation count and failure rate;
- unique relaxed structures and duplication ratio;
- energy/atom, volume/atom and RMS displacement distributions;
- pair/RDF or local-environment descriptor dispersion;
- cluster count and effective cluster population;
- source-chain/state coverage.

## B. What Crystallite preprocessing actually does

Crystallite accepts either a CIF string or `positions + cell + atomic_numbers`.

- CIF input is parsed with Pymatgen, optionally primitive-reduced, then Niggli-reduced.
- Array input is first made into a Pymatgen `Structure`, serialized to CIF, and passed through the same parser/reduction path (`preprocessing_utils.py:287-325`). Thus array input currently does make one CIF round trip.
- The resulting structure is converted directly to `atom_types` and `frac_coords` (`preprocessing_utils.py:376-431`), padded without an explicit species sort (`mp20_tokens.py:240-275`).
- Therefore Crystallite does **not** deliberately turn `[Cu, Ni, ..., Cu]` into `[Cu, Cu, ..., Ni, Ni]`. It preserves the site order returned by the canonicalized Pymatgen structure, with each species still paired to its corresponding coordinate.
- Training reads cached tensors, not CIF files (`mp20_tokens.py:180-192`). CIF is an ingestion/canonicalization representation, not the network input.
- A random global fractional translation is applied as augmentation when enabled (`mp20_tokens.py:121-134`).

For our dataset, use this same pipeline and add an invariant test that species-coordinate pairs survive conversion. Do not add an element sort merely for grouping; grouping metadata and site order are separate concerns.

## C. `cont_task`: fixed-composition Cu–Ni CSP

### Pretraining

Use the Crystallite CSP path:

- atom identities fixed;
- fractional coordinates and six-dimensional lattice latent denoised;
- same EDM preconditioning, channel losses, GEM, EMA and Karras sampling;
- lower-triangular six-variable cell representation only if selected consistently in data, loss, sampler and checkpoint configuration.

Grouping by `n_Cu` is useful for reporting, balanced sampling and conditional evaluation. It does not require separate models per composition; a single CSP model can receive the full fixed atom-type sequence.

### BMS post-training with one clean-endpoint head

This is mathematically viable for the chosen deterministic linear interpolant and Gaussian prior. Let

`x_t = (1-t)x_0 + t x_1`, and let the existing head predict `D_theta(x_t,t) ≈ E[x_1|x_t]`.

Then the same output defines

- velocity: `b_theta = (D_theta - x_t)/(1-t)`;
- for isotropic Gaussian `x_0 ~ N(0, sigma_0^2 I)`, score:
  `s_theta = -(x_t - t D_theta)/((1-t)^2 sigma_0^2)`.

Thus two separate learned heads are not mandatory. The JANUS/BMS velocity loss and generalized target-score loss can both act on the same `D_theta`. This tying is a modeling restriction, so velocity and score errors must be logged separately.

Required qualifications:

- fractional coordinates need a consistent unwrapped-cover interpolation and minimum-image loss convention;
- translation/COM gauge must be removed consistently;
- the cell prior need not be isotropic, so its score conversion must use its covariance in the actual `ltri` variables;
- MLIP Cartesian forces and virials must be chain-ruled into fractional coordinates and cell latent;
- the BMS rollout is not identical to the pretrained EDM sampler, so initialization and time conversion must be explicit.

### Continuous PDNS and the `log nu` caveat

The attached caveat correctly identifies the formula but makes one conditional choice sound mandatory.

For a pretrained reference terminal law `nu = p_base`, PDNS defines

`r(x) = -beta H(x) - log nu(x)`

to obtain the **bare physical target** `pi(x) ∝ exp(-beta H(x))`.

If `log nu` is removed, the method instead targets

`p_tilt(x) ∝ p_base(x) exp(-beta H(x))`.

That is a valid energy-tilted pretrained distribution, and endpoint-only tilting avoids the memoryless simplification. It is **not** the same target as equilibrium Boltzmann sampling. Therefore:

- do not remove `log nu` silently;
- expose two explicitly named objectives: `bare_boltzmann` and `base_tilt`;
- mark `bare_boltzmann` blocked until `log p_base` is evaluable/estimated or an alternative analytic reference construction is derived;
- `base_tilt` can be implemented first if the scientific goal accepts prior regularization.

The caveat's rollout warning is correct: Crystallite's churn-first nonlinear Heun pushforward does not admit PDNS's simple Euler Gaussian Girsanov ratio. PDNS needs a dedicated stochastic kernel with common end-of-step Gaussian noise for base/current drifts. Start with Euler; add Heun only after its transition law is derived and tested.

## D. `joint_task`: Cu–Ni realCSP

### Conditioning

The requested condition is `(N, allowed element set)`, while exact counts are generated. For a Cu–Ni-only dataset, `[Cu,Ni]` is constant and carries no learning signal; it is still a useful support mask. It becomes a meaningful condition only when training across multiple allowed element sets.

Reuse Crystallite's element encodings for element embeddings, but not its atom-generation process. Released DNG continuously diffuses subatomic tokens and nearest-token decodes them; it has no mask token/reveal process.

### Masked species pretraining

Implement a categorical masked diffusion over the allowed set:

- only real sites among the known `N` participate;
- each site's logits are hard-masked to the conditioned element set;
- forward corruption/reveal schedule and reverse rate are explicit;
- coordinate/cell EDM time and discrete mask time must share a documented coupling;
- randomized joint site permutation is preferable to species sorting so the model cannot exploit an arbitrary Cu-first convention.

This is a new training objective sharing the Crystallite trunk, not “Crystallite DNG unchanged.”

### JANUS-style post-training

- Continuous geometry/cell: BMS velocity plus generalized score loss.
- Species: soft cross-entropy from substitution-energy conditionals restricted to the allowed set.
- A single joint model/trunk may serve both, with appropriate output parameterization.
- The target Hamiltonian, temperature/pressure factors and all Jacobians must be identical across continuous forces, cell gradients and substitution energies.

The soft-CE substitution labels condition on the current geometry. Their oracle cost scales with sites × allowed alternatives, although binary Cu–Ni permits batching one alternative per site.

### Joint continuous + discrete PDNS

PDNS contains continuous and discrete implementations, but they are separate experiments. A coupled crystal target requires:

1. a joint reference process for mask jumps and stochastic continuous motion;
2. a joint path RN derivative, normally the sum of continuous Girsanov and discrete jump-process log ratios;
3. one shared terminal reward/energy evaluated on the complete crystal;
4. synchronized proximal tempering and condition-wise normalization;
5. a joint bridge/re-noising loss with no missing cross-channel conditioning.

This is implementable, but it needs a written derivation and analytic toy validation before MLIP-scale training. Copying the two PDNS loops side-by-side would generally be incorrect because the energy couples species, positions and cell.

## E. Folder contract

Use the user-created roots and keep shared code minimal:

```text
cont_task/
  data/{raw_mc,relaxed,processed,reports}/
  pre_train/{configs,scripts,outputs,ckpt}/
  post_train/
    bms/{configs,scripts,outputs,ckpt}/
    pdns/{configs,scripts,outputs,ckpt}/

joint_task/
  data/                       # link/manifest to shared relaxed data, no duplicate copy
  pre_train/{configs,scripts,outputs,ckpt}/
  post_train/
    janus_bms_softce/{configs,scripts,outputs,ckpt}/
    pdns_joint/{configs,scripts,outputs,ckpt}/
```

Do not fork the full Crystallite codebase into every method directory. Import/reuse the reference implementation through one project package, while method directories own only method-specific configs, adapters, entrypoints and artifacts.

## F. Recommended execution order

1. Inventory the existing Cu–Ni MC data and lock the relaxation Hamiltonian.
2. Build/deduplicate the relaxed dataset and issue the composition/diversity report.
3. Add a dataset adapter and preprocessing invariance tests.
4. Reproduce Crystallite CSP pretraining on fixed compositions.
5. Implement/test continuous BMS one-head conversion on a Gaussian toy, then Cu–Ni.
6. Implement continuous PDNS `base_tilt`; implement `bare_boltzmann` only with a defensible `log nu` solution.
7. Replace DNG type diffusion with conditioned masked diffusion and pretrain joint realCSP.
8. Add JANUS BMS + soft-CE.
9. Derive and toy-test coupled continuous/discrete PDNS before implementing the joint MLIP run.

This order produces a usable continuous sampler early and prevents the most speculative joint-PDNS work from blocking the rest.
