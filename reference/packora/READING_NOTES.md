# Packora (arXiv:2608.26962) reading notes

Source archive: `arXiv-2608.26962.tar.gz`  
SHA-256: `332e5e0a3f6e4b90a370f2572ecf0b5c1532200c64c634d725024c6814e6bbe0`

## Scope

Packora is a **molecular** CSP model, not an inorganic variable-composition model. It conditions on molecular graphs, stoichiometry, formula-unit count `Z`, and optionally molecular conformers, stereochemistry, and space group. It jointly generates all-atom Cartesian coordinates and a six-dimensional cell latent.

## State and parameterization

- Crystal state: atomic numbers `A`, Cartesian coordinates `X`, and a cell whose rows are lattice vectors.
- Cell canonicalization: Niggli reduction, then the Cholesky factor of the reduced-cell Gram matrix.
- Cell latent: six unconstrained lower-triangular entries with log-positive diagonal, explicitly following Crystalite.
- Coordinate output is centered by subtracting the valid-atom mean, removing global translation.
- Unlike our Cu–Ni Crystalite path, Packora uses centered **Cartesian** coordinates rather than wrapped fractional coordinates.

## Objective and one-head implication

Packora uses a linear conditional path

`y_t = (1-t) y_0 + t y_1`

and predicts the clean endpoint `mu_theta = y1_hat`. A fully factorized fixed-scale Laplace variational posterior reduces training to a weighted endpoint `L1` loss:

- coordinate loss normalized by `3N`, weight 10;
- lattice loss normalized by 6, weight 1.

For the linear path, the same clean-endpoint prediction is converted at sampling time to

- `v_theta = (mu_theta - y_t)/(1-t)`;
- `s_theta = (t v_theta - y_t)/(1-t)` for their centered standard-Gaussian prior.

This is direct evidence that one clean-endpoint head can parameterize both velocity and score. It does **not** remove the need to insert our actual prior mean/covariance and channel transformations for JANUS/BMS post-training.

## Architecture

1. Build condition-only single and pair representations.
2. Refine pairs with four Pairmixer blocks (outgoing/incoming triangle multiplication plus transition; no triangle attention).
3. Inject noisy coordinates/cell into the single representation **after** pair refinement.
4. Process with 16 DiT blocks, using the cached pair representation as attention bias.
5. Separate endpoint heads predict centered coordinates and cell latent.

The selected post-entry design is slightly weaker than pre-entry in raw solve rate but makes the condition-only pair representation cacheable. The paper reports a 20.1x reduction in 200-step sampling time on a synthetic `[batch=64, N=300]` case, at somewhat higher peak memory.

Selected sizes:

| Model | Params | Single width | Pair width | Heads | Pairmixer | DiT |
|---|---:|---:|---:|---:|---:|---:|
| Packora-M | 88M | 512 | 128 | 8 | 4 | 16 |
| Packora-L | 187M | 768 | 192 | 12 | 4 | 16 |

The scaling study finds that increasing only the single or pair width is inferior to balanced scaling at fixed `single:pair = 4:1`.

## Training recipe

- Prior: centered Gaussian for `X`, standard Gaussian for cell latent.
- Time: `0.98 Beta(1.9,1.0) + 0.02 Uniform(0,1)`.
- Random crystal translation; template rotation/translation.
- Muon for hidden matrices, AdamW otherwise.
- LR `1e-4`, 10k-step warmup, global gradient clipping 1.0.
- EMA `0.9999`.
- Weight decay: 0 for M, `1e-2` for L.
- Optional-condition dropout: template 0.5, stereochemistry 0.5, space group 0.9.
- No auxiliary pair-distance loss in the selected recipe.
- Mixed bf16 on eight H200 GPUs.
- Two-stage atom-count curriculum with fixed-shape buckets for `torch.compile`:
  - stage 1: up to 300 atoms;
  - stage 2: up to 512 atoms.
- Per-GPU batch is 16 through 300 atoms; global batch is 128 on eight GPUs. It decreases for larger buckets.

## Sampling

- EDM-Heun with stochastic churn and Karras schedule.
- 200 solver steps = 400 denoiser NFEs.
- `sigma_min=0.002`, `sigma_max=80`, `rho=7`.
- `S_churn=60`, `S_min=0`, `S_max=999`, `S_noise=1.003`.
- EMA weights, 1,000 candidates/target, seed 42.
- EDM-Heun materially outperformed matched-NFE flow ODE and Euler-Maruyama SDE; increasing from 400 to 1,000 NFE brought only a small gain.
- Autoguidance did not help.

## Dataset and evaluation

- Main training split: 912,807 CSD structures after adding missing hydrogens and enforcing <=512 atoms; validation: 1,047.
- Mean/median atoms per training cell: 201.71/180.
- Generation and ranking are evaluated separately.
- Generation metric: crystal-level solve rate `Sol_C`; a target is solved when any candidate is collision-free and COMPACK matches at least 8/15 molecules with `RMSD15 < 2 Å`. Candidate budgets are 30 and 1,000.
- Ranking: candidates are relaxed with UMA-S-1.2/OMC using ASE BFGS (`fmax=0.02 eV/Å`, max 500 steps), filtered/deduplicated, then ranked by lattice energy per formula unit. Strict recovery uses 30 matched molecules and `RMSD30 < 1 Å`.

## Relevance to this repository

Useful and transferable:

- the same lower-triangular six-DOF cell canonicalization inherited from Crystalite;
- explicit clean-endpoint-to-velocity/score reuse;
- centered-coordinate output;
- fixed-shape atom-count batching;
- balanced capacity scaling rather than widening only one track;
- separating raw generation coverage from post-relaxation energy ranking;
- cached condition-only pair reasoning for long multi-step sampling.

Not directly transferable:

- molecular graph/bond/template conditioning is absent from elemental Cu–Ni;
- Cartesian molecular coordinates and COMPACK metrics do not replace our fractional-coordinate/PBC treatment;
- Packora's standard-Gaussian score identity must be generalized to our calibrated priors;
- its ranking workflow and lattice energy target are molecular-crystal-specific.

## Primary source locations

- `sections/method.tex`
- `appendix/model_architecture.tex`
- `appendix/training_sampling_hyperparameters.tex`
- `sections/ablations/{model_architecture,model_scaling,training,inference}.tex`
- `sections/benchmarks/{setup,structure_generation,structure_ranking}.tex`
- `appendix/{dataset_preprocessing,dataset_statistics,structure_generation_evaluation}.tex`
