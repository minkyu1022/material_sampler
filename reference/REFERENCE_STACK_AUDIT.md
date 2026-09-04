# Crystallite / PDNS / Algorithm Notes: implementation map

## Scope

Reviewed locally:

- Crystallite paper source (`arXiv-2604.02270v2`) and implementation under `reference/crystalite`.
- PDNS paper source (`arXiv-2510.03824v2`) and continuous/discrete implementations under `reference/PDNS`.
- All three documents under `reference/algorithm_scripts`:
  `brief_explanation.pdf`, `with_proof_1.pdf`, and `with_proof_2.pdf`.

This is a design/audit pass only. Reproducing PDNS is explicitly out of scope; its existing code is the implementation reference.

## 1. Crystallite DNG as implemented

### State and network output

Crystallite jointly models:

- atom-type features `H`,
- fractional coordinates `F`,
- a six-dimensional lattice latent `y`.

The Transformer emits three raw heads, but EDM preconditioning converts them into clean-state denoiser predictions
`D_type`, `D_frac`, and `D_lat`. Therefore the externally meaningful output is an estimate of the clean endpoint, not a flow velocity or control. See `reference/crystalite/src/crystalite/edm_utils.py:63-134`.

Training independently adds Gaussian noise to all three channels using one sampled EDM noise level `sigma`; coordinates are centered fractional coordinates and are left unwrapped during forward noising. See `reference/crystalite/src/train_crystalite.py:452-500`.

### Coordinates and cell

- Coordinates remain fractional. Periodicity is handled by wrapping model inputs and minimum-image residuals, not by converting the learned state to Cartesian coordinates.
- Sampling wraps fractional states after every Euler/Heun update (`sampler.py:193-218`).
- The implementation supports both the paper-style six-value `Y1` representation and an unconstrained six-value lower-triangular (`ltri`) latent.
- `ltri` maps three diagonal variables through `exp` and uses three unconstrained shear entries, guaranteeing a positive-volume cell (`models/lattice_repr.py:73-94`).

### Sampler

Sampling is Karras EDM with optional churn and a Heun correction. Atom type, coordinate, and lattice updates are simultaneous at every noise step; it is not an alternating discrete-unmask/continuous-update process. Channel-wise anti-annealing is an inference-time drift rescaling and does not change the training loss (`crystalite/sampler.py:53-250`).

### Atom types

The released DNG model uses continuous subatomic token features and nearest-token decoding. It does **not** use masked categorical diffusion. Restricting generation to a supplied alloy element set therefore requires a deliberate new type-channel design (or constrained decoding); it is not already present in the released DNG path.

## 2. PDNS theory and code

### What proximal cross-entropy is

PDNS performs a path-space proximal step rather than tilting the current proposal all the way to the target in one round. For current path law `P^(k-1)`, terminal energy and the path RN derivative form a tempered importance weight. Weighted or resampled terminal endpoints are then used in a denoising/control matching objective. Repeating this outer stage yields incremental movement toward the energy target.

The paper provides two practically equivalent variants:

1. weight-based PCE: retain normalized weights in the matching loss;
2. resampling-based PCE: categorical-resample endpoints by those weights, then train unweighted.

### What the released continuous code does

The code implements the resampling-based route by default when `iws=true`:

1. rollout the current controlled SDE and accumulate a path log RN term (`components/matchers.py:37-63`);
2. combine it with terminal cost and a fixed/adaptive proximal exponent (`components/scheduler.py:22-94`);
3. categorical-resample endpoints and reset their training weights to one (`matchers.py:78-96`);
4. sample a fresh source endpoint and bridge time, obtain the reference bridge conditional score (`matchers.py:101-112`);
5. regress a control/adjoint parameterization with a weighted quadratic objective (`src/train_loop.py:29-60`);
6. regenerate the buffer at each stage (`continuous/train.py:124-220`).

The DW4 configuration uses 1,000 SDE steps, a 50,000 endpoint buffer, 1,000 epochs per stage, adaptive KL radius, and an initially annealed reference (`configs/experiment/dw4.yaml`). These are task settings, not universal PCE constants.

### Paper/code cautions

- The repository README calls this renewed code, and the Crystallite README calls its code an early refactored release. Paper-to-code equivalence must therefore be checked at the function level, not assumed from repository names.
- PDNS's continuous implementation learns a control/adjoint field. It does not demonstrate an EDM clean-endpoint (`x_hat`) parameterization.
- Its theory assumes a known reference path measure and, in the presented simplification, a memoryless reference endpoint coupling. A Crystallite EDM sampler must supply the corresponding transition/bridge and path-density terms before the same weights are valid.
- The repository itself documents a corrected GMM40 Sinkhorn evaluation, so old paper table values and current evaluation code are not identical for that metric.

## 3. Meaning of the algorithm documents

The three PDFs are consistent layers of the same theory:

- `brief_explanation.pdf`: algorithm-level map for energy-specified targets. It separates weighted DSM, proximal CE, adjoint matching, continuous BMS, and their discrete counterparts.
- `with_proof_1.pdf`: continuous derivations—Girsanov weights, weighted DSM, PCE, adjoint matching, density control, and BMS/JANUS.
- `with_proof_2.pdf`: discrete derivations—chain Girsanov/Feynman–Kac weights, weighted target concrete score matching, discrete PCE, soft CE, discrete ASBS, and discrete BMS.

The proximal CE in these notes is the same core method as the PDNS paper: a KL-proximal path-law update whose tempered RN/energy weight is used for weighted or resampled denoising matching. The notes are broader than PDNS and also describe JANUS/BMS-style fixed-point methods; those methods should not be conflated with PCE merely because both use outer rollout/retrain loops.

## 4. Consequences for Crystallite energy post-training

### What can be reused unchanged

- Transformer/GEM trunk.
- DNG state representation: subatomic atom tokens, fractional coordinates, and six cell variables.
- `ltri` lattice parameterization if that is the desired final representation.
- Padding, atom-count conditioning, periodic input geometry, and existing checkpoint/EMA infrastructure.

### What cannot be copied blindly from PDNS

- PDNS `ScoreMatcher`, because it assumes flat Euclidean SDE states and a control/adjoint output.
- PDNS path weights, until the exact Crystallite reference dynamics, stochasticity, and transition score/log-density are defined.
- A single scalar channel formula without accounting for periodic fractional coordinates, cell-coordinate coupling, and the atom-token decoding measure.

### One-head clean-endpoint parameterization

Keeping Crystallite's single clean-endpoint output is mathematically viable for Gaussian continuous channels. The denoiser can be converted to the required conditional score/control using the known Gaussian corruption coefficients (Tweedie/denoiser-to-score conversion), so a second learned score head is not intrinsically required.

However, correctness requires all of the following:

1. derive the conversion for Crystallite's exact EDM preconditioning and time orientation;
2. use minimum-image/tangent-space treatment for fractional coordinates rather than a naive Euclidean residual across the periodic boundary;
3. define the base/reference density and Jacobian for the chosen `ltri` cell variables;
4. decide separately how atom types are handled, because continuous subatomic-token EDM and masked categorical diffusion induce different path measures;
5. compute energy gradients in the actual learned variables (fractional coordinates and cell latent), including force/virial chain rules.

### Minimal credible implementation path

1. Freeze the pretrained Crystallite interface and retain its `x_hat_clean` heads.
2. First implement continuous-only energy post-training with atom count/types fixed; derive and unit-test denoiser-to-score/control conversion for fractional coordinates and `ltri` cell.
3. Add rollout log-weight accounting and verify it on an analytically tractable Gaussian energy before using an MLIP.
4. Add PCE outer stages using the existing PDNS logic conceptually (buffer, tempering, resampling), not by importing its flat-state classes.
5. Only then add an alloy-set-restricted atom channel. Choose either masked categorical diffusion or continuous token diffusion explicitly; do not mix their likelihood/weight formulas.

## 5. Bottom line

- The algorithm notes' Proximal CE and the PDNS paper/code describe the same central PCE mechanism.
- PDNS provides reusable algorithmic logic, not a drop-in Crystallite trainer.
- Crystallite's clean-endpoint head can be retained for continuous PCE post-training after an exact denoiser-to-control conversion.
- Fractional coordinates are compatible and preferable here; Cartesian conversion is not required.
- The hardest missing piece is not the network head. It is a measure-consistent path/RN-weight construction across periodic coordinates, cell variables, and any redesigned discrete species process.
