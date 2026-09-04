# Crystallite reward-tilt implementation audit

Source design: `incoming/crystallite_reward_tilt_bms_pce_confirmed_design_v2.md`.

## Adopted target

Both post-training methods target the same latent-measure reward tilt

\[
p^\star(F,y\mid c)\propto p_{\rm base}(F,y\mid c)
\exp[-\beta U(F,H(y))].
\]

No additional coordinate/cell target-measure Jacobian is added. The cell map still
enters an energy gradient through the ordinary chain rule when a method needs one.

## Public MP20 checkpoint facts

Checkpoint: `pre_train/checkpoints/csp_mp20_best.pt`.

- training step: 1,400,000;
- model: 270,047,720 parameters, `d_model=1024`, 14 layers, 16 heads;
- CSP atom types are fixed conditioning;
- `lattice_repr=ltri`;
- training method is EDM, not VFM;
- fractional-coordinate data scale is unnormalized; `sigma_data_coord=0.3`;
- cell latent uses `sigma_data_lattice=0.3`;
- sampling uses 400 Karras/Heun steps, `sigma_min=0.002`, `sigma_max=80`,
  `rho=7`, `S_churn=30`, `S_noise=1.003`;
- checkpoint contains EMA weights with decay 0.99999.

## Actual fractional-coordinate base process

1. MP20 preprocessing performs Niggli reduction and retains the conventional cell
   (`reference/crystalite/src/data/mp20_tokens.py:228-263`). Coordinates are stored
   as `F1 = frac_coords % 1`.
2. Training centers the clean state as `F1 - 0.5` and applies additive Euclidean
   Gaussian corruption `x_sigma = x + sigma * epsilon`
   (`reference/crystalite/src/train_crystalite.py:594-599`).
3. The model input alone is shifted and wrapped with `mod1(frac_noisy + 0.5)`
   (`reference/crystalite/src/crystalite/edm_utils.py:106-123`).
4. EDM preconditioning produces a clean endpoint estimate
   `D_frac = c_skip * frac_noisy + c_out * raw_head`
   (`reference/crystalite/src/crystalite/edm_utils.py:90-132`).
5. The coordinate training residual uses a minimum-image difference
   (`reference/crystalite/src/crystalite/edm_utils.py:181-188`).
6. Sampling is Karras EDM with Heun correction. Its coordinate drift already uses
   a minimum-image endpoint residual and re-centers after each step
   (`reference/crystalite/src/crystalite/sampler.py:54-276`).
7. The model's geometry features use periodic minimum-image pair geometry
   (`reference/crystalite/src/models/transformer.py:239-389`).

Therefore PCE must use this exact centered additive-Gaussian EDM/Heun process as
`P_base`; it must not substitute a wrapped-normal or winding transition law.

## Actual cell representation

The checkpoint uses six unconstrained lower-triangular parameters

\[
y=(p_0,p_1,p_2,p_3,p_4,p_5),\qquad
H(y)=\begin{bmatrix}
e^{p_0}&0&0\\
p_1&e^{p_2}&0\\
p_3&p_4&e^{p_5}
\end{bmatrix}.
\]

The exact implementation is
`reference/crystalite/src/models/lattice_repr.py:79-96`. MP20's original
`[log a, log b, log c, cos(alpha), cos(beta), cos(gamma)]` representation is
converted through its Gram matrix and Cholesky factor at lines 133-166. This map,
not a guessed six-vector ordering, must be used to decode energy-oracle cells and
to differentiate `U(F,H(y))`.

## Mainline BMS contract

- Keep the pretrained trunk, fixed composition, endpoint head, and `ltri` cell map.
- Define the terminal target by the base-aware energy tilt above.
- In the BMS-only interpolation, use the same minimum-image endpoint displacement
  in both the interpolated state and its velocity target.
- Convert Cartesian oracle force to the fractional reward gradient using the
  repository's row-vector cell convention, then verify by finite differences.
- Differentiate the exact `ltri_params_to_lattice_matrix` map for the cell reward
  gradient and verify on non-orthogonal cells.
- Do not add a representation Jacobian.

## Mainline PCE contract

- Treat the checkpoint's actual EDM process as `P_base`.
- Preserve its corruption, preconditioning, endpoint semantics, Karras schedule,
  minimum-image residual, and Heun sampler.
- Apply the energy reward through PCE weighting/control only.
- Any path-law ratio must be derived from these actual implemented transitions.
- Do not replace the base process with `post_train/common/torus.py`.

## Status and correction to the earlier note

`FRACTIONAL_POSTTRAIN_MATH.md` previously described exact wrapped-normal machinery
as mandatory. That is incompatible with the confirmed base-aware design. The
wrapped-normal and winding code remains an optional exactness/boundary-test
utility only; it is not a mainline dependency and must not block the prototype.

Already usable for the BMS mainline:

- `torus_delta`/minimum-image displacement;
- consistent minimum-image interpolation and endpoint velocity;
- Cartesian-force to fractional-gradient conversion;
- translation zero-mode projection.

Implemented and verified:

- exact analytic chain rule from a physical cell-matrix gradient to the six
  Crystalite `ltri` parameters in `post_train/common/geometry.py`;
- agreement with decoder autograd and central finite differences on a
  non-orthogonal cell (`post_train/common/test_geometry.py`).

Still required:

1. checkpoint-preserving BMS adapter and reward-gradient finite-difference tests;
2. PCE adapter tied to the existing EDM transition implementation;
3. regression tests proving the PCE base process is unchanged;
4. energy invariance tests for periodic-equivalent coordinates.
