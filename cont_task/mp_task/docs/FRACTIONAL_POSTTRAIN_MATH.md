# Fractional-coordinate post-training mathematics

## Decision

The MP20 checkpoint keeps Crystalite's fractional coordinates and six lower-triangular
cell variables. Fractional coordinates live on the flat unit torus, not on Euclidean
space. BMS and proximal CE therefore share one torus kernel implementation rather
than applying an Euclidean Tweedie formula followed by an unrelated `mod 1`.

## What remains unchanged

- The public 270M Crystalite MP20 CSP checkpoint and its endpoint head.
- Exact composition conditioning and fixed atom types.
- The outer BMS fixed-point structure and its separate velocity and generalized-score
  losses.
- The proximal target form `p_base(x | composition) exp(-beta U(x))` and the outer
  proximal/KL objective.

## What must use torus mathematics

For one fractional component, the wrapped-normal transition density is

\[
q_\sigma(y\mid x)=\sum_{k\in\mathbb Z}
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(y-x+k)^2}{2\sigma^2}\right].
\]

Its score is the normalized image-weighted sum

\[
\nabla_y\log q_\sigma(y\mid x)=
\frac{\sum_k -(y-x+k)\,q_k/\sigma^2}{\sum_k q_k}.
\]

The implementation uses `logsumexp` and chooses the number of integer images from a
Gaussian tail tolerance. The same convention must be used by:

1. prior/noising density and score;
2. endpoint-to-score conversion;
3. bridge/interpolant sampling, including winding variables when stochastic;
4. rollout SDE transitions;
5. BMS path densities/weights and proximal-CE path/control terms.

For the deterministic `gamma=0` interpolant, the coordinate path is the shortest
torus geodesic. A stochastic Brownian bridge additionally requires sampling or
marginalizing its winding number; wrapping an ordinary Euclidean bridge afterward is
not an exact substitute for its transition likelihood.

## Physical target score

With row-vector ASE convention `r = f H`, Cartesian force `F = -dU/dr`, and
`pi(f) proportional exp(-beta U(f))`, the fractional-coordinate score is

\[
\nabla_f\log\pi = \beta F H^T.
\]

This conversion, padding mask, and removal of the global translation zero mode must
be identical in training and rollout. The cell channel separately needs the chain
rule from stress/virial to the chosen six lower-triangular variables, positivity of
the diagonal representation, and the target-measure Jacobian. Those pieces are not
implemented by the torus module and must pass independent finite-difference tests.

## Current implementation status

Implemented in `post_train/common/torus.py`:

- shortest torus displacement and deterministic interpolation;
- analytic wrapped-normal log density;
- analytic wrapped-normal score;
- winding-aware stochastic Brownian-bridge sampling;
- periodicity, normalization, boundary, and autograd checks.

Implemented in `post_train/common/geometry.py`:

- Cartesian force to fractional-coordinate target-score conversion;
- padded center-of-mass/translation-zero-mode projection;
- an autograd chain-rule check using a non-orthogonal cell.

Still required before an end-to-end claim:

- rollout transition sampler and transition log-density integration;
- one-head BMS adapter with separate velocity/generalized-score losses;
- UMA force/stress target adapters and cell Jacobian;
- proximal-CE adapter using the same torus transition law;
- rollout, path-weight, and sustained-memory integration tests.
