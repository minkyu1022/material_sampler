# Unified Ni–Cr path measure

This note applies only to the new unified BCT-2D model, not to the unpublished
paper-faithful JANUS Ni–Cr reveal kernel.

## Exact-count species path

At each masked state, a Bernoulli reveal clock selects the sites revealed in
that step. Conditional on those sites, their species are drawn from the exact
fixed-cardinality subset distribution, marginalizing all sites left masked.
The lifted proposal path measure is

`q(path) = q_clock(reveal sites) q_species(revealed assignments | reveal sites)`.

The target is lifted with the same state-independent reveal clock and reverse
masking schedule. Therefore `q_clock` cancels from the importance ratio. The
stored discrete path term is exactly the conditional `q_species` factor. Its
transition identity is tested by comparing every multi-step dynamic-logit
increment with the corresponding ratio of fixed-cardinality partition
functions.

## Continuous coordinate measure

Displacements are fractional and constrained by `sum_i u_i = 0`. Pick any
orthonormal basis of this `(3N-3)`-dimensional subspace. Applying the cell
matrix to each of its `N-1` independent three-vectors gives Jacobian
`det(L)^(N-1) = V^(N-1)`. The two positive cell variables add the transform
Jacobian `a c`. Thus the restricted target density contains

`(N-1) log V + log a + log c`,

not `N log V`. Forward and backward Gaussian increments both live on the same
zero-COM subspace, so their common singular normalizer cancels in the path
ratio.
