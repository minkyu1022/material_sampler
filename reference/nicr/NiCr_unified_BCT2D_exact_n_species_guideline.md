# Ni–Cr Unified BCT-2D — Exact-n Species Unmasking Upgrade

## 0. Context

This is part of the already-active final goal:

> Build and make the N=128 unified Ni–Cr BCT-2D sampler succeed end-to-end.

Do not stop after implementing this discrete helper. Continue automatically through integration, training, rollout, ESS/path-weight diagnostics, BCC/BCT/FCC coverage, and comparison to the BCT-2D reference MC.

The current unified code already reports an “exact-n constrained rollout”, but the unpublished JANUS constrained reveal algorithm is still unknown. Therefore:

- do not claim the current implementation is paper-faithful;
- audit what is currently implemented;
- for the new unified model, replace/upgrade it with the explicit fixed-cardinality conditional distribution below;
- do not wait for Denis for this unified track.

The paper-faithful JANUS reproduction remains separate.

---

## 1. State and notation

Use N=128.

At fixed composition let the target number of Cr atoms be n.

At an intermediate masked state:

- R: already revealed sites
- M: still-masked sites
- k: revealed Cr count
- m = |M|
- r = n-k: remaining Cr quota

Always enforce

0 <= r <= m.

Invalid states must raise an error; do not silently clamp them.

---

## 2. Network output

For every masked site i in M, let the network output a Cr-vs-Ni logit l_i.

Do not independently Bernoulli-sample masked sites once exact composition is imposed.

Instead use the logits to parameterize a distribution over subsets of masked sites.

---

## 3. Exact fixed-cardinality distribution

Let S subset M be the subset of still-masked sites that will ultimately be Cr.

Require

|S| = r.

Define

p_theta(S | x_t, n)
  = exp(sum_{i in S} l_i) / Z_r(l)
    subject to |S|=r,

where

Z_r(l)
  = sum_{S' subset M, |S'|=r}
      exp(sum_{i in S'} l_i).

This distribution has support only on assignments with exactly r remaining Cr atoms, so terminal N_Cr=n is guaranteed by construction.

There must be no final ad-hoc count repair.

---

## 4. Boundary cases

Handle analytically:

- r=0: every remaining masked site is Ni
- r=m: every remaining masked site is Cr
- r<0 or r>m: invalid state, fail loudly

---

## 5. Dynamic programming for Z_r

Do not enumerate 2^m subsets.

Let w_i = exp(l_i).

Define D[j,q] as the total weight of choosing exactly q Cr sites among the first j masked sites.

Initialize:

D[0,0] = 1
D[0,q>0] = 0

Recurrence:

D[j,q] = D[j-1,q] + w_j D[j-1,q-1].

Then

Z_r = D[m,r].

Complexity is O(mr), which is trivial for N=128.

Implement in log-space with logaddexp for numerical stability, preferably float64.

---

## 6. Exact constrained marginals

The marginal probability that masked site i is Cr is

P_i(Cr)
  = exp(l_i) * Z_{r-1}^{(-i)} / Z_r.

Then

P_i(Ni) = 1 - P_i(Cr).

These constrained marginals, not independent sigmoid(l_i), must feed the masked discrete-flow reveal machinery.

Mandatory invariant:

sum_{i in M} P_i(Cr) = r

up to numerical precision.

Do not recompute a full DP separately for every site. Use prefix/suffix or forward/backward DP so the marginals remain exact and efficient.

If gradients through the DP are used, verify them against finite differences on tiny systems.

---

## 7. Preferred training loss

For the unified model, use the same fixed-cardinality distribution at training and sampling time.

Let S* be the set of masked sites whose ground-truth terminal species is Cr.

Then |S*|=r.

Use exact constrained negative log likelihood:

L_species
  = - sum_{i in S*} l_i
    + log Z_r(l).

Keep the old independent per-site CE only as an ablation.

The default unified model should use this constrained NLL unless a concrete numerical failure is demonstrated.

---

## 8. Integration with existing masked DFM

Do not redesign the entire discrete flow.

Use:

masked state
-> network logits l_i
-> remaining quota r=n-k
-> fixed-cardinality DP
-> constrained marginals P_i(Cr), P_i(Ni)
-> existing masked-DFM / reveal-rate machinery

After every realized reveal, update k and r and recompute the constrained probabilities for the remaining masked set.

Do not reuse stale marginals.

---

## 9. Important tau-leap / multi-site reveal warning

If the current code reveals multiple sites simultaneously, independent sampling from one-site constrained marginals does NOT preserve exact cardinality for the batch.

Audit the current “exact-n constrained rollout” specifically for this.

Allowed approaches:

1. Preferred: sample the reveal batch jointly from the induced fixed-cardinality subset distribution.
2. Safe fallback: reveal sequentially within the nominal tau-leap batch, updating quota and marginals after each realized reveal.

Not allowed:

- independently sample each selected site from P_i(Cr)
- then repair the count afterward.

---

## 10. Exact subset sampling reference implementation

For correctness tests, implement exact sequential conditional sampling from the DP/suffix partition functions.

At each masked site, compute the exact probability that this site is Cr given the remaining quota and remaining logits, sample, update the quota, and continue.

This must sample exactly from the fixed-cardinality distribution above.

Use this as the reference implementation for tiny-system tests and any full remaining-assignment operation.

---

## 11. Path-weight bookkeeping must change too

The current unified model already computes forward/backward path probabilities and log weights.

After this upgrade, audit every discrete probability used in:

- forward path probability
- backward/reverse path probability
- log weight
- importance ratio
- terminal proposal probability

The probability must correspond to the actual constrained transition kernel.

Do not enforce exact terminal count in rollout while still evaluating path probabilities using the old unconstrained factorized categorical model.

That would make ESS and log-weight diagnostics invalid.

Keep log-probability accumulation in float64.

---

## 12. Composition conditioning

Continue conditioning the network on n/N.

But n/N is only a neural condition; it does not by itself enforce exact count.

Use both:

- n/N: conditioning signal
- fixed-cardinality distribution: hard support constraint

---

## 13. Relation to Denis / paper-faithful JANUS

Be explicit in code comments and reports.

Paper-faithful track:
- exact original JANUS Ni–Cr constrained reveal remains unknown until author confirmation.

Unified BCT-2D track:
- use the exact fixed-cardinality method in this document.

Do not call this the published JANUS reveal algorithm.

Suggested implementation names:

- fixed_cardinality_masked_denoiser
- conditional_bernoulli_exact_n
- exact_n_subset_denoiser

---

## 14. Mandatory tests

Before larger training, add these tests.

### Tiny exhaustive test

For m <= 8, enumerate all subsets with |S|=r and compare the DP implementation against exact enumeration for:

- Z_r
- site marginals
- constrained NLL
- sampled subset frequencies
- gradients

### Cardinality tests

Test many random logits and target counts, including:

- n_Cr=0
- 1
- 32
- 64
- 96
- 127
- 128

Every terminal rollout must satisfy the count exactly.

### Marginal identity

Verify

sum_i P_i(Cr) = r.

### Equal-logit symmetry

If all logits are equal, verify

P_i(Cr) = r/m

for every masked site.

### Path-probability test

For small systems, compare sequential rollout log probability with the exact fixed-cardinality subset probability.

### Regression

Ensure the paper-faithful Cu–Ni / JANUS code path is unchanged unless intentionally sharing a tested helper.

---

## 15. Re-run the current unified smoke test after integration

Repeat the current end-to-end configuration:

- N=128
- multiple fixed compositions
- BCC and FCC-equivalent starts
- M=100
- linear interpolant
- gamma=0
- g^2 = 0.02^2 * (T/750)
- normalized 2D cell
- corrected continuous backward sign
- float64 log weights

Confirm:

1. terminal Cr count exact
2. forward/backward discrete log probabilities finite
3. no invalid quota states
4. rollout finite
5. RMS u remains local
6. cell remains valid
7. path-weight decomposition uses the new constrained kernel

---

## 16. Current low ESS is still only a smoke result

The current result:

- ESS = 1/3
- std(logW) ~ 676.3
- 100 updates
- 8 reference samples
- tiny model

should still be treated as undertraining, not as a final failure.

But after changing the discrete kernel, rerun the smoke test before scaling up so changes in path-weight variance are attributable.

---

## 17. Continue the full goal after this phase

Do not stop when the exact-n helper passes.

Continue automatically with:

1. multi-temperature / multi-composition reference buffers
2. both BCC-like and FCC-like structural coverage
3. intended larger/common-trunk unified model training
4. tracking:
   - c/a distribution
   - endpoint coverage
   - RMS u
   - species statistics
   - std(logW)
   - ESS
5. diagnosis of BCC/FCC neural mode collapse
6. if needed, test:
   - broader midpoint cell prior
   - symmetric BCC/FCC mixture prior
   - replay-buffer balancing
   - curriculum or seeding
7. compare against the new BCT-2D reference MC
8. only declare success after both structural basins and target statistics are captured acceptably

The success criterion is the complete Ni–Cr unified sampler, not this exact-n implementation.

---

## 18. Next report

Only report concrete evidence:

- changed files / git commit
- what the old exact-n implementation actually was
- whether it was replaced
- test counts
- exhaustive tiny-system max error
- terminal-count checks
- path-probability consistency result
- new smoke metrics
- launched training command/PID if a longer run has started
