# Crystallite → Reward-Tilted Energy Sampler Post-Training
## Confirmed Design for BMS and Proximal CE

This document records the **current confirmed design decisions** for post-training a pretrained **Crystallite CSP checkpoint** into an energy-based sampler.

The intended setup is now explicitly:

\[
\boxed{
p^\star(F,y\mid c)
\propto
p_{\rm base}(F,y\mid c)\,
\exp[-\beta U(F,H(y))]
}
\]

where:

- \(c\): fixed composition / atom-type conditioning;
- \(F\): fractional coordinates;
- \(y\in\mathbb R^6\): Crystallite's six-dimensional cell parameterization;
- \(H(y)\): physical cell matrix decoded from \(y\);
- \(p_{\rm base}\): the pretrained Crystallite model distribution;
- \(U(F,H(y))\): physical crystal energy from the chosen energy oracle.

Both **BMS** and **Proximal Cross-Entropy (PCE)** should be understood as post-training methods targeting this same **base-aware reward tilt**.

The two methods differ in how they realize the tilted target, not in the definition of the target distribution.

---

# 0. Core project assumptions

We want to preserve the pretrained Crystallite representation and process as much as possible.

Therefore:

- atom types / composition remain fixed conditioning;
- Crystallite's fractional-coordinate representation is retained;
- Crystallite's six-variable lower-triangular cell representation is retained;
- the pretrained trunk/checkpoint should be reused rather than replaced;
- post-training should remain close to the pretrained model regime;
- exact torus machinery should not be introduced unless the objective truly requires it.

The implementation agent has direct access to the Crystallite codebase and must inspect the exact code before changing any mathematical component whose precise form is not stated here.

Do not guess the exact Crystallite noising formula, endpoint semantics, or cell map from this prompt if the code says otherwise.

---

# 1. Target distribution: both BMS and PCE use the SAME base-aware reward tilt

The target is:

\[
\boxed{
p^\star(F,y\mid c)
\propto
p_{\rm base}(F,y\mid c)\,
e^{-\beta U(F,H(y))}
}
\]

This is the central design decision.

Equivalently, using reward

\[
r(F,y)=-\beta U(F,H(y)),
\]

the target is

\[
p^\star \propto p_{\rm base} e^r.
\]

This means:

- the pretrained Crystallite model remains the structural prior;
- post-training reweights the pretrained distribution toward low-energy crystals;
- we are **not** replacing the pretrained prior by a bare physical Boltzmann measure.

This distinction matters for both fractional-coordinate treatment and cell Jacobians.

---

# 2. IMPORTANT: no extra target-measure Jacobian is required for this reward tilt

Do **not** add an extra cell/coordinate Jacobian solely because the physical structure is represented through fractional coordinates and a six-dimensional cell latent.

The target is defined directly with respect to the same latent measure used by the pretrained model:

\[
(F,y)\sim p_{\rm base}(F,y\mid c).
\]

Then the reward tilt is simply:

\[
p^\star(F,y\mid c)
\propto
p_{\rm base}(F,y\mid c)
e^{-\beta U(F,H(y))}.
\]

If \(p_{\rm base}(F,y)\) itself arose from some physical-space density under a change of variables, the corresponding representation Jacobian is already part of \(p_{\rm base}\).

Adding another factor such as

\[
|J_{(F,y)\to \text{physical}}|
\]

would generally **change the target distribution** and double-count the representation measure.

Therefore:

\[
\boxed{
\text{No additional target-measure Jacobian should be inserted in BMS or PCE under the current base-aware tilt definition.}
}
\]

A Jacobian would become relevant only if we later redefine the target as a **bare physical thermodynamic density** independent of \(p_{\rm base}\).

That is **not** the current project.

---

# 3. Cell parameterization still matters: through the ENERGY GRADIENT, not through an extra reward Jacobian

Although no extra Jacobian is added to the reward density, the six-dimensional lower-triangular cell parameterization must still be respected whenever an energy gradient is needed.

The energy is evaluated as:

\[
U(F,H(y)).
\]

The scalar reward is simply:

\[
r(F,y)=-\beta U(F,H(y)).
\]

No Jacobian correction is added to this scalar energy reward.

However, if BMS or another gradient-based objective needs a score / reward gradient with respect to \(y\), then the actual cell parameterization matters through the chain rule:

\[
\boxed{
\nabla_y U
=
\left(\frac{\partial H}{\partial y}\right)^\top
\nabla_H U
}
\]

schematically.

The implementation agent must inspect the exact Crystallite mapping

\[
y \mapsto H(y)
\]

and derive the exact chain rule for that implementation.

Do not assume:

- the variable ordering;
- whether diagonal entries are log-parameterized;
- whether any normalization is applied;
- whether the network predicts raw or transformed cell variables.

Verify the final \(\nabla_y U\) numerically with finite differences on non-orthogonal cells.

---

# 4. Fractional coordinates: do NOT rewrite the whole pretrained process as a torus diffusion

Fractional coordinates physically live on a periodic unit cell, but the current design is pragmatic:

> Preserve the pretrained Crystallite coordinate process wherever possible, and modify only the BMS endpoint displacement where periodic geometry directly matters.

Therefore the mainline implementation does **not** require:

- wrapped-normal transition densities;
- winding-number latent variables;
- exact torus Brownian bridges;
- torus likelihood sums;
- a new wrapped prior;
- a new torus-native noising process.

These may remain as optional exactness utilities / future ablations.

---

# 5. BMS: only the endpoint interpolation / velocity geometry needs a periodic minimum-image correction initially

BMS introduces its own interpolation between a prior-side sample and a terminal sample.

For the fractional coordinate channel, suppose those endpoints are

\[
F_0,\quad F_1.
\]

The raw Euclidean displacement

\[
F_1-F_0
\]

can be wrong across periodic boundaries.

Example:

\[
F_0=0.99,\qquad F_1=0.01.
\]

Then:

\[
F_1-F_0=-0.98,
\]

while the physically short displacement is:

\[
+0.02.
\]

Therefore define a componentwise minimum-image displacement:

\[
\boxed{
\Delta F
=
\operatorname{MI}(F_1-F_0)
}
\]

with the exact boundary convention chosen consistently in code.

A typical centered representative is equivalent to:

\[
\operatorname{MI}(\delta)
=
\delta-\operatorname{round}(\delta).
\]

---

# 6. BMS: interpolant and velocity target must be changed TOGETHER

Do not change only the velocity label.

The BMS interpolation path and the velocity target must remain mathematically consistent.

If the deterministic part of the interpolation uses schematically

\[
F_t
=
F_0+\alpha(t)\Delta F,
\]

then the corresponding velocity target must use

\[
\widehat b_t^F
=
\dot\alpha(t)\Delta F.
\]

The exact BMS interpolation in the implementation may use different schedules or extra terms.

Therefore the implementation agent should:

1. inspect the actual BMS code;
2. identify every occurrence of the endpoint displacement \(F_1-F_0\);
3. replace that displacement by \(\operatorname{MI}(F_1-F_0)\);
4. ensure the same displacement is used consistently in:
   - interpolant construction;
   - velocity target;
   - any other derivative derived from that interpolant.

Do not invent a new BMS interpolation formula from this prompt.

---

# 7. BMS: score / reward-gradient side for fractional coordinates

The fractional-coordinate energy score is separate from the endpoint displacement issue.

If the code uses row-vector convention

\[
r=fH,
\]

and Cartesian force is

\[
F_{\rm cart}=-\frac{\partial U}{\partial r},
\]

then:

\[
\nabla_f U
=
- F_{\rm cart} H^\top,
\]

and for reward

\[
r=-\beta U,
\]

\[
\boxed{
\nabla_f r
=
\beta F_{\rm cart}H^\top.
}
\]

The implementation agent must verify the actual row/column convention in Crystallite and adapt the matrix orientation accordingly.

Validate with finite differences.

Do not treat this as a torus-specific correction; this is simply the correct chain rule from Cartesian forces to fractional-coordinate gradients.

---

# 8. BMS: no extra Jacobian term in the reward gradient under the current target

Because the current target is

\[
p^\star
\propto
p_{\rm base}e^{-\beta U},
\]

the **reward gradient** contributed by the energy tilt is simply:

\[
\nabla r
=
-\beta \nabla U.
\]

For the cell channel:

\[
\nabla_y r
=
-\beta \nabla_y U(F,H(y)).
\]

For the fractional coordinate channel:

\[
\nabla_F r
=
-\beta \nabla_F U(F,H(y)).
\]

Do not append terms such as:

\[
\nabla \log |J|
\]

unless the project later changes the target definition away from the base-aware reward tilt.

---

# 9. PCE: preserve the pretrained Crystallite base process exactly

For PCE, the base path law is the pretrained Crystallite process.

Therefore:

\[
\boxed{
P^{\rm base}
=
\text{the actual pretrained Crystallite path/process}
}
\]

and the target endpoint distribution is:

\[
\boxed{
p^\star_1(x\mid c)
\propto
p^{\rm base}_1(x\mid c)e^{-\beta U(x)}.
}
\]

For fractional coordinates, preserve the pretrained Crystallite:

- coordinate representation;
- corruption/noising process;
- denoising semantics;
- endpoint head semantics;
- sampling transition;
- path-density convention.

Do not replace these by wrapped-normal transitions solely because the physical variables are fractional.

The agent must inspect the actual Crystallite code and document the exact formulas and code locations.

---

# 10. PCE: no fractional minimum-image correction should be inserted unless Crystallite itself uses it

Unlike BMS, PCE does not introduce a new endpoint interpolation velocity of the form

\[
F_1-F_0.
\]

Therefore there is no generic reason to insert a minimum-image replacement into PCE.

If the pretrained Crystallite process internally uses a particular residual/noising geometry, keep it unchanged.

Do not modify the pretrained denoising residual to:

\[
\operatorname{MI}(F_t-F_1)
\]

unless the actual Crystallite process already does so or we explicitly redefine the base process.

---

# 11. PCE: path-law ratios must use the ACTUAL Crystallite transition law

If the proximal weight or objective requires a path-law ratio such as

\[
\frac{dP^{(k)}}{dP^{\rm base}},
\]

then this must be computed using the **actual implemented Crystallite transitions**.

The agent should inspect and document:

- exact forward transition;
- exact reverse / denoising semantics if needed;
- exact covariance/noise schedule;
- exact state representation;
- any normalization or scaling;
- any existing coordinate wrapping;
- any likelihood/density terms already implemented.

Do not substitute a guessed Gaussian formula from this prompt if the code differs.

---

# 12. PCE: reward evaluation in the six-dimensional cell representation

For PCE, the energy reward is:

\[
\boxed{
r(F,y)
=
-\beta U(F,H(y)).
}
\]

There is no need to add a separate cell Jacobian to the energy reward.

The six-dimensional parameterization is accounted for simply by decoding:

\[
y \mapsto H(y)
\]

before evaluating the physical energy.

If PCE itself does not require \(\nabla_y U\), then no cell-gradient chain rule is needed for the energy weight.

If some implementation variant uses reward gradients, then apply the same chain rule described above.

---

# 13. Final BMS vs PCE distinction

The current confirmed practical design is:

## BMS

- pretrained Crystallite distribution remains the structural base;
- target is reward-tilted:
  \[
  p^\star\propto p_{\rm base}e^{-\beta U};
  \]
- BMS introduces its own interpolation;
- fractional endpoint displacement in that interpolation must use minimum-image geometry;
- interpolant and velocity target must be changed consistently;
- energy-score gradients use the correct fractional/cell chain rule;
- no extra target-measure Jacobian.

## PCE

- pretrained Crystallite process is the explicit \(P^{\rm base}\);
- target is the same reward tilt:
  \[
  p^\star\propto p_{\rm base}e^{-\beta U};
  \]
- preserve the pretrained Crystallite path/noising process;
- do not add torus transitions;
- do not add minimum-image residuals unless already present in Crystallite;
- use the actual Crystallite path law for proximal/path ratios;
- no extra target-measure Jacobian.

---

# 14. Wrapping / `mod 1` is not the main mathematical change

Do not add `mod 1` everywhere by default.

Wrapping may be useful for:

- canonicalizing final outputs;
- matching existing Crystallite conventions;
- satisfying an energy-oracle interface.

But it is **not** the main reason for the BMS fractional-coordinate modification.

The central BMS issue is:

\[
\boxed{
\text{which periodic displacement connects }F_0\text{ and }F_1
}
\]

not whether every intermediate state is explicitly reduced modulo one.

The implementation agent should inspect whether Crystallite and the energy oracle already handle equivalent out-of-cell fractional coordinates under PBC.

---

# 15. Exact-torus utilities are optional

Existing wrapped-normal / winding-aware utilities may remain in the repository for:

- exactness checks;
- boundary stress tests;
- future torus-native ablation;
- future method extension.

But they should not block the mainline prototype.

The project objective is to post-train a pretrained Crystallite model, not to rewrite Crystallite into a new torus diffusion before testing whether such exactness is necessary.

---

# 16. Required code audit before implementation

Because this prompt deliberately avoids guessing Crystallite internals, the implementation agent must inspect the repository and report the actual implementation.

## Fractional coordinate channel

Document:

1. state representation;
2. exact training corruption/noising process;
3. exact denoising target;
4. exact endpoint-head semantics;
5. exact sampling transition;
6. any wrapping;
7. any minimum-image operation;
8. any coordinate normalization/scaling.

## Cell channel

Document:

1. exact six-variable representation;
2. exact map \(y\to H(y)\);
3. inverse map if used;
4. positivity/volume handling;
5. exact training noising process;
6. endpoint-head semantics;
7. any normalization/scaling.

## Geometry

Document:

1. row/column matrix convention;
2. PBC handling in the model;
3. pair-distance / neighbor construction;
4. PBC handling in the energy oracle.

Return code locations for all items.

---

# 17. Required implementation plan

## BMS

1. Load pretrained Crystallite checkpoint.
2. Preserve composition / fixed atom types.
3. Preserve the existing cell representation.
4. Preserve the trunk/checkpoint as much as possible.
5. Define the base-aware reward target:
   \[
   p^\star\propto p_{\rm base}e^{-\beta U}.
   \]
6. Inspect the BMS interpolation code.
7. Replace raw fractional endpoint displacement by minimum-image displacement wherever the interpolation depends on \(F_1-F_0\).
8. Use the same minimum-image displacement in the corresponding velocity target.
9. Keep the rest of the coordinate process unchanged initially.
10. Implement the correct energy-gradient conversion for:
    - fractional coordinates;
    - six-dimensional cell parameters,
    if required by the BMS score/reward-gradient objective.
11. Do not add an extra representation Jacobian.

## PCE

1. Load pretrained Crystallite checkpoint.
2. Treat the actual pretrained Crystallite process as \(P^{\rm base}\).
3. Define the same endpoint reward tilt:
   \[
   p^\star_1\propto p^{\rm base}_1e^{-\beta U}.
   \]
4. Preserve Crystallite's actual noising/denoising/path process.
5. Compute any required path ratio using the actual Crystallite transitions.
6. Do not replace the path by a torus kernel.
7. Decode \(y\to H(y)\) for energy evaluation.
8. Do not add an extra representation Jacobian.

---

# 18. Required validation tests

## Shared

- verify energy invariance under periodic-equivalent coordinate representations;
- verify cell decode produces valid physical cells;
- verify energy evaluation from \(y\) matches direct evaluation from decoded \(H(y)\).

## BMS fractional interpolation

### Boundary-crossing test

Example:

\[
F_0=0.99,\qquad F_1=0.01.
\]

Verify:

- minimum-image displacement is \(+0.02\) under the chosen convention;
- interpolation follows the short periodic path;
- numerical derivative of the interpolant matches the velocity target.

### Non-boundary test

Verify minimum-image and Euclidean displacement agree when no boundary is crossed.

### Fractional score

Finite-difference-check the conversion from Cartesian force to \(\nabla_F U\).

### Cell score

Finite-difference-check \(\nabla_y U\) through the exact Crystallite cell map.

## PCE

- verify pretrained path/noising code is unchanged;
- verify proximal path-ratio code uses the actual Crystallite transitions;
- verify no silent wrapped-normal substitution was introduced;
- verify energy tilt only changes the reward/weighting objective, not the base process.

---

# 19. Final confirmed design summary

\[
\boxed{
p^\star(F,y\mid c)
\propto
p_{\rm base}(F,y\mid c)
e^{-\beta U(F,H(y))}
}
\]

for **both BMS and PCE**.

Therefore:

\[
\boxed{
\text{No extra target-measure Jacobian for either method.}
}
\]

The six-dimensional cell representation matters through:

\[
\boxed{
y\to H(y)\to U
}
\]

for energy evaluation, and through:

\[
\boxed{
\nabla_y U
=
\left(\frac{\partial H}{\partial y}\right)^\top\nabla_H U
}
\]

when a gradient is required.

For fractional coordinates:

\[
\boxed{
\text{BMS: minimum-image endpoint displacement in interpolant + velocity target}
}
\]

\[
\boxed{
\text{PCE: preserve the pretrained Crystallite path/noising process}
}
\]

Exact torus machinery is optional and should not block the first prototype.

If the repository audit reveals a concrete contradiction with any assumption above, report it before changing the design.
