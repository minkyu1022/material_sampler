# Ni–Cr Unified Sampler — Final BCT-2D Model Guideline

## 0. Scope

This document is for the agent that has been reproducing JANUS.

Do not modify the paper-faithful JANUS reproduction into a different model.

Keep the published Ni–Cr reproduction intact:

- FCC model remains FCC-only.
- BCC model remains BCC-only.
- Their published system sizes/configs remain separate.

The task below is a **new unified model** that should live beside the reproduction.

Scientific goal:

> Build one Ni–Cr sampler on a common `N=128` BCT state space that can traverse BCC ↔ BCT ↔ FCC-equivalent structures by generating a 2D tetragonal cell.

Suggested separate namespace:

`nicr_unified_bct2d_N128`

---

# 0.1 Author-response dependency — NOT A BLOCKER

Implementation of `nicr_unified_bct2d_N128` should begin **now**.

Do **not** wait for Denis's reply before implementing this unified model.

The outstanding author questions are useful for improving the **paper-faithful JANUS Ni–Cr reproduction**, especially for unpublished/underspecified details such as:

- exact Ni–Cr continuous diffusion settings used by the authors,
- exact constrained masked reveal that guarantees the requested terminal Cr count,
- confirmation of the Ni–Co–Cr potential/cutoff convention.

Those answers are **not required to define the new unified BCT-2D model**, because this model is a new benchmark extension with its own explicitly documented choices.

For the unified model, proceed with the choices in this document:

- common BCT `N=128` state space,
- 2D tetragonal cell `(a,c)`,
- Cu–Ni-validated diffusion schedule as the initial baseline,
- linear interpolant,
- `gamma = 0`,
- `M = 100`,
- the current working native-cutoff Ni–Cr EAM/FS oracle,
- exact fixed-composition terminal constraint implemented with the best currently validated modular method.

If Denis later provides more exact paper details:

1. use them to improve/check the **paper-faithful Ni–Cr reproduction**;
2. only change the unified model if there is an independent technical reason to do so;
3. do not silently rewrite the unified benchmark merely to imitate unpublished paper settings.

Keep the two tracks separate in code, configs, reports, and claims.


---

# 1. Production size

Use:

\[
\boxed{N=128}
\]

for the production unified benchmark.

Reason:

- JANUS Ni–Cr BCC already uses `N=128`.
- Cu–Ni reproduction at `N=108` provides the nearest validated model scale.
- Staying near that scale gives a cleaner comparison than reducing the problem to `N=54`.

`N=54` may be used only for debugging or fast unit tests.

---

# 2. Preserve all validated reproduction fixes

Do not regress the existing validated implementation.

Keep:

- corrected reverse/backward Gaussian mean sign
- float64 log-weight / path-weight accumulation
- Cu–Ni validated rollout logic
- `M=100` as the current default unless a new ablation changes it
- separate paper-faithful Ni–Cr FCC/BCC experiment configs
- all unit/sign checks already added

The unified model is an additional experiment, not a reinterpretation of the published JANUS model.

---

# 3. Unified state

Use three state components:

\[
(\text{species},\,u,\,z_{\rm cell}).
\]

Meaning:

- `species`: discrete Ni/Cr identities on fixed sites
- \(u\): fractional displacement from fixed common-BCT reference sites
- \(z_{\rm cell}\): normalized 2D cell latent

Avoid overloaded notation in code.

Use names like:

- `species`
- `disp_u`
- `cell_a`
- `cell_c`
- `cell_z`

---

# 4. Common fixed reference sites

Use primitive BCT basis:

\[
s_1=(0,0,0),\qquad
s_2=(1/2,1/2,1/2).
\]

Replicate `4×4×4`, giving:

\[
N=2\times4^3=128.
\]

For site \(i\),

\[
f_i=s_i+u_i
\]

is the actual fractional coordinate.

With row-vector convention,

\[
r_i=f_iL=(s_i+u_i)L.
\]

The site list \(s_i\) remains fixed.

---

# 5. Cell model: exactly 2 physical variables

Use

\[
\boxed{
L(a,c)=
\begin{pmatrix}
a&0&0\\
0&a&0\\
0&0&c
\end{pmatrix}
}
\]

with

\[
a>0,\qquad c>0.
\]

Do not generate:

- shear,
- cell angles,
- independent \(b\),
- the previous 6D lower-triangular cell.

Interpretation:

\[
c/a=1
\]

→ BCC.

\[
1<c/a<\sqrt2
\]

→ BCT Bain intermediates.

\[
c/a=\sqrt2
\]

→ FCC-equivalent.

Both \(a\) and \(c\) are generated.

Neither is fixed.

---

# 6. Primitive vs full-supercell convention

Decide once whether network-facing physical lengths mean:

### primitive BCT lengths

\[
(a,c)
\]

or full `4×4×4` supercell lengths

\[
(A,C)=(4a,4c).
\]

Do not mix the conventions.

Add a unit test that constructs both BCC and FCC-equivalent cells and verifies Cartesian coordinates under the chosen convention.

---

# 7. Cell normalization

Use two stages.

## Stage 1 — dimensionless log lengths

For primitive-cell convention:

\[
y_a=\log(a/l_{\rm ref}),
\qquad
y_c=\log(c/l_{\rm ref}).
\]

Use

\[
\boxed{
l_{\rm ref}=2.765\ \AA
}
\]

for primitive BCT lengths.

If storing the full supercell instead, use

\[
\boxed{
l_{\rm ref}=11.06\ \AA.
}
\]

Because the final 2D model has only positive lengths, both dimensions can be treated in log space.

This avoids the original mixed `log-diagonal vs linear-shear` scale issue of the 6D Packora-like cell parameterization.

## Stage 2 — component-wise standardization

Define:

\[
z_a=\frac{y_a-\mu_a}{\sigma_a},
\qquad
z_c=\frac{y_c-\mu_c}{\sigma_c}.
\]

The model should operate on

\[
\boxed{
z_{\rm cell}=(z_a,z_c).
}
\]

Interpretation:

- \(l_{\rm ref}\): removes units
- \((\mu,\sigma)\): makes the learned dimensions comparable in numerical scale

Do not infer \((\mu,\sigma)\) from forbidden held-out production reference data if the benchmark is intended to remain self-bootstrapped.

Use a documented allowed calibration source.

---

# 8. Suggested prior center

A simple initial Bain midpoint is:

\[
r_0=\frac{c_0}{a_0}
=
\frac{1+\sqrt2}{2}
\approx1.2071.
\]

Using representative primitive BCT volume

\[
V_0\approx21.14\ \AA^3,
\]

gives approximately

\[
a_0\approx2.60\ \AA,
\qquad
c_0\approx3.14\ \AA.
\]

These define a prior/init center only.

They are not fixed lattice constants.

Possible cell priors:

1. one broad Gaussian around the midpoint in normalized \(z_{\rm cell}\)
2. a symmetric BCC-like/FCC-like two-component Gaussian mixture

Start simple.

Do not encode target free-energy preference in the mixture weights without justification.

---

# 9. Atomic displacement channel

Keep \(u_i\) as a fractional displacement:

\[
u_i=f_i-s_i.
\]

The intended decomposition is:

\[
\boxed{
\text{cell }(a,c)
\rightarrow
\text{global FCC/BCC/BCT deformation}
}
\]

and

\[
\boxed{
u
\rightarrow
\text{local thermal/local structural relaxation}
}
\]

A small zero-centered Gaussian displacement prior remains the natural baseline.

Do not intentionally make \(u\) large enough to carry the FCC↔BCC transition by atomic relabeling/reconstruction.

Track RMS displacement as a diagnostic.

---

# 10. Continuous learning space

The cell flow/diffusion should run in the normalized coordinates:

\[
(z_a,z_c).
\]

Conceptually:

\[
(a,c)
\longleftrightarrow
(y_a,y_c)
\longleftrightarrow
(z_a,z_c).
\]

Train velocity and/or score consistently in this normalized cell space.

If converting analytic quantities between spaces, use the chain rule exactly.

---

# 11. Default continuous diffusion schedule \(g(t)\)

For the first unified Ni–Cr implementation, reuse the **author-confirmed Cu–Ni continuous diffusion scale as the default baseline**.

Cu–Ni used:

\[
g_u^2(T)
=
0.02^2\frac{T}{750}.
\]

The same numerical baseline may be used initially for the normalized 2D cell channel:

\[
\boxed{
g_{\rm cell}^2(T)
=
0.02^2\frac{T}{750}.
}
\]

Equivalently,

\[
g(T)=0.02\sqrt{\frac{T}{750}}.
\]

Use the same initial baseline for both:

- atomic displacement channel \(u\)
- normalized cell channel \(z_{\rm cell}\)

unless the implementation requires separate channel notation.

This is a **starting/default hyperparameter**, not an author-confirmed Ni–Cr unified-model value.

The reason it is reasonable to reuse:

- Cu–Ni reproduction already validated this scale.
- the unified cell coordinates are standardized, so a shared numerical diffusion scale is much more meaningful than it would be in raw Å units.

However, because the new cell coordinate differs from Cu–Ni's original 1D log-volume channel, run a short calibration before declaring it final.

At minimum check:

1. atomic displacement magnitude remains physical/local;
2. `z_a` and `z_c` do not diffuse too weakly or explosively;
3. BCC-like and FCC-like cell regions remain reachable;
4. path-weight variance / ESS is not catastrophic;
5. numerical rollout remains stable.

Do not change \(g\) preemptively without evidence.

Use the Cu–Ni value first, then ablate only if needed.

---

# 12. Interpolant / rollout defaults

As the first baseline, inherit the validated Cu–Ni settings unless a new experiment demonstrates a problem:

- deterministic training interpolant: linear
- `gamma = 0`
- rollout steps: `M = 100`
- continuous diffusion baseline:
  \[
  g(T)=0.02\sqrt{T/750}
  \]

These are implementation defaults for the new model, not claims that the paper reported these exact unified Ni–Cr settings.

---

# 13. CRITICAL: Jacobian / transformed target density

Do not treat the coordinate Jacobian as a simple multiplicative loss weight.

The model state is normalized \(z_{\rm cell}\), while the physical target is defined in physical cell variables.

If the physical constrained target density is written with respect to

\[
da\,dc,
\]

and

\[
y_a=\log(a/l_{\rm ref}),
\qquad
y_c=\log(c/l_{\rm ref}),
\]

then

\[
a=l_{\rm ref}e^{y_a},
\qquad
c=l_{\rm ref}e^{y_c},
\]

so

\[
\left|
\det
\frac{\partial(a,c)}
{\partial(y_a,y_c)}
\right|
=
ac.
\]

Thus the change-of-variables contribution is

\[
\boxed{
\log |J|
=
\log a+\log c
}
\]

up to a constant.

The subsequent standardization

\[
y_i=\mu_i+\sigma_i z_i
\]

has constant Jacobian determinant, so it does not add a nonconstant score term.

## Implementation rule

Whenever the code uses:

- target log density,
- target score,
- path weight,
- importance weight,
- reverse/forward density ratio,
- any other density-based quantity,

the target must be evaluated in the **correct transformed coordinate measure**.

Do not copy the old 6D statement `+log V`.

The 2D transform has a different Jacobian.

Also, the physical equilibrium target may already contain a cell-measure factor.

Therefore:

1. derive the physical constrained target density with respect to \(da\,dc\);
2. transform it to \(y\);
3. then to \(z\);
4. include each Jacobian exactly once.

Add unit tests.

## FM-specific clarification

If endpoint samples are already transformed into \(z\) and one trains a vanilla flow-matching regression loss on those samples, do **not** multiply the FM loss by an extra Jacobian weight merely because the coordinate system changed.

The Jacobian matters when using the target density/score/log-weights, not as an arbitrary regression-loss multiplier.

The analytic cost is negligible relative to the EAM oracle.

---

# 14. Species channel

At fixed composition with \(n\) Cr atoms:

\[
N_{\rm Cr}=n
\]

must hold exactly.

Do not redesign the species channel unnecessarily during this cell pivot.

For the paper-faithful reproduction, continue using/awaiting the author-confirmed exact constrained reveal rule.

For the unified model, **do not block implementation on that reply**.  
Use the best currently validated exact-composition mechanism that guarantees

\[
N_{m Cr}=n
\]

at terminal time, and keep it modular so it can be swapped later if needed.

The unified model's claim is about one common FCC/BCC state space, not about reproducing the authors' unpublished constrained-reveal implementation.

---

# 15. Oracle and graph cutoff distinction

Current working physical target:

- `Ni-Co-Cr_v1.eam.fs`
- native `6.0 Å` cutoff

Published JANUS graph cutoffs:

- FCC `5.0 Å`
- BCC `5.3 Å`

The unified BCT model has no published single graph cutoff.

Therefore any graph cutoff for the unified model is a **new-model hyperparameter**.

Do not present it as paper-faithful.

Validate and document it separately.

---

# 16. Validation before expensive training

Before long runs, verify all of the following:

1. `u=0`, `c/a=1` produces the intended BCC endpoint.
2. `u=0`, `c/a=sqrt(2)` produces the FCC-equivalent endpoint.
3. The same `N=128` site list is used at both endpoints.
4. Primitive/full-supercell conventions are consistent.
5. `z -> y -> (a,c) -> L` round-trips accurately.
6. The Jacobian matches analytic/autograd checks.
7. Both endpoint cells lie at reasonable normalized \(z\) values.
8. Cu–Ni default \(g(T)\) gives stable short rollouts.
9. RMS \(u\) stays in the intended local-displacement regime.
10. No old 6D cell checkpoint is accidentally loaded.
11. Path-weight / ESS diagnostics remain numerically sane.

---

# 17. Evaluation target

Evaluate against the **new 2D tetragonal reference MC**.

Do not compare model samples to the legacy 6D reference as if they were the same target.

The desired final scientific statement is narrow:

> A single Ni–Cr generative sampler can represent and traverse BCC-like, BCT-intermediate, and FCC-equivalent structures in one common `N=128` BCT state space.

Do not claim unconstrained full-cell thermodynamics, because angles and shear are intentionally excluded.


---

# 18. Immediate implementation instruction

Status:

\[
\boxed{\text{READY TO IMPLEMENT WITHOUT FURTHER AUTHOR INPUT}}
\]

The agent should proceed with implementation and short validation runs immediately.

Only pause for author input if the task being performed is explicitly the **paper-faithful JANUS Ni–Cr reproduction**, not the new unified BCT-2D model.
