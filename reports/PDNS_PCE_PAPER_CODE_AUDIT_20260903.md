# PDNS Proximal CE / WDCE paper-to-code audit

Date: 2026-09-03  
Paper: arXiv:2510.03824v2 (2026-05-19)  
Official repository: `AlexandreGUO2001/PDNS`  
Audited commit: `2abd5569fc29c4c47416f78f067948f1474b58c2` (2026-04-23)

## Verdict

The repository implements the paper's **continuous Proximal Weighted Denoising Cross-Entropy (Proximal WDCE)** algorithm, which is the practical denoising/bridge-matching instantiation of the abstract Proximal CE projection. It does **not** directly optimize the full trajectory negative log-likelihood form of Proximal CE.

The core implementation is consistent with the paper:

- controlled OU/VP reference SDE;
- Euler-Maruyama rollout;
- online Girsanov `log(dP_ref/dP_theta)` accumulation;
- terminal reward correction `r(X_T)=-E(X_T)-log nu(X_T)` at unit inverse temperature;
- proximal exponent `gamma=eta/(eta+1)`;
- adaptive trust-region scheduler;
- terminal replay buffer;
- exact reference-bridge resampling conditioned on fresh `X_0` and stored `X_T`;
- weighted/resampled conditional bridge-score matching.

However, the official code's neural network predicts a **control or adjoint/score-like field**, not the clean endpoint `X_T`. Reusing Crystallite's endpoint `x1` head is a mathematically valid reparameterization, but it is our adaptation, not the released implementation.

## Paper objective and implementation mapping

Paper continuous Proximal WDCE:

`E[w_k(X) * 1/2 ||u_theta(t,X_t) - sigma_t grad_{X_t} log P_ref(X_T|X_t)||^2]`.

Code:

- `continuous/src/components/matchers.py:101-112`: samples fresh source `x0`, stored terminal `x1`, random `t<0.999`, then calls the analytic reference bridge.
- `continuous/src/components/sdes.py:273-292`: samples the OU bridge and evaluates its terminal conditional score.
- `continuous/src/train_loop.py:35-50`: applies weighted MSE to control/adjoint parameterizations.

Thus the code is WDCE/bridge score matching rather than direct CE likelihood training.

## Path weight mapping

Paper:

`log(dP_ref/dP_theta) = integral[-1/2||u||^2 dt - u dot dW]`.

Code:

- `continuous/src/components/sdes.py:670-709`
- line 705 accumulates `-0.5*||u||^2*dt - u*z`, where `z=sqrt(dt)*N(0,I)`.

Paper terminal reward:

`r(x)=-beta E(x)-log nu(x)`.

Code at beta=1:

- `term_cost.py:9-15` returns `log nu(x)+E(x)=-r(x)` for the Gaussian reference.
- `scheduler.py:75-93` forms `log(dP_ref/dP_theta)-term_cost`, hence `log(dP_ref/dP_theta)+r`.

This sign convention is consistent.

## Proximal update mapping

For the non-cumulative form used by all provided continuous experiment configs (`cumulate: false`), code applies

`softmax(gamma * [log(dP_ref/dP_theta)+r])`,

matching the paper's

`(dP*/dP_{k-1})^(eta/(eta+1))`, with `gamma=eta/(eta+1)`.

- `scheduler.py:22-65`: adaptive gamma optimization.
- `scheduler.py:70-94`: applies fixed/adaptive proximal weights.
- `matchers.py:78-96`: resampling implementation; after multinomial resampling, weights are reset to one.

The repository defaults to the resampling-based WDCE variant when `iws: true`, despite the variable name suggesting importance weighting. The paper presents both direct weighting and resampling variants.

## Reference dynamics

Paper uses stationary OU:

`dX_t=-alpha_t X_t/2 dt + bar_sigma sqrt(alpha_t)dW_t`, `X_0~N(0,bar_sigma^2 I)`.

Code `VPSDE` implements the same family:

- drift: `-beta(t)x/2`;
- diffusion: `sigma*sqrt(beta(t))`;
- analytic OU bridge mean, variance, and conditional score.

Particle systems use `GraphVPSDE`/mean-free Gaussian variants, which project out center-of-mass translation. This is a symmetry-adapted specialization of the Euclidean theory.

## Actual differences and qualifications

### 1. Proximal CE versus Proximal WDCE

The phrase “Proximal CE” names the abstract reverse-KL projection. The released continuous experiments replace its negative log-likelihood with a bridge/denoising matching objective. Claims about the runnable method should therefore say **Proximal WDCE**, not plain direct Proximal CE.

### 2. Network output parameterization

Official code supports:

- `param_type=control`: model outputs stochastic control `u`;
- `param_type=adjoint`: model outputs a score/adjoint field and rollout multiplies it by `sigma^2`.

It does not contain an endpoint-denoiser `E[X_T|X_t]` head. For OU, an endpoint head `m_theta` can be converted exactly:

`u_theta = sigma_t*C_t/(bar_sigma^2*(1-C_t^2)) * (m_theta-C_t X_t)`.

Using this transformed output in the same WDCE loss is equivalent up to the known time-dependent quadratic scaling. This is suitable for Crystallite but must be implemented and tested explicitly.

### 3. Schedule direction differs from the paper prose

Paper v2 states `alpha_t=(1-t)alpha_min+t alpha_max`, with `alpha_max >> 1` (increasing schedule). Released configs use `beta0=10`, `beta1=0.1`, and `torch.lerp(beta0,beta1,t)` (decreasing schedule). The total integral, and therefore the terminal memorylessness factor, is unchanged by reversal for a linear schedule, but the intermediate bridge/control weighting changes. Reproduction should follow code/config or explicitly test both; do not silently equate them.

### 4. Numerical guards and clipping

The implementation adds details not central to the theorem:

- samples `t` only up to `0.999`;
- bridge denominators include `1e-6`/`1e-8`;
- terminal weights clipped to `[0,100]` by default;
- gradient norm clipped to `100`;
- many experiments zero final-step noise;
- EMA is used for rollout.

These can materially affect a crystal implementation and should be treated as part of the runnable baseline.

### 5. Stage-zero initialization

The paper recommends informative annealed initialization for sparse high-dimensional targets. The code uses an energy-gradient annealed SDE only for the initial buffer when `annealed: true`; later stages use the learned controlled SDE. Therefore the method is not energy-only at initialization: it requires target gradients/forces for this optional warm start, although subsequent WDCE labels use reference conditional scores and terminal energy weights.

### 6. Buffer semantics

The buffer is cleared at every proximal stage (`train.py:128-157`) and repopulated from the previous controller. It is not a long-lived accumulated replay buffer. Within a stage, terminal samples are repeatedly resampled/reused for multiple epochs.

### 7. Paper/repository revision timing and disclosed evaluation correction

The audited repository commit predates arXiv v2 by about four weeks. The repository README explicitly discloses that previously reported GMM40 Sinkhorn values were understated due to evaluator convergence/cost-convention issues and provides a corrected evaluator. This affects a reported evaluation metric, not the core Proximal WDCE training objective.

## Consequences for Crystallite

A paper/code-faithful adaptation should:

1. Retain one Crystallite terminal endpoint head per continuous channel.
2. Convert endpoint estimates to OU controls analytically before rollout and WDCE loss.
3. Use the exact OU reference bridge, not Crystallite's current Karras/EDM corruption unchanged.
4. Include the time-dependent scale in the endpoint-form loss; otherwise it is not equivalent to the official control loss.
5. Accumulate Girsanov weights using the converted control.
6. Include `-log nu` and any coordinate/cell Jacobian in the terminal reward.
7. Define a torus-valid reference process for fractional coordinates, or diffuse an unwrapped lift and wrap only for model geometry/oracle evaluation.
8. Combine continuous and masked-discrete path RN terms consistently for a joint crystal sampler.

## Sources

- Paper HTML: https://arxiv.org/html/2510.03824
- Paper source: https://arxiv.org/src/2510.03824
- Official code: https://github.com/AlexandreGUO2001/PDNS
