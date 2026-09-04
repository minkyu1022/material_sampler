# Cu-Ni JANUS implementation audit

Basis: paper/SI Eqs. (1), (5), (8), (10)--(16), (S6)--(S7), (S33)--(S36),
Algorithms 1--2, and the SI experimental settings.

## Training-path checklist

| Item | Status | Evidence / action |
|---|---|---|
| Semi-grand NPT target including `exp(Nv)` | verified | finite differences cover fractional displacement, log-volume, and heat bath |
| Channel coordinates | corrected | `u` stays fractional and `v=log(V)`; fields are no longer incorrectly converted to Cartesian units |
| Species prior | corrected | removed an unpublished composition-logit bias; generation starts all-mask with fair-coin zero head |
| Continuous priors | verified | calibrated conditioned Gaussians and default `sqrt(T/Tref)` displacement width |
| Condition sampling | verified | uniform inverse temperature and equal wide/narrow offset mixture around the published line |
| Interpolation/masking | verified | linear continuous interpolant and independent mask probability `1-t` |
| Velocity and generalized score targets | verified | `x1-x0` and SI (S6)--(S7); unnecessary endpoint cutoff removed |
| Heat-bath soft labels | verified | all-site substitution energies and chemical-potential sign tested by finite differences |
| Continuous rollout | corrected | Euler--Maruyama includes `b+g^2 s` and `sqrt(2 g^2 dt)` noise |
| Diffusion temperature scaling | corrected | `g_c(T)=g_c sqrt(T/Tref)` |
| Discrete reveal | verified | paper tau-leap probability; final step reveals every remaining site |
| Network heads/inputs | corrected | zero heads; displacement magnitude, species/mask coordination, normalized conditions and log-volume |
| Species gradient isolation | verified | species head reads a detached trunk |
| Optimizer/loss/budget | corrected/verified | Adam `3e-3`, lambda=2, M=100, 120x1000 terminals, 500 updates, batch 96, replay 5000 |
| Numerical safety | corrected | non-finite output/state/loss aborts before checkpointing |
| Resume safety | corrected | pre-audit checkpoints are rejected by checkpoint version |
| DDP | verified | explicit-loopback 8-GPU smoke completed and checkpointed |

## Unpublished reconstruction choices

- Numerical `g_u(t)` and `g_v(t)` are absent. Constant base strengths plus published temperature
  scaling are swept; `g=0` is the exact probability-flow baseline.
- Numerical `sigma_v` and its fitting recipe are absent. We use calibrated `Var[log(V)]` and sweep
  its scale.
- SI gives both `p=0.5` and prose saying to fit `p` for `sigma_u(T)`. Primary uses `0.5`; fitted `p`
  is an ablation.
- Exact feature-normalization constants and coordination implementation are absent and remain
  explicitly documented reconstruction details.

## Remaining before a final reproduction claim

- The correct labels currently require two Torch-EAM kernel calls per terminal instead of the
  paper's single fused energy-force-virial-substitution pass. This affects cost accounting, not the
  label values.
- Algorithm 2 path weights, ESS, and the complete Cu-Ni Fig. 2 evaluation pipeline are still needed.
  They do not alter training but are mandatory before calling the experiment reproduced.

The old `outputs/cuni_primary_p05_sv1` checkpoint predates this audit and must not be resumed.
