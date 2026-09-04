# JANUS Ni–Cr final-v3 implementation plan

Source of truth: `JANUS_NiCr_reproduction_final_v3_CuNi_inherited_gscale_tuning.md`.

## Behavior locks

- Pre-change archive: `outputs/snapshots/pre_author_confirmed_nicr_2026-09-01/code.tar.gz`
- Archive SHA256: `7c6757e3102175cc8696fcc0fd899bfb570f8b29668a70c7a37d5d96493e3dcd`
- Cu–Ni focused regression: 22 passed.
- Frozen Cu–Ni parent: `outputs/cuni_corrected_diff002_tref750/checkpoint.pt`, round 120.

## Minimal refactor

1. Add flat `losses.py` and `samplers.py` registries; do not reorganize the whole package.
2. Route Cu–Ni through `tsm` + `sce` and `janus_tau_leap` without changing numerics.
3. Add author-confirmed one-site boundary-quota kernel and keep current DP kernel as `OUR_METHOD` ablation.
4. Add exhaustive deterministic quota/logq tests before Ni–Cr training code.
5. Add one phase-parameterized Ni–Cr trainer reused by separate FCC/BCC configs; never add a phase token.
6. Inherit the actual Cu–Ni reproduced architecture/optimizer/replay/clipping/g-scale settings; override only condition, lattice/N/M, fresh terminals, Ni–Cr prior/potential/cutoff.
7. Add thin shell wrappers, resolved config and hardware/potential/code provenance.
8. Profile, then run FCC smoke and BCC smoke sequentially. Tune only a diagnosed axis, including local g-scale reductions while preserving sqrt(T/750).

## Stop gates

- No full training until both smokes have exact counts, finite weights, stable physics, acceptable forcing tails, and no correctness failure.
- No BCT/Bain work.
- No DP baseline until separate FCC/BCC JANUS reproduction is stable.
