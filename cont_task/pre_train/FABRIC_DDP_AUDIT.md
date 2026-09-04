# Crystalite Fabric DDP audit

**Result:** PASS for the Phase-1 CSP 8-DDP path (2026-09-04).

Independently verified after remediation:

- multi-device runs use a 24-hour collective timeout for rank-zero sampling;
- rank zero alone prepares split/token caches before an all-rank barrier;
- CSP does not load the unused de-novo reference dataset;
- validation runs once through the unwrapped model on rank zero and is broadcast;
- EMA updates and checkpoint/W&B/sampling writes are rank-zero consistent;
- thermo MLIP initialization is rank-zero-only;
- batch size remains per rank (`128 × 8 = 1024` globally).

Fresh evidence:

- 2-GPU, 2-step training + validation + checkpoint smoke: PASS;
- 8-GPU, 1-step DDP smoke: PASS;
- saved checkpoint has 205 unprefixed state keys (no `module.` prefix);
- `py_compile` and `git diff --check`: PASS.
