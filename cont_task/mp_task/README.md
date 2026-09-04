# MP20 transferable continuous CSP task

## Scope

- Base model: public Crystalite MP20 CSP checkpoint (`csp_mp20_best.pt`).
- Condition: exact composition (element identities and counts) plus temperature.
- Fixed discrete species; generate only fractional coordinates and the six-element
  lower-triangular cell representation.
- Transfer across MP20 compositions rather than specializing to Cu–Ni.
- Energy/force/stress oracle: one pinned UMA checkpoint used consistently.

## Layout

- `data/`: MP20 acquisition, immutable raw data, processed splits, manifests.
- `pre_train/`: checkpoint loading and base-model validation only.
- `post_train/bms/`: continuous JANUS/BMS post-training.
- `post_train/proximal_ce/`: continuous proximal-CE post-training.
- `configs/`: reproducible experiment configurations.
- `scripts/`: launch and evaluation entry points.
- `docs/`: target-measure, coordinate, cell, and oracle audits.

## Target distinction

- Proximal CE: `p_target(x|c,T) ∝ p_base(x|c) exp[-β U_UMA(x,c)]`.
- BMS: the pretrained model initializes the sampler, while the physical endpoint
  target and its coordinate/cell measure must be specified independently.

Before training, audit the six-DOF cell measure/Jacobian, Cartesian-force to
fractional-score conversion, stress to cell-score conversion, temperature
distribution, and UMA model/version consistency.

