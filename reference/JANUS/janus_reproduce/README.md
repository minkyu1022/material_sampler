# JANUS reproduction

Reimplementation of the hybrid neural sampler and validation pipelines from arXiv:2608.19116.

## Setup

```bash
uv sync --extra dev
source .venv/bin/activate
wandb login --verify
pytest
```

Run the implemented Figure 2a path:

```bash
python scripts/run_ising.py --length 16 --temperature 2.2692 --wandb
```

Implemented and tested:

- masked periodic Ising JANUS head, fixed-point training, any-order sampling, and ghost-spin Wolff
  reference sampling;
- linear continuous interpolant, bounded generalized score target, masked soft-label objective, and
  simultaneous reveal step;
- torch-only periodic PaiNN-like alloy model with four 64-channel interaction layers, 16 radial
  basis functions, live-volume graph rebuilding, and five zero-initialized JANUS heads;
- semi-grand/canonical NPT Monte Carlo, replica exchange, neighboring-rung BAR work, path-weight
  normalization, mixing free energy, common tangents, RDF, and Warren-Cowley SRO.

The NPT target is represented in `v=log(V)` and includes the required `N*v` Jacobian.

## Reproduction boundary

The paper archive does not publish the exact EAM files/checksums, fitted prior widths, diffusion
schedule, fixed-composition reveal rule, Frenkel-Ladd integration settings, MACE/optical surrogate
artifacts, LLM protocol, or DFT outputs. The code does not silently invent these values. Exact alloy,
inverse-design, and defect figures require those provenance inputs before a defensible reproduction
claim can be made.
