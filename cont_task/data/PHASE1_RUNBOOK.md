# Cu–Ni Phase 1 relaxation and Crystalite preprocessing

## Reproducible environments

- Relaxation lock: `reference/JANUS/janus_reproduce/uv.lock`
- Crystalite preprocessing lock: `reference/crystalite/uv.lock`
- EAM: `reference/JANUS/janus_reproduce/potentials/cu_ni/Cu_Ni_Fischer_2018.eam.alloy`
- EAM SHA-256: `ee585bf9884a4ac2548abb395e81a12a941505983aaa058ef546929df74cf240`

## Relax all 376,200 reference-MC frames

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run --project reference/JANUS/janus_reproduce \
python cont_task/data/relax_cuni_dataset.py \
  --input-dir reference/JANUS/janus_reproduce/outputs/cuni_reference_n108_full \
  --potential reference/JANUS/janus_reproduce/potentials/cu_ni/Cu_Ni_Fischer_2018.eam.alloy \
  --output-dir cont_task/data/cuni_reference_relaxed_full_v2 \
  --workers 64 --fmax 0.01 --max-steps 500 --maxstep 0.1
```

This uses `BFGS(FrechetCellFilter)` at zero pressure, with atomic coordinates and all six cell strain DOFs active. Species decoding follows the reference producer: `1=Cu`, `0=Ni`.

## Build Crystalite-compatible tokens

```bash
uv run --project reference/crystalite \
python cont_task/data/build_cuni_crystalite_dataset.py \
  --input cont_task/data/cuni_reference_relaxed_full_v2 \
  --output cont_task/data/processed/cuni_tokens.pt \
  --crystalite-root cont_task/data/crystalite_cuni
```

Crystalite's source pipeline does not sort atoms by element. It parses structures, applies Niggli reduction without primitive reduction, and retains paired species/coordinates. The converter follows that behavior and records a failure if the species-coordinate order changes. Token fields are `A0` atomic numbers, wrapped fractional `F1`, and `Y1=[log(a),log(b),log(c),cos(alpha),cos(beta),cos(gamma)]`.

The converter also creates the exact files discovered by `MP20Tokens(..., nmax=108)`: `processed/mp20_tokens_{train,val}_nmax108.pt` plus the split markers `raw/{train,val}.csv`. Training uses `--dataset_name custom --data_root cont_task/data/crystalite_cuni --nmax 108`.

## Final validation

```bash
uv run --project reference/crystalite \
python cont_task/data/validate_cuni_phase1.py \
  --source reference/JANUS/janus_reproduce/outputs/cuni_reference_n108_full \
  --relaxed cont_task/data/cuni_reference_relaxed_full_v2 \
  --processed cont_task/data/processed/cuni_tokens.pt
```

Success requires all source frames recorded and converged, all raw/relaxed/CIF artifacts present, processed count equality, no preprocessing failures, order preservation, matching EAM hash, and the required optimizer/cell filter.
