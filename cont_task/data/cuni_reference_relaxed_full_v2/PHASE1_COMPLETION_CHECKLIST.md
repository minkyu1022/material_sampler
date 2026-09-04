# Cu–Ni Phase 1 completion checklist

This checklist maps every goal requirement to authoritative evidence. A final PASS is allowed only after all pending rows are verified for all 376,200 structures.

| Requirement | Evidence | Current state |
|---|---|---|
| 376,200 reference frames relaxed | `validation_report.json`: `exact_source_frame_identity`, `all_relaxations_converged` | IN PROGRESS |
| Same Fischer Cu–Ni EAM | `run_manifest.json`, `eam_identity_audit.json`, potential SHA256 `ee585bf9884a4ac2548abb395e81a12a941505983aaa058ef546929df74cf240` | PASS |
| ASE BFGS + FrechetCellFilter | `run_manifest.json`, `provenance_manifest.json` | PASS |
| Coordinates + all six cell DOF | `coordinate_cell_dof_audit.json`; final `optimizer_is_bfgs_frechet` check | PILOT PASS / FINAL PENDING |
| Raw extxyz | per-chain `raw/frame_*.extxyz`; `all_structure_artifacts_nonempty` | IN PROGRESS |
| Relaxed extxyz | per-chain `relaxed/frame_*.extxyz`; `all_structure_artifacts_nonempty` | IN PROGRESS |
| CIF | per-chain `cif/frame_*.cif`; `sampled_cifs_readable` | IN PROGRESS |
| Processed `.pt` | `cont_task/data/processed/cuni_tokens.pt`; `processed_*` checks | PENDING |
| Crystalite train/val datasets | `cont_task/data/crystalite_cuni/processed/*.pt`; split checks | PENDING |
| Provenance manifest | `provenance_manifest.json`, source/code/lock hashes | CURRENT PASS / FINAL UPDATE PENDING |
| Composition counts | `cuni_tokens.manifest.json`; match against all source frames | PENDING |
| Duplicate/diversity statistics | `cuni_tokens.manifest.json`: `diversity_by_n_cu` | PENDING |
| Species-coordinate order preserved | per-frame records + source/payload validation | LIVE PASS / FINAL PENDING |
| Failed/non-converged list | `cont_task/data/processed/cuni_tokens.failures.json` | PENDING |
| Reproducible uv command | `command.txt` | PASS |
| uv lockfiles | hashes in `provenance_manifest.json` | PASS |
| No activation checkpointing | provenance + final validation check | PASS |
| No premature 8-DDP main training | provenance + final validation check | PASS |
| Final validation | `validation_report.json`, `PHASE1_VALIDATION.md` | PENDING |

## Final command

```bash
cd /home/spml_minkyu_kim/joint_sampler/reference/crystalite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  uv run python ../../cont_task/data/validate_cuni_phase1.py \
  --source ../JANUS/janus_reproduce/outputs/cuni_reference_n108_full \
  --relaxed ../../cont_task/data/cuni_reference_relaxed_full_v2 \
  --processed ../../cont_task/data/processed/cuni_tokens.pt \
  --workers 32
```
