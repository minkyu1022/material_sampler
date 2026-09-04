# Cu–Ni Phase 1 completion checklist

This checklist maps every goal requirement to authoritative evidence for the user-authorized frozen subset of 136,826 structures. The original 376,200-frame run was stopped after the diversity audit in `paused_subset_diversity_audit.json`; exact accepted membership is fixed by `frozen_subset_manifest.json`.

| Requirement | Evidence | Current state |
|---|---|---|
| 136,826 accepted reference frames relaxed | `validation_report.json`: `exact_source_frame_identity`, `all_relaxations_converged`, `accepted_subset_manifest_matches` | PASS |
| Same Fischer Cu–Ni EAM | `run_manifest.json`, `eam_identity_audit.json`, `eam_recompute_audit.json`; potential SHA256 `ee585bf9884a4ac2548abb395e81a12a941505983aaa058ef546929df74cf240` and 64-structure energy recomputation | PASS |
| ASE BFGS + FrechetCellFilter | `run_manifest.json`, `provenance_manifest.json` | PASS |
| Coordinates + all six cell DOF | `coordinate_cell_dof_audit.json`; `optimizer_is_bfgs_frechet` | PASS |
| Raw extxyz | 136,826 accepted per-chain `raw/frame_*.extxyz`; `all_structure_artifacts_nonempty` | PASS |
| Relaxed extxyz | 136,826 accepted per-chain `relaxed/frame_*.extxyz`; `all_structure_artifacts_nonempty` | PASS |
| CIF | 136,826 accepted per-chain `cif/frame_*.cif`; `sampled_cifs_readable` | PASS |
| Processed `.pt` | `cont_task/data/processed/cuni_tokens.pt`; `processed_*` checks | PASS |
| Composition-balanced view | `cuni_tokens_balanced_cap1000.pt`; five `balanced_view_*` checks | PASS |
| Crystalite train/val datasets | `cont_task/data/crystalite_cuni/processed/*.pt`; split checks | PASS |
| Provenance manifest | `provenance_manifest.json`, source/code/lock/artifact hashes | PASS |
| Composition counts | `cuni_tokens.manifest.json`; match against accepted source frames | PASS |
| Duplicate/diversity statistics | `cuni_tokens.manifest.json`: `diversity_by_n_cu` | PASS |
| Species-coordinate order preserved | per-frame records + all processed payload validation | PASS |
| Failed/non-converged list | `cont_task/data/processed/cuni_tokens.failures.json` (empty) | PASS |
| Reproducible uv command | `command.txt` | PASS |
| uv lockfiles | hashes in `provenance_manifest.json` | PASS |
| No activation checkpointing | provenance + final validation check | PASS |
| No premature 8-DDP main training | provenance + final validation check | PASS |
| Final validation | `validation_report.json`, `PHASE1_VALIDATION.md`: 33/33 checks | PASS |

## Final command

```bash
cd /home/spml_minkyu_kim/joint_sampler/reference/crystalite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  uv run python ../../cont_task/data/validate_cuni_phase1.py \
  --source ../JANUS/janus_reproduce/outputs/cuni_reference_n108_full \
  --relaxed ../../cont_task/data/cuni_reference_relaxed_full_v2 \
  --processed ../../cont_task/data/processed/cuni_tokens.pt \
  --balanced ../../cont_task/data/processed/cuni_tokens_balanced_cap1000.pt \
  --subset-manifest ../../cont_task/data/cuni_reference_relaxed_full_v2/frozen_subset_manifest.json \
  --workers 32
```
