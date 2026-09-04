# Server migration and continuation guide

## 1. Clone code

```bash
git clone https://github.com/minkyu1022/material_sampler.git
cd material_sampler
```

Install OMX/Discord only if required by the destination runtime. Never copy a committed credential file; create credentials locally.

## 2. Restore large artifacts

Dropbox links and SHA256 checksums are recorded in `docs/handoff/DATA_MANIFEST.json` after upload. Download each archive, verify its SHA256, then extract from the repository root while preserving relative paths.

Example:

```bash
curl -L '<DROPBOX_DIRECT_DOWNLOAD_URL>' -o artifact.tar
sha256sum -c artifact.tar.sha256
tar -xf artifact.tar
```

Do not use an archive whose checksum does not match.

## 3. Recreate uv environments

```bash
uv sync --project reference/JANUS/janus_reproduce --frozen
uv sync --project reference/crystalite --frozen
```

Run the Phase-1 tests:

```bash
cd reference/crystalite
PYTHONPATH=. uv run pytest -q \
  ../../cont_task/data/test_relax_cuni_dataset.py \
  ../../cont_task/data/test_build_cuni_crystalite_dataset.py
cd ../..
```

## 4. Resume Phase-1 relaxation

After restoring the source NPZ directory and partial output, run the exact command saved in:

`cont_task/data/cuni_reference_relaxed_full_v2/command.txt`

The resume logic skips only converged frames whose raw extxyz, relaxed extxyz, and CIF all exist and are nonempty. Use tmux:

```bash
tmux new-session -d -s cuni-relax-full-v2 \
  "cd '$PWD' && exec bash cont_task/data/cuni_reference_relaxed_full_v2/command.txt \
   >> cont_task/data/cuni_reference_relaxed_full_v2/stdout.log 2>&1"
```

Start the monitor after checking paths inside it:

```bash
tmux new-session -d -s cuni-relax-monitor \
  "cd '$PWD' && exec bash cont_task/data/monitor_cuni_relax.sh"
```

Immediately verify PID, 64 child workers, increasing record count, disk space, and logs.

## 5. Phase-1 completion gate

Do not start 8-DDP main training before all 376,200 structures are processed and:

```bash
cd reference/crystalite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python ../../cont_task/data/validate_cuni_phase1.py \
  --source ../JANUS/janus_reproduce/outputs/cuni_reference_n108_full \
  --relaxed ../../cont_task/data/cuni_reference_relaxed_full_v2 \
  --processed ../../cont_task/data/processed/cuni_tokens.pt \
  --workers 32
```

returns zero and `validation_report.json` says `passed: true`.

## 6. Continue A/B/C diagnosis

1. Let full B **regular-weight** evaluation finish; do not use the short-run EMA result.
2. Run the same full regular-weight evaluation for A.
3. Render representative target/A/B structures and recompute exact match, validity, volume, and minimum-distance statistics.
4. Let corrected C finish. Evaluate regular weights first because its `ema_decay=0.9999` is also too slow for 7,100 updates.
5. Only alter EMA for future runs after a teacher-forced vs self-rollout comparison proves it is valid.
6. Select the pretraining representation based on physical sampling quality, not training loss alone.

## 7. Main training capacity rules

- No activation checkpointing.
- Measure one-GPU memory and throughput first.
- Then measure 8-DDP with the intended **per-GPU** batch size.
- Preserve Crystalite architectural ratios when resizing.
- If OOM remains, report before launching the main run.

## 8. Discord reporting

Project profile: `joint-sampler`.

```bash
printf '%s\n' 'MESSAGE' | ./.agent-tools/discord/discord-control.sh send --stdin
```

The helper code may be versioned, but `profile.env` and all tokens must remain local and ignored. Keep messages concise; split long reports.

## 9. Do not

- Do not re-run completed expensive jobs merely to reconstruct metadata.
- Do not use bad EMA checkpoints for scientific conclusions.
- Do not add reference-MC samples to a self-bootstrapped training buffer unless the experiment explicitly calls for supervised/reference injection.
- Do not mix potentials, cutoffs, priors, replay labels, path weights, or free-energy evaluation from different Hamiltonians.
- Do not hide provisional reconstruction choices; mark them.
- Do not silently omit failures.
