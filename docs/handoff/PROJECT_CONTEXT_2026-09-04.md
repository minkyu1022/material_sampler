# Material Sampler project context — 2026-09-04

This document is the current handoff source of truth. It contains no credentials.

## 1. Research objective

Build and evaluate crystal samplers in two lanes:

1. **Continuous Cu–Ni CSP (`cont_task`)**
   - Dataset: Cu–Ni JANUS reference-MC structures, fully relaxed to 0 K local minima.
   - Pretraining candidates derived from Crystalite CSP:
     - A: original fractional/EDM baseline.
     - B: fractional coordinates + Packora-style VFM endpoint-L1 training.
     - C: centered Cartesian coordinates + dataset normalization + VFM endpoint-L1 training.
   - Post-training methods planned under separate method directories:
     - JANUS/BMS continuous post-training using one clean-endpoint head reused for velocity and generalized target score.
     - PDNS/Proximal-CE post-training with base distribution supplied by the pretrained sampler.

2. **Joint Cu–Ni real-CSP (`joint_task`)**
   - Condition on atom count N and allowed element set such as `[Cu, Ni]`.
   - Composition uses constrained masked discrete diffusion.
   - Coordinates/cell use the same carefully audited continuous representation.
   - Planned post-training: JANUS BMS + soft CE, and continuous/discrete PDNS.
   - This lane is not the current Phase-1 execution target.

Prior Ni–Cr and JANUS reproduction work remains in `reference/JANUS/janus_reproduce` and associated reports. Do not conflate those outputs with the new Crystalite pretraining project.

## 2. Active native goal: Cu–Ni Phase 1

Goal file originally supplied to Codex:
`/home/spml_minkyu_kim/.codex/attachments/f516fd31-ed2a-4597-a001-e518c0a58cb7/pasted-text-1.txt`

### Required result

Relax all **376,200** Cu–Ni reference-MC frames with the same Fischer EAM using ASE `BFGS(FrechetCellFilter)` over atomic coordinates and all six cell degrees of freedom, then build all training artifacts and statistics.

### Running relaxation

- tmux: `cuni-relax-full-v2`
- controller PID at handoff: `285671` (PID is host-specific; rediscover after migration)
- monitor tmux: `cuni-relax-monitor`
- monitor PID at handoff: `309506`
- workers: 64
- input: `reference/JANUS/janus_reproduce/outputs/cuni_reference_n108_full`
- output: `cont_task/data/cuni_reference_relaxed_full_v2`
- exact command: `cont_task/data/cuni_reference_relaxed_full_v2/command.txt`
- logs: `stdout.log`, `progress.jsonl`, `status.json`, `monitor.log`
- potential: `reference/JANUS/janus_reproduce/potentials/cu_ni/Cu_Ni_Fischer_2018.eam.alloy`
- potential SHA256: `ee585bf9884a4ac2548abb395e81a12a941505983aaa058ef546929df74cf240`
- relaxation: fmax=0.01 eV/Å, max 500 BFGS steps, maxstep=0.1, zero external pressure.
- restart is resumable and skips only converged frames with nonempty raw/relaxed/CIF artifacts.

### Verified live evidence

- 990 source NPZ files / 376,200 frames / 108 atoms each.
- All N(Cu)=0..108 compositions occur in the source.
- Source hash manifest contains all 990 NPZ hashes.
- Same-EAM audit: max recomputation discrepancy vs relaxation record `6.33e-8 eV`; PASS.
- Species/coordinate/CIF round-trip random audit: errors 0; max fractional error about `4.74e-10`; PASS.
- 256-structure coordinate/cell audit: atomic coordinates, diagonal cell, and off-diagonal cell entries all changed; PASS.
- Resume safeguard corrected: zero-byte files are no longer treated as completed.
- Phase-1 tests: 7 passed.
- Full postprocessing estimate: builder roughly 92 minutes; source-payload validator benchmark ~329 structures/s and ~688 MiB RSS for 1,000 structures.
- Completion checklist: `cont_task/data/cuni_reference_relaxed_full_v2/PHASE1_COMPLETION_CHECKLIST.md`.

### Automatic finish chain

`cont_task/data/monitor_cuni_relax.sh` watches the process, reports to Discord every 30 minutes, retries recoverable failure up to three times, then runs:

1. `cont_task/data/build_cuni_crystalite_dataset.py`
2. `cont_task/data/validate_cuni_phase1.py`

Expected final products:

- per-frame raw extxyz, relaxed extxyz, CIF
- `cont_task/data/processed/cuni_tokens.pt`
- processed manifest and explicit failure list
- Crystalite train/val `.pt` splits
- composition, approximate duplicate, energy/volume diversity statistics
- provenance and final `validation_report.json` / `PHASE1_VALIDATION.md`

Do not mark Phase 1 complete until the final validator reports PASS for every explicit check.

## 3. Pretraining experiment state

### A/B completed training

Both used 10k data, 100 epochs / step 7100, and were initially evaluated with EMA weights over all 1,007 validation compositions using 200 sampling steps (~400 NFE).

Initial EMA evaluation:

- A exact match: 0/1007; physically valid: 18/1007; median volume ~750 Å³ vs target ~1267; median minimum distance ~0.264 Å.
- B exact match: 0/1007; physically valid: 0/1007; median volume ~44.4 Å³; median minimum distance ~0.0159 Å.

### Root cause of catastrophic B rendering

The B screenshot showed 108 atoms collapsed on top of one another. This was **not** primarily an NFE/integrator failure.

- NFE sweep at 1/10/50/200 steps always produced median volumes ~42–46 Å³.
- Teacher-vs-rollout diagnosis found the EMA model predicts ~52 Å³ even at teacher-forced t=0.99.
- The regular (non-EMA) model predicts ~1215 Å³ against target ~1222 Å³ and self-rolls to ~1214 Å³.
- Cause: `ema_decay=0.9999` with only 7,100 updates leaves `0.9999^7100 = 49.16%` weight on initialization. The short overfit run's EMA is therefore unusable.

Artifacts:

- `cont_task/pre_train/outputs/cuni_overfit/b_nfe_diagnostic/nfe_diagnostic.json`
- `.../teacher_vs_rollout.json` (bad EMA)
- `.../teacher_vs_rollout_regular.json` (healthy regular weights)
- diagnostic code: `cont_task/pre_train/diagnose_b_vfm_rollout.py`

At handoff, full B regular-weight 1,007-sample evaluation runs in tmux `cuni-b-regular-eval`; command and log are under `cont_task/pre_train/outputs/cuni_overfit/match_full_400nfe_regular/`.

**Action:** evaluate A and B using regular weights. Do not retrain B merely to fix the bad EMA rendering. For future short runs, either evaluate regular weights or choose an EMA schedule/decay whose effective window fits the number of updates; verify teacher-forced and rollout behavior before trusting EMA.

### C corrected Cartesian run

The first Cartesian C attempt was invalid because random fractional translation followed by modulo wrap and Cartesian COM centering creates discontinuous targets. The fix is:

- retain Crystalite Niggli reduction;
- use six lower-triangular cell parameters (`ltri`);
- convert the reduced structure directly to Cartesian coordinates;
- arithmetic COM center;
- normalize from dataset statistics;
- **no random fractional translation/modulo wrapping during training**;
- decode Cartesian→fractional and wrap only at final output.

Current run at handoff:

- tmux `cuni-c-fixed-20260904_084715`
- monitor `cuni-c-fixed-monitor`
- GPUs 4,5; 2-GPU diagnostic/overfit, batch 64/GPU (global 128)
- d_model 768, 12 heads, 14 layers, 152,896,520 params
- max steps 7100, bf16, no activation checkpointing
- command: `cont_task/pre_train/outputs/cuni_overfit/c_cartesian_fixed_direct_20260904_084715.command.txt`
- log/output share the same basename.

The same EMA caveat applies to C; retain regular checkpoints and do not judge sampling solely from `ema_decay=0.9999` EMA weights.

## 4. Mathematical/representation invariants

- Species and coordinate arrays are paired; never sort species independently.
- Cu–Ni reference encoding is `1=Cu`, `0=Ni`; atom numbers are Cu=29, Ni=28.
- Niggli reduction may change lattice basis but must preserve site/species order.
- Fractional coordinates live on a torus. Use minimum-image deltas for fractional losses.
- Cartesian training target must be direct reduced-cell Cartesian coordinates with COM centering; do not create wrap discontinuities.
- `ltri` means `[log L00, L10, log L11, L20, L21, log L22]`.
- Final Cartesian samples are converted by solving against the generated cell, then wrapped to fractional coordinates once.
- One clean-endpoint model head may parameterize endpoint regression; any derivation of velocity/score from it must be mathematically consistent with the chosen bridge/prior.
- BMS and PDNS are not automatically the same target distribution. Document the base sampler and energy tilt explicitly.

## 5. Non-negotiable operating rules

- Use `uv`; preserve lockfiles.
- Do not use activation checkpointing.
- A reported running job must include a live PID, exact command, log, output, start time, and evidence of increasing output.
- Long jobs run in tmux/background and continue after the agent turn.
- Monitor at least every 30 minutes and report milestones, failure, OOM, disk pressure, or ETA changes >15% proactively to Discord.
- Do not silently discard failed or unconverged structures.
- Do not claim completion without artifact inspection and fresh validation.
- Do not start the 8-DDP main pretraining run before Phase 1 has final PASS.
- Per-GPU batch size means per-GPU batch size. Example: batch 128/GPU on 8 GPUs means global batch 1024, not 128 split over GPUs.
- Do not enable activation checkpointing as an OOM workaround. If a required configuration still OOMs after ordinary safe memory settings, stop before main training and report.
- Never commit or publish API keys, W&B tokens, Discord credentials, Dropbox tokens, `.env`, or `.agent-tools/discord/profile.env`.

## 6. Reference code and papers

Important local references:

- `reference/crystalite`: Crystalite implementation and paper resources.
- `reference/packora`: Packora paper/code notes.
- `reference/pairmixer`: upstream PairMixer implementation.
- `reference/PDNS`: PDNS continuous/discrete code.
- `reference/BridgeMatchingSampler`: BMS reference.
- `reference/AtomMOF`: Cartesian/lattice implementation reference; do not copy MOF fragment-unwrapping logic into atomic Cu–Ni unless needed.
- `reference/JANUS/janus_reproduce`: JANUS reproduction, EAM potentials, reference MC, previous Ni–Cr/Cu–Ni work.
- `reference/algorithm_scripts`: theoretical notes.

## 7. Secrets and external services

A Materials Project API key was supplied in prior conversation. It must be treated as secret and transferred through a secure secret manager or environment variable, never this repository or handoff archives.

W&B, Discord, GitHub push, and Dropbox credentials must likewise be configured independently on the destination server.
