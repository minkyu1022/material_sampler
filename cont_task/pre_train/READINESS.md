# Cu–Ni Crystalite pretraining readiness

## Verified

- Dataset contract: Crystalite `MP20Tokens`, `nmax=108`, fixed atom order.
- Continuous CSP mode: atom types fixed; coordinate and cell losses active.
- Cell representation: `--lattice_repr ltri` (six lower-triangular parameters).
- Default CLI model: 86,851,176 parameters, per-GPU batch 128, bf16, peak 8,026 MiB.
- Larger provisional candidate: 150,198,504 parameters (`d_model=640`, `n_layers=20`),
  per-GPU batch 128, bf16, peak 11,190 MiB.
- Both configurations completed a forward/backward/optimizer step without activation
  checkpointing or OOM.
- The published MP-20 CSP checkpoint recipe is a different, 270,047,720-parameter
  model (`d_model=1024`, `n_heads=16`, `n_layers=14`). At `nmax=108` and batch 128
  per GPU it OOMs on an RTX 3090: peak observed usage was 23,860 MiB with the
  default allocator and 24,066 MiB with `expandable_segments:True`; both runs failed
  while requesting another 730 MiB. Production training with that configuration must
  not start without an explicit memory/config decision.
- With the user-approved per-GPU batch reduction to 64, the published 270,047,720-
  parameter model passes only the first optimizer step. Once AdamW state exists it
  OOMs on the second step in 8-GPU DDP. A sustained 20-step proportional sweep places
the observed boundary between 178.6M (pass) and 207.0M (OOM). The 152.3M model is
  the safer production candidate and also passes 20 DDP optimizer steps followed by
  rank-zero chunked sampling; see `cont_task/pre_train/architecture_capacity/RESULTS.md`.

Smoke data are repeated pilot structures and must not be used for training claims.

## DDP readiness

`reference/crystalite/src/train_crystalite.py` now uses Lightning Fabric while keeping
the existing manual training loop. Fabric supplies DDP gradient synchronization and a
distributed training sampler. Validation runs once on rank zero and is broadcast, so
its value does not change with world size; checkpoint/W&B/sampling side effects are
also rank-zero only. Split/cache creation is serialized before workers read it, and the
DDP collective timeout is extended for rank-zero sampling/evaluation.

The intended batch contract is **128 samples per rank**, hence global batch 1,024 on
eight GPUs. A one-step NCCL smoke completed on all eight RTX 3090s with exactly this
contract (`world_size=8 per_rank_batch=128 global_batch=1024`). A two-rank/two-step
smoke also passed epoch reshuffling, distributed validation, and rank-zero checkpoint
creation. Phase-1 policy still forbids starting the 8-DDP production run before the
full relaxed dataset and final validation exist.

The production launcher is `cont_task/pre_train/train_cuni_csp_8gpu.sh`. It
uses the proportionally scaled 152.3M CSP architecture, batch 64 per GPU (global 512), and
chunked sampling. Its hard gate refuses to start unless the Phase-1 validation
report exists, passes, and records all 376,200 converged structures; the gate's
negative-path test passes.

The repository's pre-existing `tests/test_edm_loss_normalization.py` expectations do
not match the current token-normalized loss implementation; both tests fail before
and independently of the Fabric path. This needs a separate behavior decision rather
than silently changing the training loss during DDP integration.

## Evidence

- `cont_task/pre_train/smoke_ltri_batch128_gpu0/`
- `cont_task/pre_train/smoke_ltri_d640_l20_batch128_gpu1/`
- `cont_task/pre_train/smoke_published_csp_ltri_batch128_gpu0/`
- `cont_task/pre_train/smoke_published_csp_ltri_batch128_expandable_gpu0/`
- `cont_task/post_train/bms/one_head.py`
- `cont_task/post_train/bms/test_one_head.py`
