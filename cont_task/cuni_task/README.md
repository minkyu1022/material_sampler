# Cu–Ni continuous task

This directory is the eventual home of the existing Cu–Ni continuous-only work.

## Migration gate

Do **not** move the live directories yet. Migrate `cont_task/data`,
`cont_task/pre_train`, and `cont_task/post_train` only after both conditions hold:

1. A/B/C 200-epoch diagnostics and their evaluations have finished.
2. Cu–Ni Phase 1 relaxation, preprocessing, and full validation have finished with
   `validation_report.json: {"passed": true}`.

During migration, update recorded paths and launch scripts, then leave temporary
compatibility symlinks at the old paths until path validation passes.

