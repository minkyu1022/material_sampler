#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/spml_minkyu_kim/joint_sampler
cd "$ROOT"
POT=reference/JANUS/janus_reproduce/potentials/cu_ni/Cu_Ni_Fischer_2018.eam.alloy
export CUDA_VISIBLE_DEVICES=0
START=$(date +%s)
xargs -P4 -I{} bash -c 'reference/JANUS/janus_reproduce/.venv/bin/python /tmp/bench_gpu_eam_relax.py "$1" "$2"' _ {} "$POT" < cont_task/data/gpu_parallel_benchmark/inputs.txt
END=$(date +%s)
printf 'wall_seconds=%s\n' "$((END-START))"
