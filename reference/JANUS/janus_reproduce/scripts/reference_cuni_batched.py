#!/usr/bin/env python3
"""Run sharded GPU batches of the published N=108 Cu--Ni reference."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from janus_reproduce.cuni_reference_batched import (
    ReferenceConfig,
    output_path,
    reference_states_108,
    run_reference_batch,
)
from janus_reproduce.torch_eam import TorchCuNiEAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/cuni_reference_batched"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--total-sweeps", type=int, default=40_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--config-thin", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid batch/shard arguments")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --device cpu only for smoke tests")
    dtype = getattr(torch, args.dtype)
    oracle = TorchCuNiEAM(args.potential, dtype=dtype).to(device)
    config = ReferenceConfig(
        total_sweeps=args.total_sweeps,
        burn_in=args.burn_in,
        config_thin=args.config_thin,
        displacement_step=0.01 / math.sqrt(108),
        log_volume_step=0.03 / math.sqrt(108),
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
    )
    states = [
        state
        for state in reference_states_108()
        if state.index % args.num_shards == args.shard_index
        and not output_path(args.output, state).exists()
    ]
    for start in range(0, len(states), args.batch_size):
        batch = states[start : start + args.batch_size]
        paths = run_reference_batch(oracle, batch, args.output, config=config)
        print(f"completed {len(paths)} chains: {batch[0].index}..{batch[-1].index}", flush=True)


if __name__ == "__main__":
    main()
