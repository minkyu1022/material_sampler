#!/usr/bin/env python3
"""Train the minimal JANUS Ising head and compare it with ghost-Wolff samples."""

from __future__ import annotations

import argparse
import json

import torch

from janus_reproduce.ising import (
    JANUSIsing,
    ghost_wolff_samples,
    observables,
    sample_janus,
    train_fixed_point,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.269)
    parser.add_argument("--delta-mu", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JANUSIsing().to(device)
    losses = train_fixed_point(
        model,
        length=args.length,
        temperature=args.temperature,
        delta_mu=args.delta_mu,
        rounds=args.rounds,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
    )
    generated = sample_janus(
        model, args.samples, args.length, args.temperature, args.delta_mu, device=device
    )
    reference = ghost_wolff_samples(
        args.length,
        args.temperature,
        args.delta_mu,
        num_samples=args.samples,
        burn_in=max(100, args.length * args.length),
        seed=args.seed,
    )
    result = {
        "loss": losses[-1],
        "janus": observables(generated, delta_mu=args.delta_mu),
        "wolff": observables(reference, delta_mu=args.delta_mu),
    }
    print(json.dumps(result, indent=2))
    if args.wandb:
        import wandb

        with wandb.init(project="janus-reproduce", config=vars(args)) as run:
            run.log(
                {"train/loss": losses[-1], **{f"janus/{k}": v for k, v in result["janus"].items()}}
            )


if __name__ == "__main__":
    main()
