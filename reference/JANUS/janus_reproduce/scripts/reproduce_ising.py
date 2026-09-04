#!/usr/bin/env python3
"""Reproduce the paper-condition conditional Ising experiment."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

from janus_reproduce.ising_experiment import (
    IsingExperimentConfig,
    JANUSIsing,
    evaluate_checkpoint,
    run_experiment,
    train_conditional,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ising")
    parser.add_argument("--checkpoint", help="evaluate this checkpoint without training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--damping", type=float, metavar="ETA", help="BMS-style logit damping"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--gradient-steps", type=int)
    parser.add_argument("--coexistence-fraction", type=float)
    parser.add_argument("--narrow-condition-fraction", type=float)
    parser.add_argument("--narrow-delta-mu-max", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--reference-samples", type=int)
    parser.add_argument("--reference-burn-in", type=int)
    parser.add_argument("--reference-chains", type=int)
    parser.add_argument("--reference-workers", type=int)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run a tiny end-to-end check")
    args = parser.parse_args()
    if args.checkpoint and (args.resume or args.smoke):
        parser.error("--checkpoint cannot be combined with --resume or --smoke")
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = IsingExperimentConfig(**saved["config"])
    else:
        config = IsingExperimentConfig(
            damping_eta=args.damping if args.damping is not None else 0.0,
            narrow_condition_fraction=args.narrow_condition_fraction or 0.0,
            narrow_delta_mu_max=args.narrow_delta_mu_max or 0.04,
            seed=args.seed if args.seed is not None else 0,
        )
    config = replace(
        config,
        **{
            key: value
            for key, value in {
                "rounds": args.rounds,
                "gradient_steps": args.gradient_steps,
                "coexistence_fraction": args.coexistence_fraction,
                "narrow_condition_fraction": args.narrow_condition_fraction,
                "narrow_delta_mu_max": args.narrow_delta_mu_max,
                "width": args.width,
                "depth": args.depth,
                "eval_samples": args.eval_samples,
                "reference_samples": args.reference_samples,
                "reference_burn_in": args.reference_burn_in,
                "reference_chains": args.reference_chains,
                "reference_workers": args.reference_workers,
                "damping_eta": args.damping,
                "seed": args.seed,
            }.items()
            if value is not None
        },
    )
    if args.smoke:
        config = IsingExperimentConfig(
            length=4,
            temperature_points=2,
            delta_mu_points=2,
            reveal_steps=8,
            rounds=1,
            batch_size=2,
            gradient_steps=1,
            eval_samples=2,
            reference_samples=2,
            reference_burn_in=1,
            reference_chains=1,
            reference_workers=args.reference_workers or 1,
            width=4,
            depth=1,
            damping_eta=args.damping if args.damping is not None else 0.0,
            narrow_condition_fraction=args.narrow_condition_fraction or 0.0,
            narrow_delta_mu_max=args.narrow_delta_mu_max or 0.04,
            seed=args.seed if args.seed is not None else 0,
        )
    context = nullcontext(None)
    if args.wandb:
        import wandb

        context = wandb.init(project="janus-reproduce", job_type="ising", config=vars(config))
    with context as run:
        if args.train_only:
            torch.manual_seed(config.seed)
            model = JANUSIsing(config.width, config.depth).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            losses = train_conditional(model, config, Path(args.output), resume=args.resume, run=run)
            result = {"loss": losses[-1]}
        elif args.checkpoint:
            result = evaluate_checkpoint(args.checkpoint, config, args.output, run=run)
        else:
            result = run_experiment(config, args.output, resume=args.resume, run=run)
    print(json.dumps({key: value for key, value in result.items() if key != "grid"}, indent=2))


if __name__ == "__main__":
    main()
