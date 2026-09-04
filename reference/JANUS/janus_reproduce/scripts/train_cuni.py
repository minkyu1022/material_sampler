#!/usr/bin/env python3
"""Train the paper-scale Cu--Ni JANUS model, optionally under torchrun."""

from __future__ import annotations

import argparse
from pathlib import Path

from janus_reproduce.cuni_train import CuNiTrainConfig, train_cuni


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--potential",
        type=Path,
        default=Path("potentials/cu_ni/Cu_Ni_Fischer_2018.eam.alloy"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/cuni_train"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fitted-u-exponent", action="store_true")
    parser.add_argument("--sigma-v-scale", type=float, default=1.0)
    parser.add_argument("--diffusion-u", type=float, default=0.0)
    parser.add_argument("--diffusion-v", type=float, default=0.0)
    parser.add_argument("--diffusion-temperature-ref", type=float, default=900.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--continuous-loss", choices=("tsm",), default="tsm")
    parser.add_argument("--discrete-loss", choices=("sce",), default="sce")
    parser.add_argument("--discrete-sampler", choices=("janus_tau_leap",), default="janus_tau_leap")
    parser.add_argument("--target-score-u-clip", type=float, default=100.0)
    parser.add_argument("--target-score-v-clip", type=float, default=1_000.0)
    parser.add_argument("--rollout-velocity-clip", type=float, default=0.1)
    parser.add_argument("--rollout-score-clip", type=float, default=1_000.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--minimum-learning-rate", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    common = {
        "prior": args.prior,
        "bf16": args.bf16,
        "use_fitted_sigma_u_exponent": args.fitted_u_exponent,
        "sigma_v_scale": args.sigma_v_scale,
        "diffusion_u": args.diffusion_u,
        "diffusion_v": args.diffusion_v,
        "diffusion_temperature_ref": args.diffusion_temperature_ref,
        "learning_rate": args.learning_rate,
        "continuous_loss": args.continuous_loss,
        "discrete_loss": args.discrete_loss,
        "discrete_sampler": args.discrete_sampler,
        "target_score_u_clip": args.target_score_u_clip,
        "target_score_v_clip": args.target_score_v_clip,
        "rollout_velocity_clip": args.rollout_velocity_clip,
        "rollout_score_clip": args.rollout_score_clip,
        "gradient_clip_norm": args.gradient_clip_norm,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "warmup_updates": args.warmup_updates,
        "minimum_learning_rate": args.minimum_learning_rate,
        "resume": not args.no_resume,
        "wandb_project": None if args.no_wandb else "janus-reproduce",
    }
    config = (
        CuNiTrainConfig.smoke(args.potential, args.output, **common)
        if args.smoke
        else CuNiTrainConfig(args.potential, args.output, **common)
    )
    train_cuni(config)


if __name__ == "__main__":
    main()
