#!/usr/bin/env python3
"""Diagnose Cu--Ni hybrid path-weight variance at one condition."""

from __future__ import annotations

import argparse
import json
import runpy
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from janus_reproduce.cuni_train import _load_prior, _reference, _rollout
from janus_reproduce.free_energy import path_weight_estimates
from janus_reproduce.torch_eam import TorchCuNiEAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=750.0)
    parser.add_argument("--delta-mu", type=float, default=0.8525)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--scale-u", type=float, default=1.0)
    parser.add_argument("--scale-v", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device("cuda")
    load_checkpoint = runpy.run_path("scripts/evaluate_cuni.py")["load_checkpoint"]
    model, config = load_checkpoint(args.checkpoint, device)
    config = replace(
        config,
        diffusion_u=config.diffusion_u * args.scale_u,
        diffusion_v=config.diffusion_v * args.scale_v,
    )
    torch.manual_seed(config.seed + 750)
    result = _rollout(
        model,
        config,
        _load_prior(config),
        _reference(config.n_atoms, device),
        args.samples,
        device,
        path_weights=True,
        oracle=TorchCuNiEAM(config.potential).to(device),
        temperature=torch.tensor(args.temperature, device=device),
        delta_mu=torch.tensor(args.delta_mu, device=device),
        path_weight_trace=True,
    )
    signed = {
        "target_energy": result["log_target_energy"],
        "target_chemical": result["log_target_chemical"],
        "target_volume": result["log_target_volume"],
        "minus_prior_u": -result["log_prior_u"],
        "minus_prior_v": -result["log_prior_v"],
        "minus_log_q_discrete": -result["log_q_discrete"],
        "path_u": result["log_continuous_u"],
        "path_v": result["log_continuous_v"],
    }
    names = list(signed)
    matrix = torch.stack([signed[name].double() for name in names], 1).cpu().numpy()
    total = matrix.sum(1)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    total_variance = float(np.var(total, ddof=1))
    covariance_contribution = covariance.sum(1)
    no_u = result["log_weight"] - result["log_continuous_u"]
    _, _, no_u_ess = path_weight_estimates(no_u)
    steps = result["log_continuous_u_steps"].double().cpu().numpy()
    step_mean, step_std = steps.mean(0), steps.std(0, ddof=1)

    report = {
        "condition": {"temperature_K": args.temperature, "delta_mu_eV": args.delta_mu},
        "samples": args.samples,
        "diffusion": {
            "g_u_squared": config.diffusion_u**2 * args.temperature / config.diffusion_temperature_ref,
            "g_v_squared": config.diffusion_v**2 * args.temperature / config.diffusion_temperature_ref,
            "temperature_reference_K": config.diffusion_temperature_ref,
        },
        "ess": {
            "full": float(result["ess"]),
            "without_u_path": float(no_u_ess),
        },
        "total_log_weight_variance": total_variance,
        "without_u_path_log_weight_variance": float(no_u.double().var().item()),
        "terms": {
            name: {
                "mean": float(matrix[:, index].mean()),
                "std": float(matrix[:, index].std(ddof=1)),
                "variance": float(covariance[index, index]),
                "covariance_aware_variance_contribution": float(covariance_contribution[index]),
                "fraction_of_total_variance": float(covariance_contribution[index] / total_variance),
            }
            for index, name in enumerate(names)
        },
        "covariance_term_order": names,
        "covariance_matrix": covariance.tolist(),
        "u_path_steps": {
            "mean": step_mean.tolist(),
            "std": step_std.tolist(),
            "largest_abs_mean_step": int(np.argmax(abs(step_mean))),
            "largest_std_step": int(np.argmax(step_std)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "path_weight_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    time = (np.arange(config.steps) + 0.5) / config.steps
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(time, step_mean)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel(r"mean $\Delta\log W_u$")
    axes[1].plot(time, step_std)
    axes[1].set(xlabel="generation time t", ylabel=r"std $\Delta\log W_u$")
    fig.tight_layout()
    fig.savefig(args.output / "u_path_step_statistics.png", dpi=200)
    print(json.dumps({"ess": report["ess"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
