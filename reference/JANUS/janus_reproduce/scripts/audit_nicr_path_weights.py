#!/usr/bin/env python3
"""Audit fixed-condition Ni--Cr JANUS path-weight variance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from janus_reproduce.alloy_model import AlloyPaiNN
from janus_reproduce.free_energy import path_weight_estimates
from janus_reproduce.nicr_train import (
    NiCrTrainConfig,
    _prior,
    _reference,
    effective_hamiltonian_manifest,
    resolved_config,
    rollout_nicr,
)
from janus_reproduce.torch_eam import TorchEAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=900.0)
    parser.add_argument("--target-cr", type=int, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-samples", action="store_true")
    args = parser.parse_args()
    config = NiCrTrainConfig.from_json(args.config)
    hamiltonian = effective_hamiltonian_manifest(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlloyPaiNN(
        features=config.features,
        layers=config.layers,
        radial_basis=config.radial_basis,
        cutoff=config.spec.graph_cutoff,
        temperature_reference=config.diffusion_temperature_ref,
        temperature_min=config.temperature_min,
        temperature_max=config.temperature_max,
        condition_intercept=0.0,
        condition_slope=0.0,
        condition_scale=1.0,
    ).to(device)
    checkpoint = torch.load(config.output / "checkpoint.pt", map_location=device, weights_only=False)
    if checkpoint.get("config") != resolved_config(config):
        raise ValueError("checkpoint config does not match the requested candidate config")
    if checkpoint.get("provenance", {}).get("effective_hamiltonian") != hamiltonian:
        raise ValueError("checkpoint Hamiltonian provenance mismatch")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(
        config.seed + 1 if args.seed is None else args.seed
    )
    oracle = TorchEAM(
        config.potential,
        species_indices=(0, 2),
        cutoff=config.target_cutoff,
    ).to(device)
    chunks = []
    for start in range(0, args.samples, args.batch_size):
        count = min(args.batch_size, args.samples - start)
        chunks.append(
            rollout_nicr(
                model,
                oracle,
                config,
                _prior(config),
                _reference(config, device),
                torch.full((count,), args.temperature, device=device),
                torch.full((count,), args.target_cr, dtype=torch.long, device=device),
                generator=generator,
                path_weights=True,
                path_diagnostics=True,
            )
        )
    result = {
        key: torch.cat([chunk[key].reshape(1) if chunk[key].ndim == 0 else chunk[key] for chunk in chunks])
        for key in chunks[0]
        if key not in {"log_xi", "normalized_weight", "ess"}
    }
    result["log_xi"], result["normalized_weight"], result["ess"] = path_weight_estimates(
        result["log_weight"]
    )
    components = torch.stack(
        (
            result["log_target"],
            -result["log_prior"],
            -result["log_q_discrete"],
            result["log_continuous_u"],
            result["log_continuous_v"],
        ),
        1,
    ).double()
    names = ("target", "negative_prior", "negative_discrete", "u_path", "v_path")
    covariance = torch.cov(components.T) if args.samples > 1 else torch.zeros(5, 5, device=device)
    counterfactual = {}
    for index, name in enumerate(names):
        without = components.sum(1) - components[:, index]
        counterfactual[f"without_{name}"] = float(path_weight_estimates(without)[2])
    payload = {
        "phase": config.phase,
        "temperature": args.temperature,
        "target_cr": args.target_cr,
        "samples": args.samples,
        "seed": config.seed + 1 if args.seed is None else args.seed,
        "ess": float(result["ess"]),
        "log_xi": float(result["log_xi"]),
        "std_log_weight": float(result["log_weight"].std(unbiased=False)),
        "components": {
            name: {
                "mean": float(components[:, index].mean()),
                "std": float(components[:, index].std(unbiased=False)),
            }
            for index, name in enumerate(names)
        },
        "covariance": covariance.cpu().tolist(),
        "counterfactual_ess": counterfactual,
        "u_step_mean": result["log_continuous_u_steps"].mean(0).cpu().tolist(),
        "u_step_std": result["log_continuous_u_steps"].std(0, unbiased=False).cpu().tolist(),
        "v_step_mean": result["log_continuous_v_steps"].mean(0).cpu().tolist(),
        "v_step_std": result["log_continuous_v_steps"].std(0, unbiased=False).cpu().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.save_samples:
        torch.save(
            {
                key: result[key].detach().cpu()
                for key in (
                    "species", "displacement", "log_volume", "energy", "log_weight",
                    "log_target", "log_prior", "log_q_discrete", "log_continuous_u",
                    "log_continuous_v", "normalized_weight",
                )
            },
            args.output.with_suffix(".pt"),
        )
    print(json.dumps({key: payload[key] for key in ("phase", "ess", "std_log_weight", "components", "counterfactual_ess")}))


if __name__ == "__main__":
    main()
