#!/usr/bin/env python3
"""Short end-to-end validation of the unified Ni--Cr BCT-2D pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from janus_reproduce.nicr_unified_bct2d import JANUSUnifiedBCT2D
from janus_reproduce.nicr_unified_reference import ReferenceMCConfig, reference_mc
from janus_reproduce.nicr_unified_train import UnifiedTrainConfig, flow_matching_loss, rollout
from janus_reproduce.torch_eam import TorchEAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(2026)

    cpu_oracle = TorchEAM(args.potential, species_indices=(0, 2))
    chains = [
        reference_mc(
            cpu_oracle,
            64,
            1050.0,
            ReferenceMCConfig(sweeps=4, burn_in=2, thin=1, species_moves=2),
            initial_cell_ac=torch.tensor(cell),
            generator=torch.Generator().manual_seed(2026 + index),
        )
        for index, cell in enumerate(((2.765, 2.765), (2.462, 3.482)))
    ]
    reference = {
        key: torch.cat([chain[key] for chain in chains])
        for key in ("species", "disp_u", "cell_z", "cell_ac", "log_density")
    }
    device = torch.device(args.device)
    model = JANUSUnifiedBCT2D(features=8, layers=1, radial_basis=4).to(device)
    oracle = TorchEAM(args.potential, species_indices=(0, 2)).to(device)
    terminal = {
        key: reference[key].to(device=device, dtype=torch.float32 if key != "species" else None)
        for key in ("species", "disp_u", "cell_z")
    }
    temperature = torch.full((len(terminal["species"]),), 1050.0, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses, component_history = [], []
    for _ in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        loss, components = flow_matching_loss(model, oracle, terminal, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        component_history.append({key: float(value.detach()) for key, value in components.items()})
    config = UnifiedTrainConfig(steps=args.steps)
    generated = rollout(
        model,
        torch.tensor([32, 64, 96], device=device),
        torch.tensor([700.0, 1050.0, 1400.0], device=device),
        config,
        generator=torch.Generator(device=device).manual_seed(2027),
    )
    weighted = rollout(
        model,
        torch.full((32,), 64, device=device),
        torch.full((32,), 1050.0, device=device),
        config,
        generator=torch.Generator(device=device).manual_seed(2028),
        path_weights=True,
        oracle=oracle,
    )
    payload = {
        "status": "smoke validation; not a scientific result",
        "device": str(device),
        "updates": args.updates,
        "steps": args.steps,
        "cell_normalization": asdict(model.normalization),
        "train_config": asdict(config),
        "losses": losses,
        "loss_components": component_history,
        "reference_acceptance": [chain["stats"] for chain in chains],
        "reference_cell_ac": reference["cell_ac"].tolist(),
        "generated_cr_counts": generated["species"].eq(1).sum(1).cpu().tolist(),
        "generated_cell_ac": generated["cell_ac"].cpu().tolist(),
        "generated_rms_u": generated["disp_u"].square().sum(-1).mean(-1).sqrt().cpu().tolist(),
        "path_state": {"temperature": 1050.0, "target_cr": 64, "samples": 32},
        "path_ess": float(weighted["ess"].cpu()),
        "log_weight_std": float(
            weighted["log_weight"][weighted["valid_domain"]].std(unbiased=False).cpu()
        ),
        "path_valid_fraction": float(weighted["valid_domain"].double().mean().cpu()),
        "finite": all(
            torch.isfinite(value).all().item()
            for value in (
                generated["disp_u"],
                generated["cell_z"],
                weighted["log_weight"][weighted["valid_domain"]],
            )
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    torch.save(
        {
            "model": model.state_dict(),
            "report": payload,
            "cell_normalization": asdict(model.normalization),
            "train_config": asdict(config),
        },
        args.output / "checkpoint.pt",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
