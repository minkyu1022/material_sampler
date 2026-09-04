#!/usr/bin/env python3
"""Train and validate the N=128 unified Ni--Cr BCT-2D sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from janus_reproduce.nicr_unified_bct2d import JANUSUnifiedBCT2D
from janus_reproduce.nicr_unified_train import (
    UnifiedTrainConfig,
    flow_matching_loss,
    rollout,
    target_scores,
)
from janus_reproduce.torch_eam import TorchEAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=1_000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    files = sorted(args.reference.glob("shard_*.pt"))
    if not files:
        raise FileNotFoundError(f"no reference shards in {args.reference}")
    shards = [torch.load(path, weights_only=True) for path in files]
    data = {key: torch.cat([shard[key] for shard in shards]) for key in shards[0]}
    data = {
        key: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else None)
        for key, value in data.items()
    }
    sampling_weight = None
    if args.weights:
        sampling_weight = torch.load(args.weights, weights_only=True).to(device=device).float()
        if sampling_weight.shape != (len(data["species"]),) or torch.any(sampling_weight < 0):
            raise ValueError("reference weights must be one nonnegative value per sample")
        sampling_weight /= sampling_weight.sum()
    oracle = TorchEAM(args.potential, species_indices=(0, 2)).to(device)
    model = JANUSUnifiedBCT2D(features=args.features, layers=args.layers).to(device)
    config = UnifiedTrainConfig()

    score_u, score_cell = [], []
    for start in range(0, len(data["species"]), args.batch):
        stop = start + args.batch
        _, u, cell = target_scores(
            oracle,
            data["species"][start:stop],
            data["disp_u"][start:stop],
            data["cell_z"][start:stop],
            data["temperature"][start:stop],
            normalization=model.normalization,
        )
        score_u.append(u.detach())
        score_cell.append(cell.detach())
    data["score_u"], data["score_cell"] = torch.cat(score_u), torch.cat(score_cell)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    for update in range(args.updates):
        learning_rate = args.learning_rate * min((update + 1) / max(args.warmup, 1), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        index = (
            torch.multinomial(sampling_weight, args.batch, replacement=True)
            if sampling_weight is not None
            else torch.randint(len(data["species"]), (args.batch,), device=device)
        )
        terminal = {key: data[key][index] for key in ("species", "disp_u", "cell_z", "score_u", "score_cell")}
        optimizer.zero_grad(set_to_none=True)
        loss, components = flow_matching_loss(
            model, oracle, terminal, data["temperature"][index], config
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        if update % 25 == 0 or update + 1 == args.updates:
            row = {
                "update": update + 1,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
            }
            row |= {key: float(value.detach()) for key, value in components.items()}
            history.append(row)
            print(json.dumps(row), flush=True)

    counts = torch.tensor([16, 32, 64, 96, 112] * 4, device=device)
    temperatures = torch.tensor([600, 900, 1050, 1200, 1500] * 4, device=device)
    generated = rollout(
        model,
        counts,
        temperatures,
        config,
        generator=torch.Generator(device=device).manual_seed(args.seed + 1),
    )
    path_metrics = []
    for index, (temperature, target_cr) in enumerate(((600.0, 16), (1050.0, 64), (1500.0, 112))):
        weighted = rollout(
            model,
            torch.full((32,), target_cr, device=device),
            torch.full((32,), temperature, device=device),
            config,
            generator=torch.Generator(device=device).manual_seed(args.seed + 10 + index),
            path_weights=True,
            oracle=oracle,
        )
        path_metrics.append(
            {
                "temperature": temperature,
                "target_cr": target_cr,
                "samples": 32,
                "ess": float(weighted["ess"].cpu()),
                "log_weight_std": float(
                    weighted["log_weight"][weighted["valid_domain"]].std(unbiased=False).cpu()
                ),
                "valid_fraction": float(weighted["valid_domain"].double().mean().cpu()),
            }
        )
    payload = {
        "status": "validation training; not final scientific production",
        "architecture": {"features": args.features, "layers": args.layers, "parameters": sum(p.numel() for p in model.parameters())},
        "updates": args.updates,
        "reference_samples": len(data["species"]),
        "history": history,
        "exact_composition": bool(torch.equal(generated["species"].eq(1).sum(1), counts)),
        "cell_ac_min": generated["cell_ac"].amin(0).cpu().tolist(),
        "cell_ac_max": generated["cell_ac"].amax(0).cpu().tolist(),
        "c_over_a": (generated["cell_ac"][:, 1] / generated["cell_ac"][:, 0]).cpu().tolist(),
        "rms_u": generated["disp_u"].square().sum(-1).mean(-1).sqrt().cpu().tolist(),
        "path_metrics": path_metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "config": vars(args), "report": payload}, args.output / "checkpoint.pt")


if __name__ == "__main__":
    main()
