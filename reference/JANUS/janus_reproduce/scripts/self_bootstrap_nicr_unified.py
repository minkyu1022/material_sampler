#!/usr/bin/env python3
"""Reference-free self-bootstrap training for the unified Ni--Cr sampler."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import torch

from janus_reproduce.free_energy import include_zero_weight_attempts
from janus_reproduce.nicr_unified_bct2d import (
    DEFAULT_BCT_DOMAIN,
    JANUSUnifiedBCT2D,
    cell_matrix,
    reference_sites,
)
from janus_reproduce.nicr_unified_train import (
    UnifiedTrainConfig,
    flow_matching_loss,
    rollout,
    target_scores,
)
from janus_reproduce.torch_eam import TorchEAM


def _q_bain(result):
    ratio = result["cell_ac"][:, 1] / result["cell_ac"][:, 0]
    return (ratio - 1) / (math.sqrt(2) - 1)


def _basin(q_bain):
    return torch.where(q_bain < 0.25, 0, torch.where(q_bain > 0.75, 2, 1))


def _site_anchoring_ratio(disp_u, cell_ac):
    reference = reference_sites(dtype=disp_u.dtype).to(disp_u.device)
    delta = reference[None, :, None] + disp_u[:, :, None] - reference[None, None, :]
    delta -= delta.round()
    distance = torch.einsum("bnmi,bij->bnmj", delta, cell_matrix(cell_ac)).square().sum(-1)
    expected = torch.arange(len(reference), device=disp_u.device)
    return float(distance.argmin(-1).eq(expected).double().mean())


def _scores(oracle, model, sample, batch):
    score_u, score_cell = [], []
    for start in range(0, len(sample["species"]), batch):
        stop = start + batch
        _, u, cell = target_scores(
            oracle,
            sample["species"][start:stop],
            sample["disp_u"][start:stop],
            sample["cell_z"][start:stop],
            sample["temperature"][start:stop],
            normalization=model.normalization,
        )
        score_u.append(u.detach())
        score_cell.append(cell.detach())
    sample["score_u"], sample["score_cell"] = torch.cat(score_u), torch.cat(score_cell)


def _probability(log_weight):
    probability = (log_weight.double() - torch.logsumexp(log_weight.double(), 0)).exp()
    return 0.8 * probability + 0.2 / len(probability)


def _unique_replay_indices(result, count, generator):
    valid = result["valid_domain"]
    basin = _basin(_q_bain(result))
    chosen = []
    quota = max(1, count // 3)
    for label in range(3):
        sites = torch.where(valid & basin.eq(label))[0]
        take = min(quota, len(sites))
        if take:
            chosen.append(
                sites[torch.multinomial(_probability(result["log_weight"][sites]), take, False, generator=generator)]
            )
    selected = torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long, device=valid.device)
    candidate = torch.arange(len(valid), device=valid.device)
    remaining = torch.where(valid & ~torch.isin(candidate, selected))[0]
    take = min(count - len(selected), len(remaining))
    if take:
        selected = torch.cat(
            (
                selected,
                remaining[
                    torch.multinomial(
                        _probability(result["log_weight"][remaining]), take, False, generator=generator
                    )
                ],
            )
        )
    if not len(selected):
        raise FloatingPointError("no valid unique self-bootstrap sample")
    return selected


def _rollout_with_retries(model, oracle, config, temperature, target_cr, count, generator):
    attempted = 0
    for _ in range(4):
        result = rollout(
            model,
            torch.full((count,), target_cr, device=next(model.parameters()).device),
            torch.full((count,), temperature, device=next(model.parameters()).device),
            config,
            generator=generator,
            path_weights=True,
            oracle=oracle,
        )
        attempted += count
        if result["valid_domain"].any():
            result["log_xi"] = include_zero_weight_attempts(result["log_xi"], count, attempted)
            return result, attempted
    raise FloatingPointError(f"four rollout batches were invalid at T={temperature}, n={target_cr}")


def _metrics(result, temperature, target_cr, attempted, oracle_nfe):
    valid = result["valid_domain"]
    if not valid.any():
        return {
            "temperature": temperature,
            "target_cr": target_cr,
            "ess": 0.0,
            "std_log_weight": float("inf"),
            "valid_fraction": 0.0,
        }
    q_bain = _q_bain(result)[valid]
    basin = _basin(q_bain)
    weight = result["normalized_weight"][valid]
    weight /= weight.sum()
    rms = result["disp_u"][valid].square().sum(-1).mean(-1).sqrt()
    energy = result["energy"][valid] / 128
    volume = 0.5 * result["cell_ac"][valid, 0].square() * result["cell_ac"][valid, 1]
    component_names = (
        "log_target",
        "log_prior_u",
        "log_prior_cell",
        "log_q_species",
        "log_continuous_u",
        "log_continuous_cell",
    )
    signed = torch.stack(
        (
            result["log_target"][valid],
            -result["log_prior_u"][valid],
            -result["log_prior_cell"][valid],
            -result["log_q_species"][valid],
            result["log_continuous_u"][valid],
            result["log_continuous_cell"][valid],
        ),
        1,
    ).double()
    centered = signed - signed.mean(0)
    covariance = centered.T @ centered / len(signed)
    counterfactual_ess = {}
    total = signed.sum(1)
    for index, name in enumerate(component_names):
        log_weight = total - signed[:, index]
        normalized = (log_weight - torch.logsumexp(log_weight, 0)).exp()
        counterfactual_ess[name] = float(normalized.square().sum().reciprocal())
    log_weight = total - signed[:, -2:].sum(1)
    normalized = (log_weight - torch.logsumexp(log_weight, 0)).exp()
    counterfactual_ess["continuous_paths"] = float(normalized.square().sum().reciprocal())
    return {
        "temperature": temperature,
        "target_cr": target_cr,
        "ess": float(result["ess"]),
        "log_xi": float(result["log_xi"]),
        "std_log_weight": float(result["log_weight"][valid].std(unbiased=False)),
        "log_weight_components": {
            name: {"mean": float(values.mean()), "std": float(values.std(unbiased=False))}
            for name, values in zip(component_names, signed.T, strict=True)
        },
        "log_weight_component_order": component_names,
        "log_weight_component_covariance": covariance.cpu().tolist(),
        "counterfactual_ess_without": counterfactual_ess,
        "valid_fraction": float(valid.sum() / attempted),
        "exact_count": bool(result["species"].eq(1).sum(1).eq(target_cr).all()),
        "bcc_fraction": float(basin.eq(0).double().mean()),
        "intermediate_fraction": float(basin.eq(1).double().mean()),
        "fcc_fraction": float(basin.eq(2).double().mean()),
        "weighted_bcc_fraction": float(weight[basin.eq(0)].sum()),
        "weighted_intermediate_fraction": float(weight[basin.eq(1)].sum()),
        "weighted_fcc_fraction": float(weight[basin.eq(2)].sum()),
        "q_bain_histogram": torch.histogram(q_bain.cpu(), bins=20, range=(-0.25, 1.25)).hist.tolist(),
        "q_bain_min": float(q_bain.min()),
        "q_bain_max": float(q_bain.max()),
        "rms_u": float(rms.mean()),
        "site_anchoring_ratio": _site_anchoring_ratio(result["disp_u"], result["cell_ac"]),
        "energy_per_atom_mean": float(energy.mean()),
        "energy_per_atom_std": float(energy.std(unbiased=False)),
        "volume_per_atom_mean": float(volume.mean()),
        "volume_per_atom_std": float(volume.std(unbiased=False)),
        "cell_ac_mean": result["cell_ac"][valid].mean(0).cpu().tolist(),
        "cell_ac_std": result["cell_ac"][valid].std(0, unbiased=False).cpu().tolist(),
        "oracle_nfe_per_effective_sample": oracle_nfe / max(float(result["ess"]), 1e-12),
    }


def _training_indices(data, batch, generator):
    chosen = []
    quota = max(1, batch // 3)
    for label in range(3):
        sites = torch.where(data["basin"].eq(label))[0]
        take = min(quota, len(sites))
        if take:
            probability = _probability(data["log_weight"][sites])
            chosen.append(sites[torch.multinomial(probability, take, len(sites) < take, generator=generator)])
    index = torch.cat(chosen)
    if len(index) < batch:
        extra = torch.randint(len(data["species"]), (batch - len(index),), device=index.device, generator=generator)
        index = torch.cat((index, extra))
    return index[:batch]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outer-rounds", type=int, default=12)
    parser.add_argument("--updates-per-round", type=int, default=250)
    parser.add_argument("--rollout-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--replay-per-condition", type=int, default=12)
    parser.add_argument("--replay-cap", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-updates", type=int, default=5_000)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=100.0)
    parser.add_argument("--target-score-u-clip", type=float, default=100.0)
    parser.add_argument("--target-score-cell-clip", type=float, default=1_000.0)
    parser.add_argument("--rollout-velocity-cell-clip", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temperatures", type=float, nargs="+", default=(600.0, 1050.0, 1500.0))
    parser.add_argument("--target-cr-values", type=int, nargs="+", default=(32, 64, 96, 112))
    parser.add_argument("--cell-prior-scale", type=float, default=1.0)
    parser.add_argument("--cell-prior-mixture", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-model", type=Path)
    parser.add_argument("--reset-replay", action="store_true")
    args = parser.parse_args()
    if args.resume and args.initialize_model:
        parser.error("--resume and --initialize-model are mutually exclusive")
    conditions = tuple(product(args.temperatures, args.target_cr_values))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    oracle = TorchEAM(args.potential, species_indices=(0, 2)).to(device)
    model = JANUSUnifiedBCT2D(features=args.features, layers=args.layers).to(device)
    config = replace(
        UnifiedTrainConfig(),
        steps=args.steps,
        cell_prior_scale=args.cell_prior_scale,
        cell_prior_mixture=args.cell_prior_mixture,
        gradient_clip_norm=args.gradient_clip_norm,
        target_score_u_clip=args.target_score_u_clip,
        target_score_cell_clip=args.target_score_cell_clip,
        rollout_velocity_cell_clip=args.rollout_velocity_cell_clip,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    keys = (
        "species", "disp_u", "cell_z", "temperature", "score_u", "score_cell", "log_weight", "basin"
    )
    replay = {key: [] for key in keys}
    start_outer = 0
    if args.initialize_model:
        checkpoint = torch.load(args.initialize_model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("config") != asdict(config):
            raise ValueError("resume checkpoint uses a different training/prior configuration")
        saved_args = checkpoint.get("args", {})
        for name in (
            "updates_per_round",
            "learning_rate",
            "minimum_learning_rate",
            "warmup_updates",
            "weight_decay",
        ):
            if saved_args.get(name) != getattr(args, name):
                raise ValueError(f"resume checkpoint uses a different {name}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if not args.reset_replay:
            replay = {key: [value.detach().cpu()] for key, value in checkpoint["replay"].items()}
        generator.set_state(checkpoint["generator_state"].cpu())
        torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())
        start_outer = checkpoint["outer_round"]

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    for outer in range(start_outer, args.outer_rounds):
        round_started = time.perf_counter()
        oracle_calls = 0
        for condition_index, (temperature, target_cr) in enumerate(conditions):
            result, attempted = _rollout_with_retries(
                model, oracle, config, temperature, target_cr, args.rollout_samples, generator
            )
            oracle_calls += attempted
            index = _unique_replay_indices(result, args.replay_per_condition, generator)
            sample = {
                "species": result["species"][index],
                "disp_u": result["disp_u"][index],
                "cell_z": result["cell_z"][index],
                "temperature": result["temperature"][index],
                "log_weight": result["log_weight"][index],
                "basin": _basin(_q_bain(result))[index],
            }
            _scores(oracle, model, sample, config.oracle_batch)
            oracle_calls += len(index)
            for key, values in replay.items():
                values.append(sample[key].detach().cpu())
            with (args.output / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps({"outer_round": outer + 1, "condition": condition_index + 1}) + "\n")

        data = {key: torch.cat(value)[-args.replay_cap :].to(device) for key, value in replay.items()}
        replay = {key: [value.detach().cpu()] for key, value in data.items()}
        training_started = time.perf_counter()
        loss_sums = {"total": 0.0}
        loss_maxima = {"total": 0.0}
        gradient_norm_sum = gradient_norm_max = 0.0
        total_updates = args.outer_rounds * args.updates_per_round
        for update_index in range(args.updates_per_round):
            completed_updates = outer * args.updates_per_round + update_index
            if completed_updates < args.warmup_updates:
                learning_rate = args.learning_rate * (completed_updates + 1) / max(
                    args.warmup_updates, 1
                )
            else:
                progress = (completed_updates - args.warmup_updates) / max(
                    total_updates - args.warmup_updates - 1, 1
                )
                learning_rate = args.minimum_learning_rate + 0.5 * (
                    args.learning_rate - args.minimum_learning_rate
                ) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            index = _training_indices(data, args.batch, generator)
            terminal = {
                key: data[key][index]
                for key in ("species", "disp_u", "cell_z", "score_u", "score_cell")
            }
            optimizer.zero_grad(set_to_none=True)
            loss, components = flow_matching_loss(
                model, oracle, terminal, data["temperature"][index], config
            )
            if not torch.isfinite(loss):
                values = {name: float(value.detach()) for name, value in components.items()}
                raise FloatingPointError(
                    f"non-finite loss in outer round {outer + 1}: {values}"
                )
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            )
            optimizer.step()
            loss_sums["total"] += float(loss.detach())
            loss_maxima["total"] = max(loss_maxima["total"], float(loss.detach()))
            gradient_norm_sum += gradient_norm
            gradient_norm_max = max(gradient_norm_max, gradient_norm)
            for name, value in components.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach())
                loss_maxima[name] = max(loss_maxima.get(name, 0.0), float(value.detach()))
            if (update_index + 1) % 250 == 0:
                with (args.output / "progress.jsonl").open("a") as handle:
                    handle.write(json.dumps({
                        "outer_round": outer + 1,
                        "phase": "training",
                        "update": update_index + 1,
                    }) + "\n")
        training_seconds = time.perf_counter() - training_started

        evaluation_started = time.perf_counter()
        condition_metrics = []
        for condition_index, (temperature, target_cr) in enumerate(conditions):
            result = rollout(
                model,
                torch.full((args.eval_samples,), target_cr, device=device),
                torch.full((args.eval_samples,), temperature, device=device),
                config,
                generator=generator,
                path_weights=True,
                oracle=oracle,
            )
            oracle_calls += args.eval_samples
            condition_metrics.append(
                _metrics(result, temperature, target_cr, args.eval_samples, args.eval_samples)
            )
            with (args.output / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps({
                    "outer_round": outer + 1,
                    "phase": "evaluation",
                    "condition": condition_index + 1,
                }) + "\n")
        evaluation_seconds = time.perf_counter() - evaluation_started
        row = {
            "pid": os.getpid(),
            "outer_round": outer + 1,
            "replay_size": len(data["species"]),
            "replay_unique": len(torch.unique(data["disp_u"].reshape(len(data["disp_u"]), -1), dim=0)),
            "loss": loss_sums["total"] / args.updates_per_round,
            "loss_components": {
                name: value / args.updates_per_round
                for name, value in loss_sums.items()
                if name != "total"
            },
            "loss_maxima": loss_maxima,
            "gradient_norm_preclip_mean": gradient_norm_sum / args.updates_per_round,
            "gradient_norm_preclip_max": gradient_norm_max,
            "learning_rate": learning_rate,
            "wall_seconds": time.perf_counter() - round_started,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "oracle_nfe": oracle_calls,
            "conditions": condition_metrics,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "outer_round": outer + 1,
                "config": asdict(config),
                "normalization": asdict(model.normalization),
                "domain": asdict(DEFAULT_BCT_DOMAIN),
                "args": vars(args),
                "metrics": row,
                "replay": {key: value[0] for key, value in replay.items()},
                "generator_state": generator.get_state().cpu(),
                "cpu_rng_state": torch.get_rng_state(),
            },
            args.output / "checkpoint.pt",
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
