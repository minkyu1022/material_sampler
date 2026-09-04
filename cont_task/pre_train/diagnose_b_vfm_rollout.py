#!/usr/bin/env python3
"""Compare teacher-forced and self-rollout endpoint predictions for VFM-B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.crystalite.vfm_utils import linear_interpolant, torus_delta
from src.data.mp20_tokens import MP20Tokens, VZ
from src.eval_crystalite_ckpt import _apply_ema_state_dict, _build_model_from_ckpt, _load_checkpoint
from src.models.lattice_repr import ltri_params_to_lattice_matrix, y1_to_lattice_latent
from src.models.type_encoding import build_type_encoding


def volume(ltri: torch.Tensor) -> torch.Tensor:
    return torch.exp(ltri[..., 0] + ltri[..., 2] + ltri[..., 5])


def summarize(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu().numpy()
    return {"mean": float(x.mean()), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def minimum_distance(frac: torch.Tensor, ltri: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    delta = frac[:, :, None] - frac[:, None, :]
    delta = torch.remainder(delta + 0.5, 1.0) - 0.5
    distance = torch.linalg.vector_norm(delta @ ltri_params_to_lattice_matrix(ltri)[:, None], dim=-1)
    valid = (~pad)[:, :, None] & (~pad)[:, None, :]
    valid &= ~torch.eye(frac.shape[1], device=frac.device, dtype=torch.bool)[None]
    return distance.masked_fill(~valid, torch.inf).amin((1, 2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = _load_checkpoint(args.checkpoint)
    model, cfg = _build_model_from_ckpt(ckpt=ckpt, device=device)
    if args.ema and ckpt.get("ema_state_dict") is not None:
        _apply_ema_state_dict(model, ckpt["ema_state_dict"])
    model.eval()
    ds = MP20Tokens(root=str(args.data_root), augment_translate=False, split="val", nmax=int(cfg["nmax"]))
    items = [ds[i] for i in range(min(args.batch_size, len(ds)))]
    clean_frac = torch.stack([x["F1"] for x in items]).to(device)
    clean_lattice = y1_to_lattice_latent(torch.stack([x["Y1"] for x in items]).to(device), str(cfg["lattice_repr"]))
    pad = torch.stack([x["pad_mask"] for x in items]).to(device).bool()
    a0 = torch.stack([x["A0"] for x in items]).to(device)
    encoding = build_type_encoding(str(ckpt.get("type_encoding", cfg.get("type_encoding", "atomic_number"))), vz=VZ)
    types = encoding.encode_from_A0(a0, pad)
    gen = torch.Generator(device=device).manual_seed(20260904)
    prior_frac = torch.remainder(torch.randn(clean_frac.shape, device=device, generator=gen), 1.0)
    prior_lattice = torch.randn(clean_lattice.shape, device=device, generator=gen)

    report: dict[str, object] = {
        "clean_volume": summarize(volume(clean_lattice)),
        "clean_minimum_distance_A": summarize(minimum_distance(clean_frac, clean_lattice, pad)),
        "teacher": {},
        "rollout": {},
    }
    with torch.inference_mode():
        for value in (0.0, 0.1, 0.5, 0.9, 0.99):
            t = torch.full((len(items),), value, device=device)
            frac_t, lat_t = linear_interpolant(prior_frac, clean_frac, prior_lattice, clean_lattice, t)
            pred = model(types, frac_t, lat_t, pad, t, lattice_bias_feats=lat_t, gem_sigma=1.0 - t)
            report["teacher"][str(value)] = {
                "input_volume": summarize(volume(lat_t)),
                "predicted_volume": summarize(volume(pred["lattice_vel"])),
                "lattice_endpoint_mae": float((pred["lattice_vel"] - clean_lattice).abs().mean()),
                "coordinate_endpoint_mae": float(torus_delta(pred["coord_vel"], clean_frac).abs()[~pad].mean()),
                "predicted_endpoint_minimum_distance_A": summarize(
                    minimum_distance(pred["coord_vel"], pred["lattice_vel"], pad)
                ),
            }

        x, lat = prior_frac.clone(), prior_lattice.clone()
        capture = {0, 1, 10, 50, 100, 150, args.steps - 2, args.steps - 1}
        for step in range(args.steps):
            value = step / args.steps
            t = torch.full((len(items),), value, device=device)
            pred = model(types, x, lat, pad, t, lattice_bias_feats=lat, gem_sigma=1.0 - t)
            if step in capture:
                report["rollout"][str(step)] = {
                    "t": value,
                    "state_volume": summarize(volume(lat)),
                    "predicted_endpoint_volume": summarize(volume(pred["lattice_vel"])),
                    "predicted_lattice_vs_clean_mae": float((pred["lattice_vel"] - clean_lattice).abs().mean()),
                    "state_minimum_distance_A": summarize(minimum_distance(x, lat, pad)),
                    "predicted_endpoint_minimum_distance_A": summarize(
                        minimum_distance(pred["coord_vel"], pred["lattice_vel"], pad)
                    ),
                }
            dt = 1.0 / args.steps
            remaining = max(1.0 - value, dt)
            x = torch.remainder(x + dt * torus_delta(pred["coord_vel"], x) / remaining, 1.0)
            lat = lat + dt * (pred["lattice_vel"] - lat) / remaining
        report["final_volume"] = summarize(volume(lat))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
