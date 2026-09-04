#!/usr/bin/env python3
"""Compare raw JANUS terminal and reference-MC distributions at one Cu--Ni state."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance

from janus_reproduce.cuni_reference_analysis import (
    load_chain,
    mean_lattice_displacements,
    state_thinning,
    thinned_config_indices,
)
from janus_reproduce.cuni_train import _load_prior, _reference, _rollout
from janus_reproduce.torch_eam import TorchCuNiEAM


def summary(janus: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {
        "janus_mean": float(janus.mean()), "janus_std": float(janus.std(ddof=1)),
        "reference_mean": float(reference.mean()), "reference_std": float(reference.std(ddof=1)),
        "mean_error": float(janus.mean() - reference.mean()),
        "wasserstein": float(wasserstein_distance(janus, reference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reference-chains", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=750.0)
    parser.add_argument("--delta-mu", type=float, default=0.85)
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    device = torch.device("cuda")
    load_checkpoint = runpy.run_path("scripts/evaluate_cuni.py")["load_checkpoint"]
    model, config = load_checkpoint(args.checkpoint, device)
    torch.manual_seed(config.seed + 751)
    reference_sites = _reference(config.n_atoms, device)
    states = _rollout(
        model, config, _load_prior(config), reference_sites, args.samples, device,
        temperature=torch.tensor(args.temperature, device=device),
        delta_mu=torch.tensor(args.delta_mu, device=device),
    )
    oracle = TorchCuNiEAM(config.potential).to(device)
    energies = []
    for start in range(0, args.samples, config.rollout_batch):
        stop = start + config.rollout_batch
        energies.append(oracle(
            states["species"][start:stop],
            reference_sites[None] + states["displacement"][start:stop],
            states["log_volume"][start:stop],
        ).float().cpu())
    janus = {
        "n_cu": states["species"].eq(0).sum(-1).cpu().numpy(),
        "energy_per_atom": torch.cat(energies).numpy() / config.n_atoms,
        "atomic_volume": (states["log_volume"].exp() / config.n_atoms).cpu().numpy(),
        "displacement": (
            states["displacement"].norm(dim=-1).mean(-1) * (states["log_volume"] / 3).exp()
        ).cpu().numpy(),
    }
    pattern = f"N108_T{args.temperature:07.2f}_mu0{args.delta_mu:.4f}_walker*.npz"
    chains = [load_chain(path) for path in sorted(args.reference_chains.glob(pattern))]
    if len(chains) != 2:
        raise SystemExit(f"expected two reference walkers matching {pattern}")
    reference = {
        "n_cu": np.concatenate([chain.n_cu[chain.production] for chain in chains]),
        "energy_per_atom": np.concatenate([chain.energy[chain.production] for chain in chains]) / 108,
        "atomic_volume": np.concatenate([chain.volume[chain.production] for chain in chains]) / 108,
        "displacement": np.concatenate([
            mean_lattice_displacements(chain, thinned_config_indices(chain, state_thinning(chains)))
            for chain in chains
        ]),
    }
    metrics = {key: summary(janus[key], reference[key]) for key in janus}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "terminal_distribution_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    labels = {
        "n_cu": r"$N_{Cu}$", "energy_per_atom": "U/N (eV/atom)",
        "atomic_volume": r"V/N ($\AA^3$/atom)", "displacement": r"mean |u| ($\AA$)",
    }
    for axis, key in zip(axes.flat[:4], labels, strict=True):
        axis.hist(reference[key], bins=40, density=True, histtype="step", color="#333333",
                  linestyle="--", linewidth=1.6, label="Reference")
        axis.hist(janus[key], bins=40, density=True, histtype="step", color="#D55E00",
                  linewidth=1.8, label="JANUS raw")
        axis.set_xlabel(labels[key])
    axes[0, 0].legend(frameon=False)
    axes[1, 1].hexbin(reference["n_cu"], reference["energy_per_atom"], gridsize=35,
                      mincnt=1, cmap="Greys")
    axes[1, 1].set(title="Reference", xlabel=r"$N_{Cu}$", ylabel="U/N (eV/atom)")
    axes[1, 2].hexbin(janus["n_cu"], janus["energy_per_atom"], gridsize=35,
                      mincnt=1, cmap="Oranges")
    axes[1, 2].set(title="JANUS raw", xlabel=r"$N_{Cu}$", ylabel="U/N (eV/atom)")
    fig.suptitle(rf"T={args.temperature:g} K, $\Delta\mu$={args.delta_mu:g} eV")
    fig.tight_layout()
    fig.savefig(args.output / "terminal_distribution_comparison.png", dpi=220)
    np.savez_compressed(args.output / "terminal_distributions.npz",
                        **{f"janus_{key}": value for key, value in janus.items()},
                        **{f"reference_{key}": value for key, value in reference.items()})
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
