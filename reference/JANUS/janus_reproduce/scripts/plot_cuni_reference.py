#!/usr/bin/env python3
"""Create the N=108 Cu--Ni reference figures from completed two-walker chains."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from janus_reproduce.cuni_reference_analysis import (
    averaged_partial_rdf,
    json_safe,
    load_chain,
    mean_lattice_displacements,
    mixing_free_energy_from_semigrand,
    solve_semigrand_mbar,
    state_thinning,
    thinned_config_indices,
    thinned_scalar_indices,
    two_walker_diagnostics,
)

TARGET_CURVE_T = (600.0, 900.0, 1200.0)
TARGET_RDF_T = 800.0
TARGET_RDF_X = (0.23, 0.52, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="directory containing per-chain NPZ files")
    parser.add_argument("--output", type=Path, default=Path("outputs/cuni_reference_figures"))
    parser.add_argument("--rhat-limit", type=float, default=1.05)
    parser.add_argument("--minimum-ess", type=float, default=100.0)
    return parser.parse_args()


def _identity(path: Path) -> tuple[float, float, int]:
    with np.load(path, allow_pickle=False) as data:
        raw = data["metadata"].item()
    if isinstance(raw, bytes):
        raw = raw.decode()
    metadata = json.loads(str(raw))
    return (
        float(metadata["temperature_K"]),
        float(metadata["delta_mu_Cu_minus_Ni_eV"]),
        int(metadata["walker"]),
    )


def _save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    paths = sorted(args.input.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no NPZ chains found in {args.input}")
    grouped: dict[tuple[float, float], dict[int, Path]] = defaultdict(dict)
    for path in paths:
        temperature, mu, walker = _identity(path)
        if walker in grouped[(temperature, mu)]:
            raise SystemExit(f"duplicate walker {walker} at T={temperature}, mu={mu}")
        grouped[(temperature, mu)][walker] = path
    incomplete = {state: walkers.keys() for state, walkers in grouped.items() if set(walkers) != {0, 1}}
    if incomplete:
        raise SystemExit(f"two independent walkers are required at every state; incomplete={incomplete}")

    args.output.mkdir(parents=True, exist_ok=True)
    states = []
    diagnostics = {}
    histograms = {}
    n_atoms_seen = set()
    for (temperature, mu), walker_paths in sorted(grouped.items()):
        chains = [load_chain(walker_paths[walker]) for walker in (0, 1)]
        n_atoms_seen.update(chain.n_atoms for chain in chains)
        interval = state_thinning(chains)
        diagnostic = two_walker_diagnostics(chains)
        key = f"T={temperature:g},mu={mu:g}"
        min_ess = min(
            diagnostic["metrics"][metric]["effective_samples"][walker]
            for metric in ("composition", "energy", "volume")
            for walker in (0, 1)
        )
        max_rhat = max(
            diagnostic["metrics"][metric]["split_rhat"]
            for metric in ("composition", "energy", "volume")
        )
        diagnostic["reconstruction_pass"] = bool(
            np.isfinite(max_rhat) and max_rhat <= args.rhat_limit and min_ess >= args.minimum_ess
        )
        diagnostics[key] = diagnostic

        raw_x = np.mean(
            [chain.n_cu[chain.production].mean() / chain.n_atoms for chain in chains]
        )
        config_volume, config_displacement = [], []
        state_hist = np.zeros(chains[0].n_atoms + 1, dtype=np.int64)
        for chain in chains:
            scalar_indices = thinned_scalar_indices(chain, interval)
            state_hist += np.bincount(
                chain.n_cu[scalar_indices], minlength=chain.n_atoms + 1
            )
            config_indices = thinned_config_indices(chain, interval)
            if len(config_indices):
                sweeps = chain.config_sweeps[config_indices]
                config_volume.extend((chain.volume[sweeps] / chain.n_atoms).tolist())
                config_displacement.extend(
                    mean_lattice_displacements(chain, config_indices).tolist()
                )
        if not config_volume:
            raise SystemExit(f"no retained production configurations at {key}")
        histograms[(temperature, mu)] = state_hist
        states.append(
            {
                "temperature": temperature,
                "delta_mu": mu,
                "composition": float(raw_x),
                "atomic_volume": float(np.mean(config_volume)),
                "mean_displacement": float(np.mean(config_displacement)),
                "thinning_sweeps": interval,
            }
        )
    if n_atoms_seen != {108}:
        raise SystemExit(f"this figure reconstruction requires N=108, found {n_atoms_seen}")

    temperatures = np.array(sorted({state["temperature"] for state in states}))
    mus = np.array(sorted({state["delta_mu"] for state in states}))
    if len(states) != len(temperatures) * len(mus):
        raise SystemExit("the raw T/delta-mu grid is not rectangular")
    by_key = {(state["temperature"], state["delta_mu"]): state for state in states}
    heatmap = np.array(
        [[by_key[(temperature, mu)]["composition"] for mu in mus] for temperature in temperatures]
    )
    displacement_grid = np.array(
        [[by_key[(temperature, mu)]["mean_displacement"] for mu in mus] for temperature in temperatures]
    )
    volume_grid = np.array(
        [[by_key[(temperature, mu)]["atomic_volume"] for mu in mus] for temperature in temperatures]
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.pcolormesh(mus, temperatures, heatmap, shading="nearest", vmin=0, vmax=1)
    ax.set(xlabel=r"$\Delta\mu_{Cu-Ni}$ (eV)", ylabel="Temperature (K)")
    fig.colorbar(image, ax=ax, label=r"raw $\langle x_{Cu}\rangle$")
    _save_figure(fig, args.output / "composition_heatmap_raw.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    resolved_curve_t = []
    for target in TARGET_CURVE_T:
        temperature = float(temperatures[np.argmin(abs(temperatures - target))])
        resolved_curve_t.append(temperature)
        selected = sorted(
            (state for state in states if state["temperature"] == temperature),
            key=lambda state: state["composition"],
        )
        x = [state["composition"] for state in selected]
        axes[0].plot(x, [state["mean_displacement"] for state in selected], ".-", label=f"{temperature:g} K")
        axes[1].plot(x, [state["atomic_volume"] for state in selected], ".-", label=f"{temperature:g} K")
    axes[0].set(xlabel=r"raw $\langle x_{Cu}\rangle$", ylabel="mean displacement (Å)")
    axes[1].set(xlabel=r"raw $\langle x_{Cu}\rangle$", ylabel=r"$V/N$ (Å$^3$/atom)")
    axes[0].legend()
    _save_figure(fig, args.output / "displacement_and_atomic_volume.png")

    rdf_temperature = float(temperatures[np.argmin(abs(temperatures - TARGET_RDF_T))])
    rdf_candidates = [state for state in states if state["temperature"] == rdf_temperature]
    rdf_selection = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for axis, target_x in zip(axes, TARGET_RDF_X):
        state = min(rdf_candidates, key=lambda item: abs(item["composition"] - target_x))
        chain_paths = grouped[(rdf_temperature, state["delta_mu"])]
        chains = [load_chain(chain_paths[walker]) for walker in (0, 1)]
        radius, rdf = averaged_partial_rdf(chains, state["thinning_sweeps"], r_max=5.3, dr=0.02)
        for name, curve in rdf.items():
            axis.plot(radius, curve, label=name)
        axis.set(
            xlabel="r (Å)",
            title=rf"$x_{{Cu}}={state['composition']:.3f}$ ($\Delta\mu={state['delta_mu']:.3f}$ eV)",
        )
        rdf_selection.append(
            {"target_composition": target_x, "state": state, "temperature": rdf_temperature}
        )
    axes[0].set_ylabel("partial g(r)")
    axes[-1].legend()
    _save_figure(fig, args.output / "partial_rdf_T800.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    mbar_diagnostics = {}
    free_energy_curves = {}
    for target in TARGET_CURVE_T:
        temperature = float(temperatures[np.argmin(abs(temperatures - target))])
        state_mus = np.array(sorted(mu for t, mu in histograms if t == temperature))
        hist = np.stack([histograms[(temperature, mu)] for mu in state_mus])
        f, mbar_info = solve_semigrand_mbar(hist, state_mus, temperature)
        x, g_mix = mixing_free_energy_from_semigrand(f, state_mus, temperature, 108)
        ax.plot(x, 1000 * g_mix, label=f"{temperature:g} K")
        mbar_diagnostics[f"T={temperature:g}"] = mbar_info
        free_energy_curves[f"T={temperature:g}"] = {"x": x.tolist(), "Gmix_eV_atom": g_mix.tolist()}
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(xlabel=r"$x_{Cu}$", ylabel=r"$G_{mix}/N$ (meV/atom)")
    ax.legend()
    _save_figure(fig, args.output / "mixing_free_energy.png")

    np.savez_compressed(
        args.output / "reconstructed_data.npz",
        temperatures_K=temperatures,
        delta_mu_eV=mus,
        raw_mean_x_cu=heatmap,
        mean_displacement_A=displacement_grid,
        atomic_volume_A3=volume_grid,
    )
    manifest = {
        "source": str(args.input.resolve()),
        "chain_count": len(paths),
        "state_count": len(states),
        "convergence": {
            "criterion": f"split-Rhat <= {args.rhat_limit:g} and per-walker ESS >= {args.minimum_ess:g}",
            "criterion_is_reconstruction_choice": True,
            "all_states_pass": all(item["reconstruction_pass"] for item in diagnostics.values()),
            "states": diagnostics,
        },
        "resolved_curve_temperatures_K": resolved_curve_t,
        "rdf_selection": rdf_selection,
        "mbar": mbar_diagnostics,
        "free_energy_curves": free_energy_curves,
        "conventions": [
            "IAT: Geyer initial-positive adjacent ACF pairs; state stride is ceil(max IAT over 3 traces and 2 walkers).",
            "Raw heatmap: arithmetic mean of the two unthinned production-trace composition means; no interpolation or MBAR.",
            "Displacement: affine ideal fcc sites, periodic minimum image, uniform fractional translation removed.",
            "RDF: closest raw-composition state at nearest available 800 K; unique periodic pairs; instantaneous-volume ideal-mixture normalization; rmax=5.3 A, dr=0.02 A.",
            "MBAR: independent fixed-temperature solve over all sampled delta-mu states; Cu-count histogram is sufficient because fixed-T energy/volume terms cancel.",
            "Gmix: Phi/N=kBT*f/N (one division by N), Legendre-Fenchel max_mu[Phi/N+mu*x], then pure-endpoint subtraction.",
            "Nearest sampled temperature is used when a requested plotting temperature is absent.",
        ],
    }
    (args.output / "diagnostics_and_conventions.json").write_text(
        json.dumps(json_safe(manifest), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"output": str(args.output), "all_states_pass": manifest["convergence"]["all_states_pass"]}))


if __name__ == "__main__":
    main()
