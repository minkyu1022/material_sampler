"""Analysis of the two-walker N=108 Cu--Ni reference calculation.

Conventions that are not fixed by the paper are deliberately explicit here:

* autocorrelation times use Geyer's initial-positive-pair truncation;
* a state uses one thinning interval, the ceiling of the largest composition,
  energy, or volume IAT from either independent walker;
* displacement is the minimum-image distance from the affinely scaled ideal
  fcc site after removing the configuration's uniform fractional translation;
* RDFs use unique periodic pairs and the instantaneous cell volume;
* semi-grand MBAR is solved independently at each temperature.  At fixed T,
  energy and volume terms cancel between chemical-potential states, so the
  sufficient statistic is the Cu-count histogram;
* ``Phi/N = k_B T f/N`` is divided by N exactly once before the
  Legendre--Fenchel transform; endpoint subtraction then gives ``G_mix/N``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .cuni import KB_EV_K, build_cuni_fcc

TRACE_KEYS = ("composition", "energy", "volume")


@dataclass(frozen=True)
class Chain:
    path: Path
    temperature: float
    delta_mu: float
    walker: int
    n_atoms: int
    production_start: int
    energy: np.ndarray
    n_cu: np.ndarray
    volume: np.ndarray
    config_sweeps: np.ndarray
    fractional_positions: np.ndarray
    species: np.ndarray
    initial_fractional_positions: np.ndarray | None

    @property
    def production(self) -> slice:
        return slice(self.production_start, None)

    def traces(self) -> dict[str, np.ndarray]:
        selected = self.production
        return {
            "composition": self.n_cu[selected] / self.n_atoms,
            "energy": self.energy[selected],
            "volume": self.volume[selected],
        }


def _metadata(data: np.lib.npyio.NpzFile) -> dict:
    raw = data["metadata"].item()
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(str(raw))


def load_chain(path: str | Path) -> Chain:
    """Load the batched reference schema, with compatibility for old NPZ files."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = _metadata(data)
        energy = np.asarray(data["energy_eV"] if "energy_eV" in data else data["energies"])
        n_atoms = int(meta["n_atoms"])
        if "n_cu" in data:
            n_cu = np.asarray(data["n_cu"], dtype=int)
        else:
            n_cu = np.asarray(data["species"], dtype=int).sum(axis=1)
        if "log_volume" in data:
            volume = np.exp(np.asarray(data["log_volume"], dtype=float))
        else:
            volume = np.abs(np.linalg.det(np.asarray(data["cells"], dtype=float)))
        fractional = (
            np.asarray(data["fractional_positions"], dtype=float)
            if "fractional_positions" in data
            else np.linalg.solve(
                np.asarray(data["cells"], dtype=float).transpose(0, 2, 1),
                np.asarray(data["positions"], dtype=float).transpose(0, 2, 1),
            ).transpose(0, 2, 1)
        )
        species = np.asarray(data["species"], dtype=np.uint8)
        config_sweeps = (
            np.asarray(data["config_sweeps"], dtype=int)
            if "config_sweeps" in data
            else np.arange(len(fractional), dtype=int)
        )
        initial = (
            np.asarray(data["initial_fractional_positions"], dtype=float)
            if "initial_fractional_positions" in data
            else None
        )
    if not (energy.ndim == n_cu.ndim == volume.ndim == 1):
        raise ValueError(f"{path}: scalar traces must be one-dimensional")
    if not (len(energy) == len(n_cu) == len(volume)):
        raise ValueError(f"{path}: scalar trace lengths differ")
    if fractional.shape != species.shape + (3,):
        raise ValueError(f"{path}: fractional_positions/species shapes differ")
    if len(config_sweeps) != len(fractional):
        raise ValueError(f"{path}: config_sweeps/configuration lengths differ")
    # Legacy files contain only already-retained samples; batched files contain
    # the complete sweep trace and explicitly mark the production boundary.
    production_start = int(meta.get("production_start", 0))
    if not 0 <= production_start < len(energy):
        raise ValueError(f"{path}: invalid production_start")
    return Chain(
        path,
        float(meta["temperature_K"]),
        float(meta["delta_mu_Cu_minus_Ni_eV"]),
        int(meta["walker"]),
        n_atoms,
        production_start,
        energy.astype(float),
        n_cu,
        volume,
        config_sweeps,
        fractional,
        species,
        initial,
    )


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    """FFT IAT with Geyer's initial-positive sequence of adjacent ACF pairs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("IAT requires a one-dimensional trace with at least two values")
    centered = values - values.mean()
    variance = float(centered @ centered)
    if variance == 0:
        return 1.0
    size = 1 << (2 * values.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, size)
    acov = np.fft.irfft(spectrum * spectrum.conjugate(), size)[: values.size]
    acov /= np.arange(values.size, 0, -1)
    acf = acov / acov[0]
    paired = acf[1 : 1 + 2 * ((values.size - 1) // 2)].reshape(-1, 2).sum(axis=1)
    stop = np.flatnonzero(paired <= 0)
    positive = paired[: int(stop[0]) if stop.size else len(paired)]
    return max(1.0, float(1.0 + 2.0 * positive.sum()))


def chain_iats(chain: Chain) -> dict[str, float]:
    return {name: integrated_autocorrelation_time(trace) for name, trace in chain.traces().items()}


def state_thinning(chains: list[Chain]) -> int:
    """One conservative sweep interval shared by both walkers at a state."""
    if len(chains) != 2 or {chain.walker for chain in chains} != {0, 1}:
        raise ValueError("a state must contain walkers 0 and 1 exactly once")
    return max(1, int(np.ceil(max(v for chain in chains for v in chain_iats(chain).values()))))


def split_rhat(walker_a: np.ndarray, walker_b: np.ndarray) -> float:
    """Classic split-Rhat for two equal-target independent walkers."""
    length = min(len(walker_a), len(walker_b))
    half = length // 2
    if half < 2:
        return float("nan")
    split = np.stack((walker_a[:half], walker_a[-half:], walker_b[:half], walker_b[-half:]))
    within = float(split.var(axis=1, ddof=1).mean())
    between = half * float(split.mean(axis=1).var(ddof=1))
    if within == 0:
        return 1.0 if between == 0 else float("inf")
    variance = (half - 1) / half * within + between / half
    return float(np.sqrt(variance / within))


def two_walker_diagnostics(chains: list[Chain]) -> dict:
    """Means, split-Rhat, IAT and effective counts for one thermodynamic state."""
    ordered = sorted(chains, key=lambda chain: chain.walker)
    stride = state_thinning(ordered)
    metrics = {}
    for name in TRACE_KEYS:
        traces = [chain.traces()[name] for chain in ordered]
        iats = [integrated_autocorrelation_time(trace) for trace in traces]
        metrics[name] = {
            "walker_means": [float(trace.mean()) for trace in traces],
            "mean_difference": float(abs(traces[0].mean() - traces[1].mean())),
            "split_rhat": split_rhat(*traces),
            "iat_sweeps": iats,
            "effective_samples": [float(len(trace) / tau) for trace, tau in zip(traces, iats)],
        }
    return {"thinning_sweeps": stride, "metrics": metrics}


def thinned_scalar_indices(chain: Chain, interval: int) -> np.ndarray:
    return np.arange(chain.production_start, len(chain.energy), interval, dtype=int)


def thinned_config_indices(chain: Chain, interval: int) -> np.ndarray:
    """Select saved configurations separated by at least the state IAT interval."""
    candidates = np.flatnonzero(chain.config_sweeps >= chain.production_start)
    selected: list[int] = []
    last = -10**18
    for index in candidates:
        sweep = int(chain.config_sweeps[index])
        if sweep - last >= interval:
            selected.append(int(index))
            last = sweep
    return np.asarray(selected, dtype=int)


def mean_lattice_displacements(chain: Chain, indices: np.ndarray) -> np.ndarray:
    """Per-configuration mean displacement from affine ideal fcc lattice sites."""
    reference = chain.initial_fractional_positions
    if reference is None:
        reference = build_cuni_fcc(chain.n_atoms).get_scaled_positions(wrap=False)
    delta = chain.fractional_positions[indices] - reference
    delta -= np.round(delta)
    # A uniform translation is not an internal atomic displacement.
    delta -= delta.mean(axis=1, keepdims=True)
    delta -= np.round(delta)
    lengths = np.cbrt(chain.volume[chain.config_sweeps[indices]])
    return np.linalg.norm(delta * lengths[:, None, None], axis=-1).mean(axis=1)


def averaged_partial_rdf(
    chains: list[Chain], interval: int, *, r_max: float = 5.3, dr: float = 0.02
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Average Cu-Cu, Cu-Ni and Ni-Ni periodic RDFs over selected configurations."""
    bins = round(r_max / dr)
    if not np.isclose(bins * dr, r_max):
        raise ValueError("r_max must be an integer multiple of dr")
    edges = np.linspace(0.0, r_max, bins + 1)
    shell = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
    totals = {"Cu-Cu": np.zeros(bins), "Cu-Ni": np.zeros(bins), "Ni-Ni": np.zeros(bins)}
    used = 0
    for chain in chains:
        for index in thinned_config_indices(chain, interval):
            fractional = chain.fractional_positions[index]
            delta = fractional[:, None] - fractional[None, :]
            delta -= np.round(delta)
            length = np.cbrt(chain.volume[chain.config_sweeps[index]])
            distance = np.linalg.norm(delta * length, axis=-1)
            upper = np.triu_indices(chain.n_atoms, 1)
            pair_distance = distance[upper]
            cu = chain.species[index].astype(bool)
            a, b = cu[upper[0]], cu[upper[1]]
            pair_masks = {
                "Cu-Cu": a & b,
                "Cu-Ni": a ^ b,
                "Ni-Ni": ~a & ~b,
            }
            n_cu = int(cu.sum())
            possible = {
                "Cu-Cu": n_cu * (n_cu - 1) / 2,
                "Cu-Ni": n_cu * (chain.n_atoms - n_cu),
                "Ni-Ni": (chain.n_atoms - n_cu) * (chain.n_atoms - n_cu - 1) / 2,
            }
            volume = float(chain.volume[chain.config_sweeps[index]])
            for name, mask in pair_masks.items():
                if possible[name]:
                    totals[name] += np.histogram(pair_distance[mask], bins=edges)[0] * volume / (
                        possible[name] * shell
                    )
            used += 1
    if not used:
        raise ValueError("no production configurations available for RDF")
    return (edges[:-1] + edges[1:]) / 2, {name: value / used for name, value in totals.items()}


def solve_semigrand_mbar(
    count_histograms: np.ndarray,
    delta_mu: np.ndarray,
    temperature: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, dict]:
    """Solve fixed-T MBAR from rows of Cu-count histograms.

    Histogram columns correspond to ``n_Cu = 0, ..., N``.  This is the exact
    fixed-temperature MBAR equation after cancelling state-independent terms.
    """
    hist = np.asarray(count_histograms, dtype=float)
    mu = np.asarray(delta_mu, dtype=float)
    if hist.ndim != 2 or hist.shape[0] != len(mu) or np.any(hist < 0):
        raise ValueError("histograms must have one nonnegative row per chemical potential")
    sample_counts = hist.sum(axis=1)
    if np.any(sample_counts == 0) or temperature <= 0:
        raise ValueError("every MBAR state needs samples and temperature must be positive")
    total_hist = hist.sum(axis=0)
    present = total_hist > 0
    n_cu = np.arange(hist.shape[1], dtype=float)[present]
    beta_mu_n = mu[:, None] * n_cu[None, :] / (KB_EV_K * temperature)
    scale = float(total_hist.sum())

    def objective(free: np.ndarray) -> tuple[float, np.ndarray]:
        f = np.r_[0.0, free]
        logits = np.log(sample_counts)[:, None] + f[:, None] + beta_mu_n
        log_denominator = logsumexp(logits, axis=0)
        value = (total_hist[present] @ log_denominator - sample_counts @ f) / scale
        probabilities = np.exp(logits - log_denominator)
        gradient = (probabilities @ total_hist[present] - sample_counts) / scale
        return float(value), gradient[1:]

    result = minimize(
        objective,
        np.zeros(len(mu) - 1),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "gtol": tolerance, "ftol": 1e-15, "maxls": 100},
    )
    error = float(np.max(np.abs(objective(result.x)[1]), initial=0.0))
    if not result.success and error > tolerance * 10:
        raise RuntimeError(f"MBAR optimization failed: {result.message} (gradient={error:g})")
    f = np.r_[0.0, result.x]
    return f, {
        "iterations": int(result.nit),
        "max_stationarity_error": error,
        "optimizer_success": bool(result.success),
        "gauge_state": 0,
    }


def mixing_free_energy_from_semigrand(
    f: np.ndarray, delta_mu: np.ndarray, temperature: float, n_atoms: int
) -> tuple[np.ndarray, np.ndarray]:
    """Legendre--Fenchel reconstruct convex ``G_mix/N`` on n/N, in eV/atom."""
    f, delta_mu = np.asarray(f, float), np.asarray(delta_mu, float)
    if f.shape != delta_mu.shape or n_atoms < 1:
        raise ValueError("f/mu shapes must match and n_atoms must be positive")
    x = np.arange(n_atoms + 1, dtype=float) / n_atoms
    phi_per_atom = KB_EV_K * temperature * f / n_atoms  # exactly one division by N
    g_per_atom = np.max(phi_per_atom[:, None] + delta_mu[:, None] * x, axis=0)
    g_mix = g_per_atom - (1 - x) * g_per_atom[0] - x * g_per_atom[-1]
    return x, g_mix


def json_safe(value):
    """Convert NumPy scalars and non-finite diagnostics for strict JSON output."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
