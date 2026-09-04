"""Free-energy analysis helpers for alloy paths."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from torch import Tensor


def cuni_prior_log_density(
    displacement: Tensor,
    log_volume: Tensor,
    sigma_u: Tensor,
    mean_log_volume: Tensor,
    sigma_log_volume: float,
) -> Tensor:
    """Conditioned Cu--Ni Gaussian prior density on the zero-COM subspace."""
    atoms = displacement.shape[-2]
    sigma_u = sigma_u.reshape(-1)
    u_quadratic = displacement.square().sum((-2, -1)) / sigma_u.square()
    u_dimensions = 3 * (atoms - 1)
    u = -0.5 * (u_quadratic + u_dimensions * torch.log(2 * torch.pi * sigma_u.square()))
    variance_v = torch.as_tensor(
        sigma_log_volume**2, dtype=log_volume.dtype, device=log_volume.device
    )
    v = -0.5 * (
        (log_volume - mean_log_volume).square() / variance_v + torch.log(2 * torch.pi * variance_v)
    )
    return u + v


def cuni_terminal_log_density(
    energy: Tensor,
    species: Tensor,
    log_volume: Tensor,
    temperature: Tensor,
    delta_mu: Tensor,
) -> Tensor:
    """Unnormalised semi-grand NPT density, including the ``N log(V)`` Jacobian."""
    from .cuni import KB_EV_K

    beta = 1 / (KB_EV_K * temperature)
    return -beta * (energy - delta_mu * species.eq(0).sum(-1)) + species.shape[-1] * log_volume


def path_weight_estimates(log_weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``log Xi``, normalized weights, and ESS for one trajectory batch."""
    log_weights = log_weights.double()
    normalized = (log_weights - torch.logsumexp(log_weights, 0)).exp()
    return (
        torch.logsumexp(log_weights, 0) - torch.log(
            torch.as_tensor(log_weights.numel(), dtype=log_weights.dtype, device=log_weights.device)
        ),
        normalized,
        normalized.square().sum().reciprocal(),
    )


def include_zero_weight_attempts(log_mean_weight: Tensor, returned: int, attempted: int) -> Tensor:
    """Include discarded all-invalid retry batches as zero-weight samples."""
    if not 0 < returned <= attempted:
        raise ValueError("require 0 < returned <= attempted")
    return log_mean_weight + log_mean_weight.new_tensor(returned / attempted).log()


def recenter_temperature_conditioning(
    state_dict: dict[str, Tensor],
    old_reference: float,
    new_reference: float,
    temperature_min: float = 600.0,
    temperature_max: float = 1200.0,
) -> dict[str, Tensor]:
    """Change inverse-temperature centering without changing model outputs."""
    migrated = {key: value.clone() for key, value in state_dict.items()}
    denominator = 1 / temperature_min - 1 / temperature_max
    feature_shift = (1 / old_reference - 1 / new_reference) / denominator
    weight = migrated["condition.0.weight"]
    migrated["condition.0.bias"] -= weight[:, 3] * feature_shift
    return migrated


def gaussian_path_log_ratio(
    x0: Tensor,
    x1: Tensor,
    forward_velocity: Tensor,
    forward_score: Tensor,
    backward_velocity: Tensor,
    backward_score: Tensor,
    diffusion_squared: Tensor,
    dt: float,
) -> Tensor:
    """Eq. 19 backward/forward Gaussian log-density ratio for one channel."""
    x0, x1 = x0.double(), x1.double()
    forward_velocity, forward_score = forward_velocity.double(), forward_score.double()
    backward_velocity, backward_score = backward_velocity.double(), backward_score.double()
    diffusion_squared = diffusion_squared.double()
    variance = 2 * diffusion_squared * dt
    while variance.ndim < x0.ndim:
        variance = variance.unsqueeze(-1)
    if torch.any(variance <= 0):
        raise ValueError("diffusion variance must be positive")
    forward_mean = x0 + (forward_velocity + diffusion_squared.reshape(
        diffusion_squared.shape + (1,) * (x0.ndim - diffusion_squared.ndim)
    ) * forward_score) * dt
    backward_mean = x1 - (backward_velocity - diffusion_squared.reshape(
        diffusion_squared.shape + (1,) * (x0.ndim - diffusion_squared.ndim)
    ) * backward_score) * dt
    dimensions = tuple(range(1, x0.ndim))
    normalizer = torch.log(2 * torch.pi * variance)
    log_forward = -0.5 * (((x1 - forward_mean).square() / variance) + normalizer)
    log_backward = -0.5 * (((x0 - backward_mean).square() / variance) + normalizer)
    ratio = log_backward - log_forward
    return ratio if not dimensions else ratio.sum(dim=dimensions)


def revealed_token_log_probability(logits: Tensor, tokens: Tensor, revealed: Tensor) -> Tensor:
    """Eq. 17 realized species log-probability, summed per trajectory."""
    selected = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return (selected * revealed).sum(dim=tuple(range(1, tokens.ndim)))


def normalized_path_weights(log_weights, axis: int = -1):
    """Return normalized importance weights and their effective sample size."""
    log_weights = np.asarray(log_weights, dtype=float)
    weights = np.exp(log_weights - logsumexp(log_weights, axis=axis, keepdims=True))
    ess = 1.0 / np.sum(weights**2, axis=axis)
    return weights, ess


normalize_path_weights = normalized_path_weights


def path_weight_ess(log_weights, axis: int = -1):
    """Effective number of paths represented by log importance weights."""
    return normalized_path_weights(log_weights, axis)[1]


def importance_weighted_bar(
    forward_work,
    reverse_work,
    forward_log_weights=None,
    reverse_log_weights=None,
) -> float:
    """Bennett free-energy difference with normalized path importance weights."""
    forward = np.asarray(forward_work, dtype=float)
    reverse = np.asarray(reverse_work, dtype=float)
    if forward.size == 0 or reverse.size == 0:
        raise ValueError("BAR needs forward and reverse work samples")
    fw = normalized_path_weights(
        np.zeros_like(forward) if forward_log_weights is None else forward_log_weights
    )[0]
    rw = normalized_path_weights(
        np.zeros_like(reverse) if reverse_log_weights is None else reverse_log_weights
    )[0]
    if fw.shape != forward.shape or rw.shape != reverse.shape:
        raise ValueError("work and log-weight shapes must match")

    def equation(delta):
        return np.sum(fw * expit(delta - forward)) - np.sum(rw * expit(reverse - delta))

    scale = max(100.0, float(np.max(np.abs(np.r_[forward, reverse]))) + 10.0)
    return float(brentq(equation, -scale, scale))


weighted_bar = importance_weighted_bar


def canonical_ladder_bar(
    forward_delta_u,
    reverse_delta_u,
    beta: float,
    n_atoms: int,
    n_cr: int,
    forward_log_weights=None,
    reverse_log_weights=None,
) -> dict[str, float]:
    """SI canonical neighboring-rung BAR for the edge ``n_cr -> n_cr + 1``."""
    forward = np.asarray(forward_delta_u, dtype=float)
    reverse = np.asarray(reverse_delta_u, dtype=float)
    if forward.ndim != 2 or reverse.ndim != 2:
        raise ValueError("substitution energies must be trajectory-by-site arrays")
    if forward.shape[1] != n_atoms - n_cr or reverse.shape[1] != n_cr + 1:
        raise ValueError("eligible-site counts do not match the composition edge")
    if beta <= 0 or not 0 <= n_cr < n_atoms:
        raise ValueError("require beta > 0 and 0 <= n_cr < n_atoms")
    nf, nr = len(forward), len(reverse)
    if not nf or not nr:
        raise ValueError("both edge directions require samples")
    fw = normalized_path_weights(
        np.zeros(nf) if forward_log_weights is None else forward_log_weights
    )[0]
    rw = normalized_path_weights(
        np.zeros(nr) if reverse_log_weights is None else reverse_log_weights
    )[0]
    factor = (n_atoms - n_cr) / (n_cr + 1)
    forward_work = beta * forward - np.log(factor)
    reverse_work = -beta * reverse - np.log(factor)
    sample_offset = np.log(nr / nf)

    def equation(delta):
        forward_term = nf * np.sum(fw * expit(delta + sample_offset - forward_work).mean(1))
        reverse_term = nr * np.sum(rw * expit(delta + sample_offset - reverse_work).mean(1))
        return forward_term + reverse_term - nr

    scale = max(100.0, float(np.max(np.abs(np.r_[forward_work.ravel(), reverse_work.ravel()]))) + 10)
    delta = float(brentq(equation, -scale, scale))
    log_forward_average = logsumexp(
        np.log(fw) + logsumexp(-beta * forward, axis=1) - np.log(forward.shape[1])
    )
    log_reverse_average = logsumexp(
        np.log(rw) + logsumexp(-beta * reverse, axis=1) - np.log(reverse.shape[1])
    )
    forward_one_sided = float(-np.log(factor) - log_forward_average)
    reverse_one_sided = float(-np.log(factor) + log_reverse_average)
    return {
        "delta_beta_g": delta,
        "forward_one_sided": forward_one_sided,
        "reverse_one_sided": reverse_one_sided,
        "one_sided_discrepancy": forward_one_sided - reverse_one_sided,
        "forward_ess": float(1 / np.square(fw).sum()),
        "reverse_ess": float(1 / np.square(rw).sum()),
    }


def lower_convex_hull(x, free_energy) -> np.ndarray:
    """Indices of the lower convex hull of sampled free energy versus composition."""
    x = np.asarray(x, dtype=float)
    free_energy = np.asarray(free_energy, dtype=float)
    if x.ndim != 1 or x.shape != free_energy.shape or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("x and free_energy must be same-length 1-D arrays with increasing x")
    hull: list[int] = []
    for index in range(x.size):
        while len(hull) > 1:
            left, middle = hull[-2:]
            old_slope = (free_energy[middle] - free_energy[left]) / (x[middle] - x[left])
            new_slope = (free_energy[index] - free_energy[middle]) / (x[index] - x[middle])
            if old_slope < new_slope:
                break
            hull.pop()
        hull.append(index)
    return np.asarray(hull)


def common_tangent(x, free_energy):
    """Return endpoints and slope of the widest sampled lower-hull tie line."""
    x = np.asarray(x, dtype=float)
    free_energy = np.asarray(free_energy, dtype=float)
    hull = lower_convex_hull(x, free_energy)
    gaps = np.diff(hull)
    if not np.any(gaps > 1):
        return None
    position = int(np.argmax(gaps))
    left, right = int(hull[position]), int(hull[position + 1])
    slope = (free_energy[right] - free_energy[left]) / (x[right] - x[left])
    return (float(x[left]), float(x[right]), float(slope))


def binodal(x, free_energy):
    """Return the two sampled coexistence compositions, or ``None``."""
    tangent = common_tangent(x, free_energy)
    return None if tangent is None else tangent[:2]
